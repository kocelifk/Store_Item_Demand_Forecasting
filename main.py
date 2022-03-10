#Store Item Demand Forecasting
#İş Problemi
#Bir mağaza zinciri, 10 farklı mağazası ve 50 farklı ürünü için 3 aylık bir talep tahmini istemektedir.

#Veri Seti HikayesiBu veri seti farklı zaman serisi tekniklerini denemek için sunulmuştur.Bir mağaza zincirinin 5 yıllık verilerinde 10 farklı mağazası ve 50 farklı ürünün bilgileri yer almaktadır.

#Değişkenler
#date–Satış verilerinin tarihi
#Tatil efekti veya mağaza kapanışı yoktur.

#Store –Mağaza ID’si
#Her bir mağazaiçin eşsiz numara.

#Item–Ürün ID’si
#Her bir ürün için eşsiz numara.

#Sales–Satılan ürün sayıları,
#Belirli bir tarihte belirli bir mağazadan satılan ürünlerin sayısı.


#Görev
#Aşağıdaki zaman serisi ve makine öğrenmesi tekniklerini kullanarak ilgili mağaza zinciri için 3 aylık bir talep tahmin modeli oluşturunuz.


#Farklı mağazalarda yer alan farklı ürünlerin üç ay sonrası için tahminlerini gerçekleştirmek
#Üç aylık ürün taleplerini tahmin etmemiz gerekmektedir.

#▪Random Noise
#▪Lag/Shifted Features
#▪Rolling Mean Features
#▪Exponentially Weighted Mean Features
#▪Custom Cost Function(SMAPE)
#▪LightGBM ile Model Validation


import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

########################
# Loading the data
########################

train = pd.read_csv('demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('demand_forecasting/test.csv', parse_dates=['date']) #satış sayıları yok
#
sample_sub = pd.read_csv('demand_forecasting/sample_submission.csv')

df = pd.concat([train, test], sort=False)


#####################################################
# EDA
#####################################################

df["date"].min(), df["date"].max()  #3. ay test setini, 1.ay train setini ifade etmekte

check_df(df) #id na çünkü train setinde id diye bir değişken yok, id sadece test setinde var, sales test setinde yok bu yüzden nan

df[["store"]].nunique()#10 mağaza var
df[["item"]].nunique()#50 ürün var

#mağaza ürün kırılımındaki eşsiz değer sayısı
#her mağazada aynı 50 ürün satılıyor mu buna bakmak amaç

df.groupby(["store"])["item"].nunique() #mağazalara göre groupby aldıktan sonra itemların eşsiz değer sayısına bakılır
#tüm mağazalara 50 eşsiz ürün gitmiş


#her bir mağazada bu ürünlerden kaçar tane satıldığını bulmak için
df.groupby(["store","item"]).agg({"sales":["sum"]})
#store item   #salessum
#1     1      36468.0 #1. mağazadaki 1.üründen toplam 36468 satılmış

#mağaza ürün kırılımında satış istatistiklerinin belirlenmesi
df.groupby(["store","item"]).agg({"sales":["sum","mean","median","std"]})


#####################################################
# FEATURE ENGINEERING
#####################################################

df.head()

def create_date_features(df):
    df['month']= df.date.dt.month
    df['day_of_month'] = df.date.dt.day

    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year

    df['is_wknd'] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df=create_date_features(df)
df.groupby(["store","item","month"]).agg({"sales": ["sum","mean","median","std"]})

########################
# Random Noise
########################
#rastgele gürültü üretecek bir fonksiyon tanımlanması;  belirli bir patternı barındırsın ama aynı zamanda rastsallığını da devam ettirsin

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe)))

#üretilecek lag featureları bağımlı değişken(sales) üzerinden üretilecek
#üretildikten sonra aşırı öğrenmenin önüne geçmek için veri setine rastgele gürültü ekliyorum.
#dataframe'in boyutunda normal dağılımlı bir veri seti oluşturuyoruz.
#satış değişkeni odağındaki featurelara gürültü olarak ekliyor olacağız

#üretilecek olan değişkenler sales üzerinden olacak, bu sebeple bunlar sales ın aynısı olacak


########################
# Lag/Shifted Features
########################
#Bir zaman serisi görevini makine öğrenmesi araçları ile halletmek isteriz.
#Mevsimsellik gibi, trend gibi bileşenleri bir feature olarak nasıl ekleyebiliriz noktasını değerlndirdik


#Geçmiş dönem satış sayılarına ilişkin featurelar türetmek amaç

df.sort_values(by=['store','item','date'],axis=0, inplace=True)
#date featureları ile mevsimselliği yakalaybileceğimiz varsaydık

pd.DataFrame({"sales": df["sales"].values[0:10], #satışın gerçek değerleri  --> yt
              "lag1":df["sales"].shift(1).values[0:10],  #yt-1
              "lag2":df["sales"].shift(2).values[0:10],  #yt-2
              "lag3":df["sales"].shift(3).values[0:10],  #yt-3
              "lag4":df["sales"].shift(4).values[0:10]}) #yt-4 #gecikmeleri hesaplatan shift fonksiyonudur


  #sales  lag1  lag2  lag3  lag4
     #alt satırd 13'den önce bir değer olmadığı için yanındaki na dir.
#0   13.0   NaN   NaN   NaN   NaN #bir gecikmeyi al demek satır açısından konuya bakarak ilgili değerin bir öncesindeki değeri yanına koy demek.
#1   11.0  13.0   NaN   NaN   NaN
#2   14.0  11.0  13.0   NaN   NaN
#3   13.0  14.0  11.0  13.0   NaN
#4   10.0  13.0  14.0  11.0  13.0
#5   12.0  10.0  13.0  14.0  11.0


#Bir zaman serisinin değeri kendisinden önceki değerlerden etkilenir.
#En çok da kendisine en yakın olandan etkilenir.

df.groupby(["store","item"])['sales'].head()
df.groupby(["store","item"])['sales'].transform(lambda x: x.shift(1))

#geçmişe yönelik featurelar türeteceğiz ama 1 gecikme mi olacak 2 gecikmemi olacak yoksa 3 aylık tahminle ilgilenildiği için 90 birimlik
#bir gecikmemi olacak
def lag_features(dataframe, lags):
    for lag in lags:#farklı gecikme değerlerinde gezilsin
        #üretilecek featureların dinamik olarak isimlendirilmesi
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)#featureların üzerine rastgele gürültü eklenmesi
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
#91 birimlik shift alındığında 3 ay öncesine gidilir; 98 birimlik alındığında 3 ayın bir hafta öncesine gibi zaman periyotları girilir

check_df(df)


########################
# Rolling Mean Features
########################
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})

#   sales  roll2      roll3  roll5
#0   13.0    NaN        NaN    NaN #13 kendisi dahil öncesinde 2 değişken barındırmıyor; bu yüzden ortalaması alınamadı
#1   11.0   12.0        NaN    NaN #11 kendisi dahil öncesinde 2 değişken barındırıyor ve ortalaması 12
#2   14.0   12.5  12.666667    NaN #14 kendisi dahil öncesinde 2 değer(14,11) barındırıyor ve ortalaması 12.5; 14 kendisi dahil öncesinde 3 değer(13,11,14) barındırıyor
#3   13.0   13.5  12.666667    NaN
#4   10.0   11.5  12.333333   12.2
#5   12.0   11.0  11.666667   12.0

#Hareketli ortalama feature ları türetmek için ROLLING kullanılır. Kaç adım gidileceğini de window argümanı ifade eder; geçmiş 2 değerin, 3 değerin ortalaması gibi


#Geçmiş bilgiyi barındırmak istiyoruz ama ürettiğimiz featurelarda rolligi yukarıdaki gibi kullandığımızda kendisi
#dahil önceki 2 tane, 3 taneye bakıyor. Bu problemli. Yarının tahminlerini elde etmeye çalışıyoruz diyelim.
#Yarının kendisi dahil geçmiş iki gözlem birimine gidip tahminde bulunma ihtimalimiz yok, çünkü elimizde yarın yok
#Bir diğer yandan geçmiş bilgiyi de yansıtmaya tam olarak uygun değil

#hareketli ortalama feature ını bir gecikme aldıktan sonra türetmemiz lazım
pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})


#   sales  roll2      roll3  roll5
#0   13.0    NaN        NaN    NaN #kendisini dahil etmeden ortalama hesaplaması yapıyoruz
#1   11.0    NaN        NaN    NaN
#2   14.0   12.0        NaN    NaN
#3   13.0   12.5  12.666667    NaN
#4   10.0   13.5  12.666667    NaN
#5   12.0   11.5  12.333333   12.2

#hareketli ortalama featureları türetirken mutlaka shift alınmalı

def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe


df = roll_mean_features(df, [365, 546]) #1 yıl ve 1,5 yıl öncesine ilişkin bilgiyi veriye yansıtmaya çalışıyoruz
#1 yıl önceki değerlerin hareketli ortalaması

########################
# Exponentially Weighted Mean Features
########################

#Ağırlıklı ortalama ile hareketli ortalama arasında fark vardır.
#Ağırlıklı ortalama daha yakındaki gözlemlere ağırlık verecek şekilde çalışıyor ve birçok noktada daha iyi sonuç vermekte.

pd.DataFrame({"sales": df["sales"].values[0:10], #gerçek değer
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10], #hareketli ortalama
              #farklı alfa değerlerine göre üstsel ağırlık ortalamaları getirdik
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm02": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})


#roll2 --> hareketli ortalama

#   sales  roll2     ewm099     ewm095      ewm07      ewm02
#0   13.0    NaN        NaN        NaN        NaN        NaN #kendisinden iki önceki değerin ortalaması yok
#1   11.0    NaN  13.000000  13.000000  13.000000  13.000000 #kendisinden iki önceki değerin ortalaması yok
#2   14.0   12.0  11.019802  11.095238  11.461538  11.947368 #kendisinden iki önceki değerin ortalaması 13+11/2
#3   13.0   12.5  13.970201  13.855107  13.287770  12.704797

#alfa 99 olduğunda en yakına ağırlık verecek, 95 olduğunda biraz daha en yakına az, 07 olduğunda biraz daha az, 02 olduğunda en yakına oldukça az değer veriyor olacak
#alfa değerlerinin sütunlarda düşmesiyle ortalamanın 13e en azından 12ye yakın çıkmasını beklerim

#İki öncesine mi gideceğim üç öncesine mi gideceğim problemi var
#Üstsel ağırlıklı ortalamalar için belirlenmesi gereken bir ortalama olacak
#Gecikme sayısı ve alfaların ne olacağının belirlenmesi lazım

#shift sayısı kullanılacak--> bir öncesine mi iki öncesine mi gideyim yoksa 90 105 gün öncesine mi gideyim gibi gecikmeleri ifade edebilmem lazım
#ve bu gecikmelere ne şekilde ağırlık vermemiz gerektiğini vermemiz lazım yani alfa 99 mu olacak 95 mi olacak 0.5 mi

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas: #en yakın değere ne kadar önem vermem gerektiğini ifade eder alfa
        for lag in lags: #gecikmeler yani lagler 30 gün öncesine mi gideyim 90 gün öncesine mi gideyimi ifade ediyor
            # dinamik isimlendirme
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())  #ilgili alfa ve ilgili gecikmeye göre değişkenlerin üretilmesi
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728] #3 aylık periyoda uygun şekilde oluşturulan gecikmeler

df = ewm_features(df, alphas, lags)
check_df(df)


#alfa 1e ne kadar yakınsa geçmiş en yakındaki değere o kadar ağırlık ver


########################
# One-Hot Encoding
########################
#Veri setinde var olan kategorik değişkenleri one hot encoderdan geçireceğiz
#mağaza 1, mağaza 2 ... mağaza 10 buradaki sayılar büyüklük küçüklük taşıyor
#ama bu mağazaların arasında büyüklük küçüklük ilişkisi yok dolayısıyla bunları one hot encoderdan geçirip
#yeni bir feature olarak türetmek gerekmekte.

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

check_df(df)

########################
# Converting sales to log(1+sales)
#Bağımlı değişken sayısal olduğundan dolayı iterasyon işleminin süresi tahmin edilen değerlerle
#artıkların etkileşimine dayalı olması sebebiyle standartlaştırmaya yönelik olacak.

#Not: Burada standartlaştırmayı bağımsız değişkenlere değil de bağımlı değişkene uyguluyoruz.
#Regresyon problemlerinde tahminleme yaptıktan sonra  eğer bağımlı değşişkenimiz büyük değerlerden oluşuyorsa
#hata metriklerini daha net yorumlayabilmemiz adına değerleri baskılıyoruz. Zorunlu bir işlem değil ancak
#eğer büyük değerlerle uğraşılıyorsa bağımlı değişkeni bu şekilde standartlaştırmak verimli olacaktır.
df['sales'] = np.log1p(df["sales"].values)
#0 değerinin logaritması alınamaz hata verebilir, bu yüzden log1 olası hataların önüne geçmek içindir.
check_df(df)

#####################################################
# Model
#####################################################

########################
# Custom Cost Function
########################
#Regresyon problemlerinde başarımızı ölçmek için kullandığımız metrikler
# MAE, MSE, RMSE, SSE

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

#Bu metriklerden başka bir metriği çalışmamıza dahil edip kullanacak olduğumuz lightgbm modelindeki
#optimizasyon işlemlerimizi bu özelleştirilmiş maliyet fonksiyonuna göre yapıyor olacağız.

#Hem başarımızı değerlendiriyor olacağız hem de lightgbm modelindeki optimizasyon sayısını
#iterasyon sayısını ifade eden bir hiperparametreyi gözlemlemek adına hatalarımızı SMAPE değerine göre
#inceliyor olacağız.

#Symmetric mean absolute percentage error--> Temelinde cost bir diğer ifadesiyle loss fonksiyonlarında
#gerçek değerler ile tahmin ettiğimiz değerler arasındaki farkları inceliyor oluruz. Bu fonksiyonda aynı
#amaca hizmet etmektedir. SMAPE değerinin düşük olması başarılı olduğumuz anlamına gelir.

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False
#lightgbm üzerinde smape fonksiyonunun çağrılması



########################
# Time-Based Validation Sets
########################

#Zamana göre validasyon setlerini oluşturacağız
#LightGBM kullanarak modelleme işlemini gerçekleştireceğiz
#Model valdiasyonu açısından bir train sete bir de test sete ihtiyacımız var

train #2013ten başlayıp 2017'ye kadar devam ediyor  #bu tarih aralığında train işlemi yapacağız
test  #2018'in ilk ayından 2018'in 3. ayına kadar  #tahminde bulunacağız


#train seti normalde

# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df["date"] < "2017-01-01"), :]

# 2017'nin ilk 3'ayı validasyon seti.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]#test için

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales'] #test
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

########################
# LightGBM ile Zaman Serisi Modeli   :Ağaca dayalı bir makine öğrenmesi yöntemi
#Ağaca dayalı yöntemlerin en iyilerinden birisidir.
#LightGBM Gradient Boosting temelli bir optimizasyon yöntemi
########################

# !pip install lightgbm
# conda install lightgbm
# LightGBM parameters
#Tek bir model formunda modeller serisi oluşturmaktadır LightGBM. Her bir iterasyonda bulmuş olduğu artıkları
#önceki tahmin sonuçlarının üzerine eklemekte ya da önceki tahmin sonuçlarından çıkarmaktadır.
#Bu şekilde hata minimum olana kadar belirli bir iterasyon sayısınca optimizasyon işlemleri sürer.
# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,#iterasyon sayısı, optimizasyon sayısı
              'early_stopping_rounds': 200,#aşırı öğrenmenin önüne geçmek için kullanılan hiperparametre
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.

# early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# hem train süresini kısaltır hem de overfit'e engel olur.
# nthread: num_thread, nthread, nthreads, n_jobs

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)#train seti hazırlandı

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)#test seti hazırlandı

model = lgb.train(lgb_params, lgbtrain, #fit gibi
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'], #belirli bir iterasyon sonucunda hatada ilerleme olmazsa ilerlemeyi durdurur; train süresini kısaltır ve aşırı öğrenmenin önüne geçer
                  feval=lgbm_smape,
                  verbose_eval=100)



#aşırı öğrenmnenin önüne model karmaşıklığını optimum noktada bırakarak da geçeriz. optimum noktada bırakmak demek;
#validasyon hatasının eğitim hatasına kıyasla düşmeyi bıraktığı yerde model karmaşıklığını durdurmak demek

#En iyi iterasyon sayısını kullanarak validasyon setinin bağımsız değişkenlerini modele soracağız.
#Daha sonra validasyon setinin bağımlı değişkenini tahmin etmesini isteyeceğiz
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))#logaritmayı geri almak


########################
# Değişken Önem Düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)



test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

########################
# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)


#Tek bir model formunda modeller serisi oluşturmaktadır LightGBM. Her bir iterasyonda bulmuş olduğu artıkları
#önceki tahmin sonuçlarının üzerine eklemekte ya da önceki tahmin sonuçlarından çıkarmaktadır.
#Bu şekilde hata minimum olana kadar belirli bir iterasyon sayısınca optimizasyon işlemleri sürer.
# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,#iterasyon sayısı, optimizasyon sayısı
              'early_stopping_rounds': 200,#aşırı öğrenmenin önüne geçmek için kullanılan hiperparametre
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.

# early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# hem train süresini kısaltır hem de overfit'e engel olur.
# nthread: num_thread, nthread, nthreads, n_jobs

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)#train seti hazırlandı

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)#test seti hazırlandı

model = lgb.train(lgb_params, lgbtrain, #fit gibi
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'], #belirli bir iterasyon sonucunda hatada ilerleme olmazsa ilerlemeyi durdurur; train süresini kısaltır ve aşırı öğrenmenin önüne geçer
                  feval=lgbm_smape,
                  verbose_eval=100)



#aşırı öğrenmnenin önüne model karmaşıklığını optimum noktada bırakarak da geçeriz. optimum noktada bırakmak demek;
#validasyon hatasının eğitim hatasına kıyasla düşmeyi bıraktığı yerde model karmaşıklığını durdurmak demek

#En iyi iterasyon sayısını kullanarak validasyon setinin bağımsız değişkenlerini modele soracağız.
#Daha sonra validasyon setinin bağımlı değişkenini tahmin etmesini isteyeceğiz
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))#logaritmayı geri almak


########################
# Değişken Önem Düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)



test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

########################
# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)


#Tek bir model formunda modeller serisi oluşturmaktadır LightGBM. Her bir iterasyonda bulmuş olduğu artıkları
#önceki tahmin sonuçlarının üzerine eklemekte ya da önceki tahmin sonuçlarından çıkarmaktadır.
#Bu şekilde hata minimum olana kadar belirli bir iterasyon sayısınca optimizasyon işlemleri sürer.
# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,#iterasyon sayısı, optimizasyon sayısı
              'early_stopping_rounds': 200,#aşırı öğrenmenin önüne geçmek için kullanılan hiperparametre
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.

# early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# hem train süresini kısaltır hem de overfit'e engel olur.
# nthread: num_thread, nthread, nthreads, n_jobs

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)#train seti hazırlandı

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)#test seti hazırlandı

model = lgb.train(lgb_params, lgbtrain, #fit gibi
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'], #belirli bir iterasyon sonucunda hatada ilerleme olmazsa ilerlemeyi durdurur; train süresini kısaltır ve aşırı öğrenmenin önüne geçer
                  feval=lgbm_smape,
                  verbose_eval=100)



#aşırı öğrenmnenin önüne model karmaşıklığını optimum noktada bırakarak da geçeriz. optimum noktada bırakmak demek;
#validasyon hatasının eğitim hatasına kıyasla düşmeyi bıraktığı yerde model karmaşıklığını durdurmak demek

#En iyi iterasyon sayısını kullanarak validasyon setinin bağımsız değişkenlerini modele soracağız.
#Daha sonra validasyon setinin bağımlı değişkenini tahmin etmesini isteyeceğiz
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))#logaritmayı geri almak


########################
# Değişken Önem Düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)



test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

########################
# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)


#Tek bir model formunda modeller serisi oluşturmaktadır LightGBM. Her bir iterasyonda bulmuş olduğu artıkları
#önceki tahmin sonuçlarının üzerine eklemekte ya da önceki tahmin sonuçlarından çıkarmaktadır.
#Bu şekilde hata minimum olana kadar belirli bir iterasyon sayısınca optimizasyon işlemleri sürer.
# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,#iterasyon sayısı, optimizasyon sayısı
              'early_stopping_rounds': 200,#aşırı öğrenmenin önüne geçmek için kullanılan hiperparametre
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.

# early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# hem train süresini kısaltır hem de overfit'e engel olur.
# nthread: num_thread, nthread, nthreads, n_jobs

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)#train seti hazırlandı

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)#test seti hazırlandı

model = lgb.train(lgb_params, lgbtrain, #fit gibi
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'], #belirli bir iterasyon sonucunda hatada ilerleme olmazsa ilerlemeyi durdurur; train süresini kısaltır ve aşırı öğrenmenin önüne geçer
                  feval=lgbm_smape,
                  verbose_eval=100)



#aşırı öğrenmnenin önüne model karmaşıklığını optimum noktada bırakarak da geçeriz. optimum noktada bırakmak demek;
#validasyon hatasının eğitim hatasına kıyasla düşmeyi bıraktığı yerde model karmaşıklığını durdurmak demek

#En iyi iterasyon sayısını kullanarak validasyon setinin bağımsız değişkenlerini modele soracağız.
#Daha sonra validasyon setinin bağımlı değişkenini tahmin etmesini isteyeceğiz
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))#logaritmayı geri almak


########################
# Değişken Önem Düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)



test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

########################
# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)


#Tek bir model formunda modeller serisi oluşturmaktadır LightGBM. Her bir iterasyonda bulmuş olduğu artıkları
#önceki tahmin sonuçlarının üzerine eklemekte ya da önceki tahmin sonuçlarından çıkarmaktadır.
#Bu şekilde hata minimum olana kadar belirli bir iterasyon sayısınca optimizasyon işlemleri sürer.
# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,#iterasyon sayısı, optimizasyon sayısı
              'early_stopping_rounds': 200,#aşırı öğrenmenin önüne geçmek için kullanılan hiperparametre
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.

# early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# hem train süresini kısaltır hem de overfit'e engel olur.
# nthread: num_thread, nthread, nthreads, n_jobs

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)#train seti hazırlandı

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)#test seti hazırlandı

model = lgb.train(lgb_params, lgbtrain, #fit gibi
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'], #belirli bir iterasyon sonucunda hatada ilerleme olmazsa ilerlemeyi durdurur; train süresini kısaltır ve aşırı öğrenmenin önüne geçer
                  feval=lgbm_smape,
                  verbose_eval=100)



#aşırı öğrenmnenin önüne model karmaşıklığını optimum noktada bırakarak da geçeriz. optimum noktada bırakmak demek;
#validasyon hatasının eğitim hatasına kıyasla düşmeyi bıraktığı yerde model karmaşıklığını durdurmak demek

#En iyi iterasyon sayısını kullanarak validasyon setinin bağımsız değişkenlerini modele soracağız.
#Daha sonra validasyon setinin bağımlı değişkenini tahmin etmesini isteyeceğiz
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))#logaritmayı geri almak


########################
# Değişken Önem Düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)



test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

########################
# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)


#Tek bir model formunda modeller serisi oluşturmaktadır LightGBM. Her bir iterasyonda bulmuş olduğu artıkları
#önceki tahmin sonuçlarının üzerine eklemekte ya da önceki tahmin sonuçlarından çıkarmaktadır.
#Bu şekilde hata minimum olana kadar belirli bir iterasyon sayısınca optimizasyon işlemleri sürer.
# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,#iterasyon sayısı, optimizasyon sayısı
              'early_stopping_rounds': 200,#aşırı öğrenmenin önüne geçmek için kullanılan hiperparametre
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.

# early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# hem train süresini kısaltır hem de overfit'e engel olur.
# nthread: num_thread, nthread, nthreads, n_jobs

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)#train seti hazırlandı

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)#test seti hazırlandı

model = lgb.train(lgb_params, lgbtrain, #fit gibi
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'], #belirli bir iterasyon sonucunda hatada ilerleme olmazsa ilerlemeyi durdurur; train süresini kısaltır ve aşırı öğrenmenin önüne geçer
                  feval=lgbm_smape,
                  verbose_eval=100)



#aşırı öğrenmnenin önüne model karmaşıklığını optimum noktada bırakarak da geçeriz. optimum noktada bırakmak demek;
#validasyon hatasının eğitim hatasına kıyasla düşmeyi bıraktığı yerde model karmaşıklığını durdurmak demek

#En iyi iterasyon sayısını kullanarak validasyon setinin bağımsız değişkenlerini modele soracağız.
#Daha sonra validasyon setinin bağımlı değişkenini tahmin etmesini isteyeceğiz
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))#logaritmayı geri almak


########################
# Değişken Önem Düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)



test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

########################
# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)


#Tek bir model formunda modeller serisi oluşturmaktadır LightGBM. Her bir iterasyonda bulmuş olduğu artıkları
#önceki tahmin sonuçlarının üzerine eklemekte ya da önceki tahmin sonuçlarından çıkarmaktadır.
#Bu şekilde hata minimum olana kadar belirli bir iterasyon sayısınca optimizasyon işlemleri sürer.
# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,#iterasyon sayısı, optimizasyon sayısı
              'early_stopping_rounds': 200,#aşırı öğrenmenin önüne geçmek için kullanılan hiperparametre
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.

# early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# hem train süresini kısaltır hem de overfit'e engel olur.
# nthread: num_thread, nthread, nthreads, n_jobs

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)#train seti hazırlandı

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)#test seti hazırlandı

model = lgb.train(lgb_params, lgbtrain, #fit gibi
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'], #belirli bir iterasyon sonucunda hatada ilerleme olmazsa ilerlemeyi durdurur; train süresini kısaltır ve aşırı öğrenmenin önüne geçer
                  feval=lgbm_smape,
                  verbose_eval=100)



#aşırı öğrenmnenin önüne model karmaşıklığını optimum noktada bırakarak da geçeriz. optimum noktada bırakmak demek;
#validasyon hatasının eğitim hatasına kıyasla düşmeyi bıraktığı yerde model karmaşıklığını durdurmak demek

#En iyi iterasyon sayısını kullanarak validasyon setinin bağımsız değişkenlerini modele soracağız.
#Daha sonra validasyon setinin bağımlı değişkenini tahmin etmesini isteyeceğiz
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))#logaritmayı geri almak


########################
# Değişken Önem Düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)



test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

########################
# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)


#Tek bir model formunda modeller serisi oluşturmaktadır LightGBM. Her bir iterasyonda bulmuş olduğu artıkları
#önceki tahmin sonuçlarının üzerine eklemekte ya da önceki tahmin sonuçlarından çıkarmaktadır.
#Bu şekilde hata minimum olana kadar belirli bir iterasyon sayısınca optimizasyon işlemleri sürer.
# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'num_boost_round': 1000,#iterasyon sayısı, optimizasyon sayısı
              'early_stopping_rounds': 200,#aşırı öğrenmenin önüne geçmek için kullanılan hiperparametre
              'nthread': -1}

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.

# early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# hem train süresini kısaltır hem de overfit'e engel olur.
# nthread: num_thread, nthread, nthreads, n_jobs

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)#train seti hazırlandı

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)#test seti hazırlandı

model = lgb.train(lgb_params, lgbtrain, #fit gibi
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  early_stopping_rounds=lgb_params['early_stopping_rounds'], #belirli bir iterasyon sonucunda hatada ilerleme olmazsa ilerlemeyi durdurur; train süresini kısaltır ve aşırı öğrenmenin önüne geçer
                  feval=lgbm_smape,
                  verbose_eval=100)



#aşırı öğrenmnenin önüne model karmaşıklığını optimum noktada bırakarak da geçeriz. optimum noktada bırakmak demek;
#validasyon hatasının eğitim hatasına kıyasla düşmeyi bıraktığı yerde model karmaşıklığını durdurmak demek

#En iyi iterasyon sayısını kullanarak validasyon setinin bağımsız değişkenlerini modele soracağız.
#Daha sonra validasyon setinin bağımlı değişkenini tahmin etmesini isteyeceğiz
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val))#logaritmayı geri almak


########################
# Değişken Önem Düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):  #hangi değişkenlerin daha önemli olabileceği ile ilgili bilgi
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp



plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################

train = df.loc[~df.sales.isna()]#test setinde sales değişkeninde na vardır
#na yoksa sales değişkeninde train seti
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': 0,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)

test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

########################
# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

submission_df.to_csv("submission_demand.csv", index=False)

