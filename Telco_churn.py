#############################################
# Telco Churn Feature Engineering
#############################################

#############################################
# İş Problemi / Business Problem
#############################################
# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.
#  Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir

# # It is desirable to develop a machine learning model that can predict customers who will leave the company.
# Before developing the model, the necessary data analysis and feature engineering steps are expected of you.

#############################################
# Veri Seti Hikayesi / Dataset Story
#############################################
# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali
# bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu
# gösterir


# 21 Değişken 7043 Gözlem 977.5 KB
# CustomerId: Müşteri İd’si
# Gender: Cinsiyet
# SeniorCitizen: Müşterinin yaşlı olup olmadığı (1, 0)
# Partner: Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents: Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır)
# tenure: Müşterinin şirkette kaldığı ay sayısı
# PhoneService: Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines: Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService: Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity: Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup: Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection: Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport: Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV: Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies: Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract: Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling: Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod: Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges: Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges: Müşteriden tahsil edilen toplam tutar
# Churn: Müşterinin kullanıp kullanmadığı (Evet veya Hayır)


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error




pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#############################################
# PROJE GÖREVLERİ / PROJECT TASKS
#############################################

#############################################
# GÖREV 1: Keşifçi Veri Analizi / Exploratory Data Analysis
#############################################

def load_customer_churn():
    data = pd.read_csv("C:/Users/Lenovo/PycharmProjects/datasets/TelcoChurn/Telco-Customer-Churn.csv")
    return data


df = load_customer_churn()
df.head()

# Adım 1: Genel resmi inceleyiniz. / Overview


def df_summary(df):
    print("\n" + 20 * "*" + "SHAPE".center(20) + 20 * "*")
    print("\n")
    print(df.shape)
    print("\n" + 20 * "*" + "INDEX".center(20) + 20 * "*")
    print("\n")
    print(df.index)
    print("\n" + 20 * "*" + "COLUMNS".center(20) + 20 * "*")
    print("\n")
    print(df.columns)
    print("\n" + 20 * "*" + "DATAFRAME INFORMATIONS".center(20) + 20 * "*")
    print("\n")
    print(df.info())
    print("\n"+ 20 * "*" + "DATAFRAME INFORMATIONS".center(20) + 20 * "*")
    print("\n")
    print(df.describe().T)
    print("\n" + 20 * "*" + "MISSING VALUES".center(20) + 20 * "*")
    print(df.isnull().sum())

df_summary(df)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.dtypes
df['SeniorCitizen'] = df['SeniorCitizen'].astype("object")  # Korelasyon analizinde problem çıkıyor. Bu değişken de geliyor o yüzden object yapıldı

# Adım 2: Numerik ve kategorik değişkenleri yakalayınız. / Identify the numerical and categorical variables.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız. / Analyze the numerical and categorical variables.

# Kategorik değişken analizi / Analysis of Categorical variables

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)


# Numerik değişken analizi / Analysis of Numerical variables

def num_summary(dataframe, num_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_cols].describe(quantiles).T)

    if plot:
        dataframe[num_cols].hist(bins=20)
        plt.xlabel(num_cols)
        plt.title(num_cols)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col)

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması) / Analyze of the Target variable

df["Churn"].value_counts()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Churn oranını pasta graği ile görelim / Let's visualize the target variable.

fig, ax = plt.subplots(figsize = (10,10))
labels = 'No Churn', 'Churn'
x = df.groupby('Churn').size().values
ax.pie(x, autopct='%1.1f%%', labels=labels)
plt.show()

# Kategorik değişkenlere göre Target analizi / Target analysis by categorical variables

def target_summary_cat(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


target_summary_cat(df, "Churn", cat_cols)

# Grafiksel inceleme / Visualize of the categorical variables

for cat in cat_cols:
    fig, ax = plt.subplots(figsize=(10, 10))
    churn_no = []
    churn_yes = []
    x = []
    for i in df[cat].unique():
        # each category has a sub-category: e.g. Gender category has male and female
        # looping through each subcategory and adding churn and no churn data to list
        churn_no.append(df.groupby([cat, 'Churn']).size()[i][0])
        churn_yes.append(df.groupby([cat, 'Churn']).size()[i][1])
        x.append(i)
    print(churn_no, churn_yes)

    p1 = plt.bar(x, churn_no)
    p2 = plt.bar(x, churn_yes, bottom=churn_no)

    # Plotting the bar labels inside the bars, as percentage
    for r1, r2, in zip(p1, p2):
        height1 = r1.get_height()
        height2 = r2.get_height()
        plt.text(r1.get_x() + r1.get_width() / 2.,  # x
                 height1 / 2.,  # y
                 f'{round(height1 / (height1 + height2) * 100, 1)} %',  # s
                 ha="center", va="center", color="white", fontsize=12)
        plt.text(r2.get_x() + r2.get_width() / 2.,  # x
                 height1 + height2 / 2.,  # y
                 f'{round(height2 / (height1 + height2) * 100, 1)} %',  # s
                 ha="center", va="center", color="white", fontsize=12)

    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Number of Customers in Category', fontsize=12)
    plt.legend(['No Churn', 'Churn'])
    plt.title(cat, fontsize=16)
    plt.show()


# Numeric değişkenlere göre Target analizi / Target analysis by numerical variables

def target_summary_num(dataframe, target, num_cols):
    print(dataframe.groupby(target).agg({num_cols: "mean"}), end="\n\n")


for col in num_cols:
    target_summary_num(df,"Churn",col)


df["tenure"].nunique()
len(df["tenure"].unique())

# Grafiksel inceleme / Visualize
for col in num_cols:
    df.groupby('Churn').agg({col:'mean'}).plot(kind='bar', rot = 0,figsize=(16,8))
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
df.tenure[df["Churn"] == "1"].hist(bins=20)
df.tenure[df["Churn"] == "0"].hist(bins=20, alpha = 0.5)
plt.legend(["Churn Customers", "Non-Churn Customers"])
plt.title("Customer Tenure")
plt.xlabel("Tenure")
plt.ylabel("Count of Customers")
plt.show(block=True)

# MonthlyCharges
fig, ax = plt.subplots(figsize = (10,10))
df.tenure[df.Churn == "Yes"].hist(bins=20)
df.tenure[df.Churn == "No"].hist(bins=20, alpha = 0.5)
plt.legend(["Churn Customers", "Non-Churn Customers"])
plt.title("Customer Monthly Charges")
plt.xlabel("Monthly Charge Amount")
plt.ylabel("Count of Customers")


# TotalCharges
fig, ax = plt.subplots(figsize = (10,10))
df.tenure[df.Churn == "Yes"].hist(bins=20)
df.tenure[df.Churn == "No"].hist(bins=20, alpha = 0.5)
plt.legend(["Churn Customers", "Non-Churn Customers"])
plt.title("Customer Total Charges")
plt.xlabel("Total Charge Amount")
plt.ylabel("Count of Customers")


# Adım 5: Aykırı gözlem analizi yapınız. / Check the outliers 

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name): 
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))


# Adım 6: Eksik gözlem analizi yapınız. / Check the missing values 

def missing_values_table(dataframe, na_name=False): 
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)  # aylık ödenecek miktarlarıyla totalcharge doldurulailir (daya iyi olur denensin)  veya 11 değişken drop edilebilir

df.isnull().sum()

# Korelesyon analizinde tenure ile pozitif korelasyon gösteriyor dolayısı ile düşürme durumunda etkisi az.
# Silmek için etkilerini araştır (medyan ile değişim şiddetle öneriliyor)

##################################
# BASE MODEL KURULUMU
##################################
dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
dff = one_hot_encoder(dff, cat_cols, drop_first=True)


y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred,y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 4)}")

# Accuracy: 0.7899
# Recall: 0.6464
# Precision: 0.5
# F1: 0.5639
# Auc: 0.7372


# Adım 7: Korelasyon analizi yapınız.
num_cols
df[num_cols].corr()

# Korelasyon Matrisi
corr_matrix = df[num_cols].corr()

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

fig, ax = plt.subplots()
heatmap = ax.imshow(corr_matrix, interpolation='nearest', cmap=cm.coolwarm)

# colorbar düzenlenmesi
cbar_min = corr_matrix.min().min()
cbar_max = corr_matrix.max().max()
cbar = fig.colorbar(heatmap, ticks=[cbar_min, cbar_max])

# etiketler oluşturulması
labels = ['']
for column in num_cols:
    labels.append(column)
    labels.append('')
ax.set_yticklabels(labels, minor=False)
ax.set_xticklabels(labels, minor=False)


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(), annot=True)
plt.show()

# TotalChargers'in aylık ücretler ve tenure ile yüksek korelasyonlu olduğu görülmekte
df.corrwith(df["Churn"]).sort_values(ascending=False)


#############################################
# GÖREV 2: Özellik Mühendisliği / Feature Engineering
#############################################
# Adım 1: Yeni değişkenler oluşturunuz.

# Tüm sütunları ve sınıfların unique değerlerini bir arada görmek adına;
for column in df.columns:
    print(f"Column: {column} --> Unique Values: {df[column].unique()}")


# Tenure  değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"] >= 0) & (df["tenure"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"] > 12) & (df["tenure"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"] > 24) & (df["tenure"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"] > 36) & (df["tenure"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"] > 48) & (df["tenure"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"] > 60) & (df["tenure"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "Yes") or (x["DeviceProtection"] != "Yes") or (x["TechSupport"] != "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1)

# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1)

df.head()
df.shape

##################################
# ENCODING
##################################
# Değişkenlerin tiplerine göre ayrılması işlemi

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# LABEL ENCODING

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "object" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)


df.head()
df.dtypes

# Kategorik değilkenleri sayısallaştırdığımız için bi daha korelasyon inceleyebiliriz.

def heatMap(df):
    #Create Correlation df
    corr = df.corr()
    #Plot figsize
    fig, ax = plt.subplots(figsize=(15, 15))
    #Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(corr, cmap="Blues", annot=True, fmt=".2f", linewidths=.2)
    #Apply xticks
    plt.xticks(range(len(corr.columns)), corr.columns);
    #Apply yticks
    plt.yticks(range(len(corr.columns)), corr.columns)
    #show plot
    plt.show()

heatMap(df)

# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
# Sınıfı 2 den fazla olan değişkenlere one hot encoding yapalım.
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn", "NEW_TotalServices"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.dtypes


# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])
df.head()
df.dtypes


# Adım 5: Model oluşturunuz.
# Model oluşturma esnasında datasetin orijinal hali ile base bir model oluşturmak gerekmektedir. Karşılaştırma açısından


y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.79
# Recall: 0.64
# Precision: 0.49
# F1: 0.56
# Auc: 0.73

# Base Model
# Accuracy: 0.7899
# Recall: 0.6464
# Precision: 0.5
# F1: 0.5639
# Auc: 0.7372


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(catboost_model, X_train)

df.columns

