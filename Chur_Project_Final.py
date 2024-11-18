
#### IMPORTING THE LIBRARIES ####

import seaborn as sns
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report,  RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report,  RocCurveDisplay
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics as met


##### ADJUSTING THE DISPLAY SETTINGS #####
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows',  None)
pd.set_option('display.max_colwidth', None)


#### LOADING THE DATASET ####
df = pd.read_csv("Final Project/Churntelco.csv", delimiter=';')

df = df.drop(["Unnamed: 0.1", "Unnamed: 0", "subscriber_sk"], axis=1)

########## SAYISAL OLUP OBJECT VERI TIPINDE OLAN KOLONLARI TEMIZLEMEK ############
# hucrelerdeki veriyi sayisala cevirir, sayisal olmayan yani '27Eyl veya 123.456.789.234.567.897 gibi veriyi NaN yapar

columns = ["inv_data_overage_1m", "inv_data_overage_6m_avg", "inv_data_overage_to_due_amount_6m_ratio", "cxi_voice_traffic_4g_avg_1m", "inv_due_amount_1m", "cxi_voice_traffic_2g_avg_1m", "cxi_voice_retainability_4g_avg_1m", "inv_due_amount_6m_min", "demog_cvs_6m_std", "demog_cvs_1m_3m_ratio", "usg_voice_onnet_cnt_6m_avg", "prod_addon_cnt_6m_avg", "inv_due_amount_1m_3m_ratio", "cxi_voice_quality_4g_avg_1m","cxi_voice_quality_4g_avg_3m_avg", "cxi_voice_retainability_2g_avg_3m_avg", "cxi_voice_retainability_3g_avg_3m_avg", "cxi_voice_traffic_3g_avg_3m_avg", "cxi_voice_quality_3g_avg_1m", "cxi_voice_accessibility_4g_avg_3m_avg", "cxi_voice_accessibility_4g_avg_1m", "cxi_dxi_4g_avg_1m", "cxi_dxi_avg_1m", "cxi_data_traffic_3g_avg_1m", "cxi_data_tput_4g_avg_1m", "cxi_data_tput_3g_avg_3m_avg", "cxi_data_speed_3g_avg_1m", "cxi_data_speed_3g_avg_3m_avg", "cxi_data_tput_3g_avg_1m", "cxi_data_speed_4g_avg_1m", "cxi_data_speed_4g_avg_3m_avg", "cxi_voice_retainability_4g_avg_3m_avg"]
for col in columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df["new_tenure"] = df.apply(lambda x: 1 if pd.isnull(x["mnp_total_tenure_org"]) else 0, axis=1)


#### Hedef Degiskenin Dagilimi #####
df["port_out"].value_counts()

# Sınıflar için sari/kirmizi
colors = ["yellow", "red"]

sns.countplot(
    data=df,
    x="port_out",
    hue="port_out",
    palette=colors,
    dodge=False,
    legend=False
)

plt.title("Target Variable Distribution")
plt.xlabel("Classes")
plt.ylabel("Observation Count")
plt.show()




df.shape

#### SUMMARRY ####
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
   numeric_df = dataframe.select_dtypes(include=[np.number])
   print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df, head=2)

df.head()

#### IDENTIFYING AND CATEGORIZING THE FEATURES #####
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
       car_th: int, optional
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


   """
   # cat_cols, cat_but_car
   cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
   num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
   cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
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

cat_cols
num_cols

df[cat_but_car].dtypes

df.head(10)



######## OUTLIER ANALYSIS #######
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
   print(col, check_outlier(df, col))



##### CATEGORICAL VARIABLE ANALYSIS ####
def cat_summary(dataframe, cat_cols, plot=False):
   for col_name in cat_cols:
       print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                           "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
       print("##########################################")
       if plot:
           sns.countplot(x=dataframe[col_name], data=dataframe)
           plt.show()

cat_summary(df, cat_cols)



#### NUMERICAL VARIABLE ANALYSIS #####
def num_summary(dataframe, numerical_col, plot=False):
   quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
   print(dataframe[numerical_col].describe(quantiles).T)


   if plot:
       dataframe[numerical_col].hist(bins=20)
       plt.xlabel(numerical_col)
       plt.title(numerical_col)
       plt.show()


for col in num_cols:
   num_summary(df, col, plot=False)




##### TARGET VS CATEGORICAL VARIABLE ANALYSIS ######
def target_summary_with_cat(dataframe, target, categorical_col):
   print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col, observed=False)[target].mean()}), end="\n\n\n")


for col in cat_cols:
   target_summary_with_cat(df, "port_out", col)




#### TARGET VS NUMERICAL FEATURE ANALYSIS #####
def target_summary_with_num(dataframe, target, numerical_col):
   print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
   target_summary_with_num(df, "port_out", col)




######## DATA PROCESSING  ########
#### OUTLIER CAPPING ####
for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

for col in num_cols:
   print(col, check_outlier(df, col))


##### MISSING VALUES #####
def missing_values_table(dataframe, na_name=False):
   na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
   n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
   ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
   missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
   print(missing_df, end="\n")

   if na_name:
       return na_columns

na_cols = missing_values_table(df, True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)




#### MISSING VALUES VS TARGET ####
def missing_vs_target(dataframe, target, na_columns):
   temp_df = dataframe.copy()

   for col in na_columns:
       temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

   na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

   for col in na_flags:
       print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                           "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "port_out", na_cols)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

######## MISSING VALUE FEATURES THAT HAVE SIGNIFICANT EFFECT ON TARGET #########
def find_significant_na_flags(dataframe, target, na_columns, threshold=0.02):
    significant_na_columns = []  # Anlamlı sütunları kaydetmek için liste
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

        # Hedef değişken üzerindeki ortalama farkı kontrol et
        target_mean = temp_df.groupby(col + '_NA_FLAG')[target].mean()
        difference = abs(target_mean[1] - target_mean[0])

        # Eşik değerinden büyükse anlamlı kabul et
        if difference > threshold:
            significant_na_columns.append(col)

    return significant_na_columns

# Eksik değere sahip sütunların listesini otomatik oluşturma
na_columns = df.columns[df.isnull().any()].tolist()

# Hedef değişken üzerindeki etkisi yüksek olan sütunları belirleyin
significant_na_columns = find_significant_na_flags(df, "port_out", na_columns, threshold=0.02)
print("Anlamlı etkisi olan sütunlar:", significant_na_columns)

len(significant_na_columns)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#### Creating NA_FLAG columns for features that have significant impact on target ########
for col in significant_na_columns:
    df[col + '_NA_FLAG'] = np.where(df[col].isnull(), 1, 0)

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# FILLING OUT THE MISSING VALUES
def filling_out_missing_values(dataframe, significant_na_columns, target="port_out"):
    target_values = dataframe[target]
    dataframe = dataframe.drop(columns=[target])

    # Eksik değeri olan sütunların listesini oluşturma
    missing_variables = [col for col in significant_na_columns if dataframe[col].isnull().sum() > 0]

    # Kategorik sütunlar için mod (en sık görülen değer) ile doldurma
    for col in dataframe.select_dtypes(include=['object']).columns:
        if col in missing_variables:
            dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)

    # Sayısal sütunlar için ortanca (median) ile doldurma
    for col in dataframe.select_dtypes(exclude=['object']).columns:
        if col in missing_variables:
            dataframe[col].fillna(dataframe[col].median(), inplace=True)

    # Hedef değişkeni tekrar veri setine ekleme
    dataframe[target] = target_values

    return dataframe

# Fonksiyonu veri setinin tamamına uygulama
df = filling_out_missing_values(df, significant_na_columns)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.isnull().sum()

############## hedef degiskende etkisi olmayan ve eksik degeri olan degiskenleri sil ##########
insignificant_na_columns = [
    col for col in df.columns
    if col not in significant_na_columns and df[col].isnull().sum() > 0
]
df.drop(columns=insignificant_na_columns, inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
# eksik degerleri tekrar kontrol et
df.isnull().sum().sum()

df.head()

##### CORRELATION MATRIX #####
def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.05, annot_kws={'size': 1}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

correlation_matrix(df, num_cols)

# degisken veri tipi  siniflandirmasini tekrar cagir
cat_cols, num_cols, cat_but_car = grab_col_names(df)
#cat_but_car'i kontrol et
cat_but_car

# prod_tariff_name', 'demog_city' degiskenleri cat_cols'a manuel ata. #grab_col_names i tekrar cagirdiginda bu islemleri tekrar yap.
cat_cols.extend(['prod_tariff_name', 'demog_city'])

# cat_but_car listesinden prod_tariff_name', 'demog_city' cikar. #grab_col_names i tekrar cagirdiginda bu islemleri tekrar yap.
cat_but_car = [col for col in cat_but_car if col not in ['prod_tariff_name', 'demog_city']]
cat_but_car




#########################################################################################################
####### BASE MODEL #######
#Copying the dataframe to df_base for base model
df_base = df.copy()

#Removing the target variable from cat_cols to exclude it from encoding process
cat_cols = [col for col in cat_cols if col not in ["port_out"]]

#Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

df_base = one_hot_encoder(df_base, cat_cols)

df_base.head()

#Scaling
#num_cols = [col for col in num_cols if col not in ["subscriber_sk"]]
sc = StandardScaler()
df_base[num_cols] = sc.fit_transform(df_base[num_cols])

df_base.head()

#establishing the model
y = df_base["port_out"]
X = df_base.drop(["port_out"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=24)

X_train.shape
X_test.shape

models = [('LR', LogisticRegression(random_state=24, max_iter=1500)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=24)),
          ('RF', RandomForestClassifier(random_state=24)),
          ('XGB', XGBClassifier(random_state=24)),
          ("LightGBM", LGBMClassifier(verbose=-1,random_state=24, force_col_wise=True)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=24))]

# a list that we can store the performance results of the model
results = []

# creating a for loop on models
for model_name, model in models:
    # training the model
    model.fit(X_train, y_train)

    # prediction on Test dataset
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # calculating the metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # stroring the results
    results.append({
        'Model':  model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc
    })

# converting the results' list to a dataframe
import pandas as pd

result_df = pd.DataFrame(results)
result_df



#### BASE MODEL RESULTS #####
# Out[16]:
#       Model  Accuracy  Precision    Recall  F1 Score       AUC
# 0        LR    0.9040   0.520833  0.128866  0.206612  0.727894
# 1       KNN    0.8995   0.379310  0.056701  0.098655  0.601733
# 2      CART    0.8340   0.200000  0.237113  0.216981  0.567615
# 3        RF    0.9040   0.750000  0.015464  0.030303  0.710512
# 4       XGB    0.9035   0.516129  0.082474  0.142222  0.708926
# 5  LightGBM    0.9065   0.629630  0.087629  0.153846  0.736035
# 6  CatBoost    0.9065   0.652174  0.077320  0.138249  0.742619

# BASE MODEL GRAPH
#BASE MODEL RESULTS

# Verileri hazırlama
models = result_df['Model']
metrics = result_df.columns[1:]  # İlk sütun model isimleri, diğer sütunlar metrikler
num_metrics = len(metrics)

# Grupların (her model) arasındaki boşluk ve bar genişliği
bar_width = 0.1
index = np.arange(len(models))

# Grafiği oluşturma
plt.figure(figsize=(12, 8))

# Her bir metrik için bir bar grubu oluşturma
for i, metric in enumerate(metrics):
    plt.bar(index + i * bar_width, result_df[metric], width=bar_width, label=metric)

# X eksenine model isimlerini yerleştirme
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison (Grouped Bar Chart)')
plt.xticks(index + bar_width * (num_metrics - 1) / 2, models, rotation=45)

# Gösterge ve açıklama ekleme
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()
#######################################################################################################


##### FEATURE ENGINEERING & INTERACTION ########
# Kullanım ve Tarife Uyumluluğu
#df['new_usage_tariff_accept_ratio'] = df['ev_offer_accept_avg_within_tariff_1m']
# Çağrı ve İnternet Kullanımı Kombinasyonu
df['new_combined_call_data_usage'] = (df['cxi_voice_traffic_2g_avg_1m'] + df['cxi_voice_traffic_4g_avg_1m'] + df['cxi_voice_traffic_3g_avg_3m_avg'] ) / 3

# Sadakat veya Bağlılık Skoru

df['new_mnp_tenure_hot_call']=  df['mnp_total_tenure_org'] * df['hot_call_cnt_1m']
df['new_demog_tenure_hot_call']=  df['demog_customer_tenure'] * df['hot_call_cnt_1m']

####### ekelmeler
df['new_cxi_data_retainability'] = (df['cxi_data_retainability_3g_avg_3m_avg'] + (df['cxi_data_retainability_4g_avg_3m_avg'] )) /2
df['new_inv_data_usage_abroad'] = df['inv_data_usage_abroad_6m_avg'] * 0.75 + (df['inv_data_usage_abroad_1m'] * 0.25 )


df["new_ticket_growth_1m_to_3m"] = round((df["ticket_info_tariff_cnt_1m"] - (df["ticket_info_tariff_cnt_3m"] / 3)),2)

#son eklenen
# Bölme işlemi için 1 eklenerek sıfırdan kaçınılır

# mnp_total_tenure_org değeri ile sna_mnp_churn_sum değerini birleştirerek,
# uzun süre bir operatörde kalıp daha sonra taşıma yapan müşterilerin churn riskini analiz edecegim.
# Kategorik değişkeni sayısal forma dönüştürme
df['new_mnp_op_sequence_numeric'] = pd.Categorical(df['mnp_op_sequence']).codes
# Yeni özellik oluşturma
df['new_mnp_op_sequence_to_tenure_ratio'] = df['new_mnp_op_sequence_numeric'] / (df['mnp_total_tenure_org'] + 1)

# sna_mnp_churn_sum değeriyle mnp_op_sequence değerini oranlayarak, taşıma işlemi sonrasında churn yapma riskini gösteren oran elde edecgim.
df['new_churn_post_mnp_ratio'] = df['sna_mnp_churn_sum'] / (df['new_mnp_op_sequence_numeric'] + 1)

#mnp_total_tenure_org değeri ile sna_mnp_churn_sum değerini birleştirerek,
# uzun süre bir operatörde kalıp daha sonra taşıma yapan müşterilerin churn riskini analiz edecegim
# Çarpma işlemlerinde sıfırları ele almak için koşullu ifade kullanımı
df['new_tenure_churn_score'] = np.where(
    (df['mnp_total_tenure_org'] == 0) | (df['sna_mnp_churn_sum'] == 0),
    0,
    df['mnp_total_tenure_org'] * df['sna_mnp_churn_sum']
)
#mnp_op_sequence ve sna_mnp_churn_sum etkileşimi ile müşteri taşıma işlemleri ve churn eğilimini birleştirerek
# yüksek taşıma ve yüksek churn durumlarını tanimliyorum

df['new_mnp_op_churn_interaction'] = np.where(
    (df['new_mnp_op_sequence_numeric'] == 0) | (df['sna_mnp_churn_sum'] == 0),
    0,
    df['new_mnp_op_sequence_numeric'] * df['sna_mnp_churn_sum']
)

df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# prod_tariff_name', 'demog_city' degiskenleri cat_cols'a manuel ata. #grab_col_names i tekrar cagirdiginda bu islemleri tekrar yap.
cat_cols.extend(['prod_tariff_name', 'demog_city'])

# cat_but_car listesinden prod_tariff_name', 'demog_city' cikar. #grab_col_names i tekrar cagirdiginda bu islemleri tekrar yap.
cat_but_car = [col for col in cat_but_car if col not in ['prod_tariff_name', 'demog_city']]



##### SCALING ####
# num_cols = [col for col in num_cols if col not in ["subscriber_sk"]]
sc = StandardScaler()
df[num_cols] = sc.fit_transform(df[num_cols])


##### RARE ENCODING ######
#RARE ANALYSER
def rare_analyser(dataframe, target, cat_cols):
   for col in cat_cols:
       print(col, ":", len(dataframe[col].value_counts()))
       print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                           "RATIO": dataframe[col].value_counts() / len(dataframe),
                           "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "port_out", cat_cols)

#RARE ENCODER

#Sadece sehirlere uyguladigim frekans (count) bazli rare encoding
def rare_encoder_count_based(dataframe, column, count_threshold=300):
    """
    Count'a dayalı rare encoding yapan fonksiyon.

    Args:
    dataframe (pd.DataFrame): Veri seti.
    column (str): Kategorik değişkenin ismi.
    count_threshold (int): Rare için eşik değer (default: 350).

    Returns:
    pd.DataFrame: Rare encoding uygulanmış veri seti.
    """
    temp_df = dataframe.copy()
    # Frekansı threshold'un altında olan kategoriler 'Rare' olarak adlandırılır
    value_counts = temp_df[column].value_counts()
    rare_labels = value_counts[value_counts < count_threshold].index
    temp_df[column] = np.where(temp_df[column].isin(rare_labels), 'Rare', temp_df[column])
    return temp_df

# fonksiyonu cagir
df = rare_encoder_count_based(df, column='demog_city', count_threshold=300)



# Sadece tarife paketleri icin encoding
# Bu fonskiyon hem count hem de target mean degerine gore rare encoding yapar tarifeler uzerinde
def rare_encoder(dataframe, target, column, count_threshold=20, target_mean_threshold=0.05):
    temp_df = dataframe.copy()

    # Tarifelerin Count ve Target Mean hesaplamaları
    tmp = temp_df.groupby(column)[target].agg(['mean', 'size']).rename(
        columns={'mean': 'Target_Mean', 'size': 'Count'})
    tmp['Ratio'] = tmp['Count'] / len(temp_df)

    # Nadir kategorilere atama koşulu
    rare_labels = tmp[(tmp['Count'] < count_threshold) & (tmp['Target_Mean'] < target_mean_threshold)].index
    temp_df[column] = np.where(temp_df[column].isin(rare_labels), 'Rare', temp_df[column])

    return temp_df

# Kullanım örneği
df = rare_encoder(df, target='port_out', column='prod_tariff_name', count_threshold=20, target_mean_threshold=0.05)



# Diğer kategorik değişkenlere genel rare encoding
def general_rare_encoder(dataframe, target, rare_perc=0.01, exclude_cols=None):
    """
    Tüm kategorik değişkenlere, belirli bir orana göre rare encoding uygular.

    Args:
    dataframe (pd.DataFrame): Veri seti.
    target (str): Hedef değişken ismi.
    rare_perc (float): Rare kategori için oran eşiği.
    exclude_cols (list): Rare encoding işleminden hariç tutulacak sütunlar (default: None).

    Returns:
    pd.DataFrame: Rare encoding uygulanmış veri seti.
    """
    temp_df = dataframe.copy()
    exclude_cols = exclude_cols or []

    # Hariç tutulacak kolonlar dışındaki tüm kategorik değişkenleri seç
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and col not in exclude_cols
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        # Kategoriler için rare encoding
        tmp = temp_df.groupby(var)[target].agg(['mean', 'size']).rename(
            columns={'mean': 'Target_Mean', 'size': 'Count'})
        tmp['Ratio'] = tmp['Count'] / len(temp_df)

        # Rare kategorilerini belirleme
        rare_labels = tmp[tmp['Ratio'] < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


# Şehir dışındaki diğer tüm kategorik değişkenler için genel rare encoding
df = general_rare_encoder(df, target='port_out', rare_perc=0.01, exclude_cols=['demog_city', 'prod_tariff_name'])


#####  ONE HOT ENCODING ENCODING ####
cat_cols = [col for col in cat_cols if col not in ["port_out"]]
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

df = one_hot_encoder(df, cat_cols)



df.head()

df.isnull().sum().sum()

### Original feature list before applying PCA
original_features = df.drop(["port_out"], axis=1).columns.tolist()
len(original_features) #length of it 440 (without the target)
#Out[66]: 369

##### MODELLING WITH PCA #####
# Hedef değişken ve özelliklerin ayrılması
y = df["port_out"]
X = df.drop(["port_out"], axis=1)

pca = PCA()
pca_fit = pca.fit_transform(X)

cov_matrix = np.cov(X.T)

pca.explained_variance_
pca.explained_variance_ratio_
pca.components_
np.cumsum(pca.explained_variance_ratio_)


# PCA bileşen yüklemeleri ile orijinal özelliklerin ilişkisi
pca_importance_df = pd.DataFrame(pca.components_, columns=original_features)
print("PCA importance shape:", pca_importance_df.shape)
#PCA importance shape: (369, 369)


##### OPTIMAL COMPONENT GRAPH
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısını")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show()
####################
pca = PCA(n_components=300)
pca_fit = pca.fit_transform(X)

pca.explained_variance_
pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)

############
pca_df = pd.DataFrame(data=pca_fit, columns=[f'PC{i+1}' for i in range(300)])
pca_df['port_out'] = y.values  # Hedef değişkeni PCA sonucuna ekleme


pca_df.head()

######## veri setini iki degiskene indirip gorsellestirmek
# def plot_pca(dataframe, target):
#    fig = plt.figure(figsize=(7, 5))
#    ax = fig.add_subplot(1, 1, 1)
#    ax.set_xlabel('PC1', fontsize=15)
#    ax.set_ylabel('PC2', fontsize=15)
#    ax.set_title(f'{target.capitalize()} ', fontsize=20)
#
#
#    targets = list(dataframe[target].unique())
#    colors = random.sample(['r', 'b', "g", "y"], len(targets))
#
#
#    for t, color in zip(targets, colors):
#        indices = dataframe[target] == t
#        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
#    ax.legend(targets)
#    ax.grid()
#    plt.show()
#
# plot_pca(pca_df, 'port_out')

########### BASLANGIC MODELLERI  ################
X_pca = pca_df.drop(['port_out'], axis=1)
y_pca = pca_df['port_out']

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y_pca, test_size=0.2, random_state=24)


# Modeller listesi
models = [
    ('Logistic Regression', LogisticRegression(random_state=24, max_iter=2000, class_weight='balanced')),
    ('LightGBM', LGBMClassifier(random_state=24, force_col_wise=True, class_weight='balanced', learning_rate=0.01, num_leaves=100, verbose=-1)),
    ('Random Forest', RandomForestClassifier(random_state=24, class_weight='balanced', n_estimators=200, max_depth=10)),
    ('XGBoost', XGBClassifier(random_state=24, scale_pos_weight=(y_train_pca.value_counts()[0] / y_train_pca.value_counts()[1]), learning_rate=0.01, n_estimators=100, max_depth=3)),
    ('CatBoost', CatBoostClassifier(verbose=0, random_state=24, class_weights=[1, (y_train_pca.value_counts()[0] / y_train_pca.value_counts()[1])])),
]


# Performans sonuçlarını saklamak için liste
results = []

for model_name, model in models:
    # Modeli eğitme
    model.fit(X_train_pca, y_train_pca)

    # Test veri setinde tahmin
    y_pred = model.predict(X_test_pca)
    y_prob = model.predict_proba(X_test_pca)[:, 1]

    # Metrikleri hesaplama
    accuracy = accuracy_score(y_test_pca, y_pred)
    precision = precision_score(y_test_pca, y_pred)
    recall = recall_score(y_test_pca, y_pred)
    f1 = f1_score(y_test_pca, y_pred)
    auc = roc_auc_score(y_test_pca, y_prob)

    # Sonuçları saklama
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc
    })

# Sonuçları bir DataFrame olarak görüntüleme
result_df = pd.DataFrame(results)
result_df

#sonuclar
# Out[15]:
# Out[71]:
#                  Model  Accuracy  Precision    Recall  F1 Score       AUC
# 0  Logistic Regression    0.7335   0.191257  0.541237  0.282638  0.724444
# 1             LightGBM    0.9015   0.492228  0.489691  0.490956  0.801913
# 2        Random Forest    0.9025   0.478261  0.056701  0.101382  0.675289
# 3              XGBoost    0.8230   0.289474  0.567010  0.383275  0.761127
# 4             CatBoost    0.9190   0.629032  0.402062  0.490566  0.809952

######## BASLANGIC MODELLERI GRAFIK ##########
import matplotlib.pyplot as plt
import numpy as np

# Verileri hazırlama
models = result_df['Model']
metrics = result_df.columns[1:]  # İlk sütun model isimleri, diğer sütunlar metrikler
num_metrics = len(metrics)

# Grupların (her model) arasındaki boşluk ve bar genişliği
bar_width = 0.1
index = np.arange(len(models))

# Grafiği oluşturma
plt.figure(figsize=(12, 8))

# Her bir metrik için bir bar grubu oluşturma
for i, metric in enumerate(metrics):
    plt.bar(index + i * bar_width, result_df[metric], width=bar_width, label=metric)

# X eksenine model isimlerini yerleştirme
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison (Grouped Bar Chart)')
plt.xticks(index + bar_width * (num_metrics - 1) / 2, models, rotation=45)

# Gösterge ve açıklama ekleme
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()


########################################################################################################### Oldukca yavas calisiyo bu part
#### OPTIMIZASYON  ve STACKING + ENSEMBLE + STRATIFIED K FOLD

# PCA ile özelliklerin ayrılması
X_pca = pca_df.drop(['port_out'], axis=1)
y_pca = pca_df['port_out']

# Eğitim ve bağımsız test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_pca, y_pca, test_size=0.2, random_state=24, stratify=y_pca)

# Hiperparametre aralıkları
param_grid = {
    'CatBoost': {
        'learning_rate': [0.01, 0.05, 0.08, 0.1],
        'depth': [4, 6, 8, 10],
        'class_weights': [[1, y_train.value_counts()[0] / y_train.value_counts()[1]]]
    },
    'LightGBM': {
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 75],
        'max_depth': [5, 10, 15],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    },
    'XGBoost': {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'scale_pos_weight': [y_train.value_counts()[0] / y_train.value_counts()[1]],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
}

# Stratified K-Fold ve SMOTE ayarları
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
smote = SMOTE(sampling_strategy=0.7, random_state=24)

# Modelleri optimize etme ve en iyi hiperparametreleri seçme
best_estimators = {}
for model_name, param in param_grid.items():
    print(f"Şu anda {model_name} modelinin optimizasyonu yapılıyor...")
    if model_name == 'CatBoost':
        model = CatBoostClassifier(verbose=0, random_state=24)
    elif model_name == 'LightGBM':
        model = LGBMClassifier(verbose=-1, random_state=24)
    elif model_name == 'XGBoost':
        model = XGBClassifier(random_state=24)

    random_search = RandomizedSearchCV(
        model, param_distributions=param, scoring='recall', cv=stratified_kfold, n_iter=10, random_state=24, n_jobs=-1
    )

    # SMOTE ve Cross-Validation süreci
    cv_results = []
    for train_idx, val_idx in stratified_kfold.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # SMOTE uygulama
        X_res, y_res = smote.fit_resample(X_train_fold, y_train_fold)

        # Modeli eğit ve değerlendirme skorunu sakla
        random_search.fit(X_res, y_res)
        y_val_pred = random_search.predict(X_val_fold)
        cv_results.append(recall_score(y_val_fold, y_val_pred))

    best_estimators[model_name] = random_search.best_estimator_
    print(f"{model_name} için en iyi parametreler: {random_search.best_params_}")
    print(f"{model_name} için Cross-Validation Recall: {np.mean(cv_results)}")

# Stacking Ensemble oluşturma
print("Stacking Ensemble modeli eğitiliyor...")
stacking_model = StackingClassifier(
    estimators=[
        ('CatBoost', best_estimators['CatBoost']),
        ('LightGBM', best_estimators['LightGBM']),
        ('XGBoost', best_estimators['XGBoost'])
    ],
    final_estimator=LogisticRegression(max_iter=1000, class_weight='balanced', random_state=24),
    cv=stratified_kfold,
    n_jobs=-1
)

# Stacking modelini eğitme
stacking_model.fit(X_train, y_train)

# Bağımsız test setinde tahmin yapma
y_pred = stacking_model.predict(X_test)
y_prob = stacking_model.predict_proba(X_test)[:, 1]

# Performans metrikleri
results = {
    'Model': 'Stacking Ensemble (CatBoost + LightGBM + XGBoost)',
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'AUC': roc_auc_score(y_test, y_prob)
}

print("Bağımsız Test Seti Ensemble Model Sonuçları:")
print(results)
results_df = pd.DataFrame(results)
results_df

# # Sonuclar
# CatBoost için en iyi parametreler: {'learning_rate': 0.01, 'depth': 6, 'class_weights': [1, 9.0]}
# LightGBM için en iyi parametreler: {'subsample': 1.0, 'num_leaves': 75, 'max_depth': 10, 'learning_rate': 0.1, 'colsample_bytree': 0.8}
# XGBoost için en iyi parametreler: {'subsample': 0.6, 'scale_pos_weight': 9.0, 'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.01, 'colsample_bytree': 0.6}
# {'Model': 'Stacking Ensemble (CatBoost + LightGBM + XGBoost)', 'Accuracy': 0.7865, 'Precision': 0.27524752475247527, 'Recall': 0.695, 'F1 Score': 0.3943262411347518, 'AUC': 0.8394194444444443}
# {'Model': 'Stacking Ensemble (CatBoost + LightGBM + XGBoost)',
#  'Accuracy': 0.7865,
#  'Precision': 0.27524752475247527,
#  'Recall': 0.695,
#  'F1 Score': 0.3943262411347518,
#  'AUC': 0.8394194444444443}





######### FINAL AGIRLIKLI MODEL STRATIFIED K FOLD VE SMOTE KURULUMU EN IYI PARAMETRELERLE #########
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np

# Veri seti ve hedef değişkenin ayrılması
X = pca_df.drop(['port_out'], axis=1)
y = pca_df['port_out']

# Stratified K-Fold tanımlama
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

# SMOTE tanımlama
smote = SMOTE(sampling_strategy=0.15, random_state=24)

# Modellerin ağırlıkları
weights = {
    'XGBoost': 0.35,  # Hafif düşür
    'CatBoost': 0.4,  # Recall odaklı olduğu için artır
    'LightGBM': 0.1,
    'Logistic Regression': 0.15  # Daha az etkili olduğu için düşür
}

# PCA tanımlama
pca = PCA(n_components=300)

# K-Fold içinde sonuçları tutmak için listeler
all_cm = []  # Confusion Matrix'leri tutmak için
blended_probs = []  # Tüm fold'lardan olasılıkları tutmak için
blended_preds = []  # Tahmin edilen sınıflar
y_tests = []  # Gerçek sınıflar

# Stratified K-Fold cross-validation ile eğitim ve test işlemleri
for train_index, test_index in stratified_kfold.split(X, y):
    # Eğitim ve test setlerinin oluşturulması
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # PCA ve SMOTE işlemleri
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)

    # Modelleri tanımlama
    xgb_model = XGBClassifier(
        scale_pos_weight=9.0,
        learning_rate=0.01, n_estimators=150, max_depth=5, subsample=0.6, colsample_bytree=0.6,
        random_state=24
    )
    catboost_model = CatBoostClassifier(
        verbose=0, learning_rate=0.01, depth=6, class_weights=[1, 9.0], random_state=24
    )
    lgbm_model = LGBMClassifier(
        learning_rate=0.1, num_leaves=75, max_depth=10, subsample=1.0, colsample_bytree=0.8, random_state=24
    )
    logreg_model = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced', random_state=24)

    # Modelleri eğitme
    xgb_model.fit(X_train_resampled, y_train_resampled)
    catboost_model.fit(X_train_resampled, y_train_resampled)
    lgbm_model.fit(X_train_resampled, y_train_resampled)
    logreg_model.fit(X_train_resampled, y_train_resampled)

    # Tahminler
    y_pred_xgb = xgb_model.predict_proba(X_test_pca)[:, 1]
    y_pred_cat = catboost_model.predict_proba(X_test_pca)[:, 1]
    y_pred_lgbm = lgbm_model.predict_proba(X_test_pca)[:, 1]
    y_pred_logreg = logreg_model.predict_proba(X_test_pca)[:, 1]

    blended_prob = (
        weights['XGBoost'] * y_pred_xgb +
        weights['CatBoost'] * y_pred_cat +
        weights['LightGBM'] * y_pred_lgbm +
        weights['Logistic Regression'] * y_pred_logreg
    )
    blended_pred = (blended_prob > 0.42).astype(int)

    # Performans ölçümü
    blended_probs.extend(blended_prob)
    blended_preds.extend(blended_pred)
    y_tests.extend(y_test_fold)

    # Confusion Matrix
    cm = confusion_matrix(y_test_fold, blended_pred)
    all_cm.append(cm)

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test_fold, blended_prob)
    plt.title("ROC Curve (Fold Bazında)")
    plt.show(block=True)

# Fold'lar arasında ortalama Confusion Matrix
avg_cm = np.mean(all_cm, axis=0)
print("Average Confusion Matrix:")
print(avg_cm)

# K-Fold genel metrikleri
print("\nCross-Validation Genel Performans:")
print(f"ROC AUC: {roc_auc_score(y_tests, blended_probs):.4f}")
print(f"Accuracy: {accuracy_score(y_tests, blended_preds):.4f}")
print(f"Precision: {precision_score(y_tests, blended_preds):.4f}")
print(f"Recall: {recall_score(y_tests, blended_preds):.4f}")
print(f"F1 Score: {f1_score(y_tests, blended_preds):.4f}")

# Final model eğitimi (Tüm veri setiyle)
print("Final modeli eğitiliyor...")
X_resampled, y_resampled = smote.fit_resample(X, y)

xgb_model.fit(X_resampled, y_resampled)
catboost_model.fit(X_resampled, y_resampled)
lgbm_model.fit(X_resampled, y_resampled)
logreg_model.fit(X_resampled, y_resampled)

def blended_prediction(X_new):
    # Modellerin tahminleri
    y_pred_xgb = xgb_model.predict_proba(X_new)[:, 1]
    y_pred_cat = catboost_model.predict_proba(X_new)[:, 1]
    y_pred_lgbm = lgbm_model.predict_proba(X_new)[:, 1]
    y_pred_logreg = logreg_model.predict_proba(X_new)[:, 1]

    # Ağırlıklı tahminlerin hesaplanması
    blended_prob = (
        weights['XGBoost'] * y_pred_xgb +
        weights['CatBoost'] * y_pred_cat +
        weights['LightGBM'] * y_pred_lgbm +
        weights['Logistic Regression'] * y_pred_logreg
    )

    # Threshold belirleyerek sınıflandırma
    return np.where(blended_prob > 0.42, 1, 0), blended_prob

print("Final model eğitimi tamamlandı.")

# Confusion Matrix için grafik
avg_cm = np.mean(all_cm, axis=0)  # Fold'lar arasında ortalama Confusion Matrix

plt.figure(figsize=(8, 6))
sns.heatmap(avg_cm, annot=True, fmt=".1f", cmap="Blues", cbar=False,
            xticklabels=["Non-Churn", "Churn"], yticklabels=["Non-Churn", "Churn"])
plt.title("Average Confusion Matrix (Cross-Validation)", fontsize=14)
plt.xlabel("Predicted Labels", fontsize=12)
plt.ylabel("True Labels", fontsize=12)
plt.tight_layout()
plt.show(block=True)


# ROC Curve için grafik
RocCurveDisplay.from_predictions(y_tests, blended_probs)
plt.title("ROC Curve (Cross-Validation Overall)", fontsize=14)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Chance Line')  # Diagonal çizgi
plt.legend()
plt.show(block=True)


# SONUCLAR;

# CV Sonuclari;
# Average Confusion Matrix:
# [[1254.2  545.8]
#  [  46.6  153.4]]

# Cross-Validation Genel Performans:
# ROC AUC: 0.8135
# Accuracy: 0.7038
# Precision: 0.2194
# Recall: 0.7670
# F1 Score: 0.3412
########################################################
# Asagidaki test seti sonuclarini buraya da ekledim, farki kolay gorebilelim diye
# Test Seti Sonuclari;
# Confusion Matrix:
# [[1322  484]
#  [  48  146]]

# Test Seti Performansi
# Precision: 0.232
# Recall: 0.753
# F1 Score: 0.354
# Accuracy: 0.734
# ROC AUC: 0.812

####### CROSS VALIDATION SONUCLARININ GRAFIGI #######
# sonuclari sozluk yapisinda tutuyoruz
results_ensemble_cv = {
    'Model': 'Blended Ensemble',
    'Accuracy': 0.703,
    'Precision': 0.219,
    'Recall': 0.767,
    'F1 Score': 0.354,
    'AUC': 0.813
}

#DataFrame'e çevirme
result_df_final_cv = pd.DataFrame([results_ensemble_cv])

# Verileri hazırlama
models = result_df_final_cv['Model']
metrics = result_df_final_cv.columns[1:]  # İlk sütun model isimleri, diğer sütunlar metrikler
num_metrics = len(metrics)

# Grupların (her model) arasındaki boşluk ve bar genişliği
bar_width = 0.15
index = np.arange(len(models))

# Grafiği oluşturma
plt.figure(figsize=(12, 8))

# Her bir metrik için bir bar grubu oluşturma ve değerleri ekleme
for i, metric in enumerate(metrics):
    bars = plt.bar(index + i * bar_width, result_df_final_cv[metric], width=bar_width, label=metric)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, yval,
            f'{metric}\n{yval:.2f}', ha='center', va='bottom', fontsize=8
        )

# X eksenine model isimlerini yerleştirme
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison With Cross Validation SKF (Grouped Bar Chart)')
plt.xticks(index + bar_width * (num_metrics - 1) / 2, models, rotation=0)

# Gösterge ve açıklama ekleme
plt.legend(title="Metrics")
plt.tight_layout()
plt.show(block=True)







######## FINAL MODEL'IN TEST SETI UZERINDE TEST EDILMESI, TAHMINLER YAPILMASI ###########
# Veri setinin eğitim ve test olarak ayrılması
X = pca_df.drop(['port_out'], axis=1)
y = pca_df['port_out']

# %80 eğitim, %20 test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

# PCA ve SMOTE işlemlerini yalnızca eğitim setine uygulayın, ardından test setine aynı PCA dönüşümünü uygulayın.
pca = PCA(n_components=300)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# SMOTE yalnızca eğitim verisine uygulanır
smote = SMOTE(sampling_strategy=0.15, random_state=24)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)

# Modelleri yeniden eğitme
xgb_model.fit(X_train_resampled, y_train_resampled)
catboost_model.fit(X_train_resampled, y_train_resampled)
lgbm_model.fit(X_train_resampled, y_train_resampled)
logreg_model.fit(X_train_resampled, y_train_resampled)

# blended_prediction fonksiyonu tanımlanır
def blended_prediction(X):
    prob_xgb = xgb_model.predict_proba(X)[:, 1]
    prob_catboost = catboost_model.predict_proba(X)[:, 1]
    prob_lgbm = lgbm_model.predict_proba(X)[:, 1]
    prob_logreg = logreg_model.predict_proba(X)[:, 1]

    blended_prob = (0.35 * prob_xgb + 0.4 * prob_catboost + 0.1 * prob_lgbm + 0.15 * prob_logreg)
    y_pred = (blended_prob > 0.42).astype(int)
    return y_pred, blended_prob

# Tahmin yapma
y_pred, y_prob = blended_prediction(X_test_pca)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix of Test:")
print(cm)

# Classification Report
print("\nClassification Report of Test:")
print(classification_report(y_test, y_pred))

# ROC AUC Score
auc = roc_auc_score(y_test, y_prob)
print(f"\nROC AUC Score of Test: {auc}")

# ROC Curve
disp = RocCurveDisplay.from_predictions(y_test, y_prob)
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Chance Line')  # Diagonal çizgi
plt.legend()
plt.title("ROC Curve of Final Model on Test Set")
plt.show(block=True)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Non-Churn", "Churn"], yticklabels=["Non-Churn", "Churn"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix of Final Model On The Test Set")
plt.show(block=True)

########## FINAL MODEL'IN TEST SETI UZERINDE PERFORMANS METRIKLERININ HESAPLANMASI #########
auc = roc_auc_score(y_test, y_prob)

# Performans metriklerini hesaplama
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Sonuçları yazdırma
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"Accuracy: {accuracy:.3f}")
print(f"AUC: {auc:.3f}")

#Test Seti Sonuclari
# Confusion Matrix:
# [[1322  484]
#  [  48  146]]

# Precision: 0.232
# Recall: 0.753
# F1 Score: 0.354
# Accuracy: 0.734
# AUC: 0.812



########## TEST SONUCLARININ GRAFIGI  #########

# sonuclari sozluk yapisinda tutuyoruz
results_ensemble = {
    'Model': 'Blended Ensemble',
    'Accuracy': 0.734,
    'Precision': 0.232,
    'Recall': 0.753,
    'F1 Score': 0.352,
    'AUC': 0.812
}

#DataFrame'e çevirme
result_df_final = pd.DataFrame([results_ensemble])

# Verileri hazırlama
models = result_df_final['Model']
metrics = result_df_final.columns[1:]  # İlk sütun model isimleri, diğer sütunlar metrikler
num_metrics = len(metrics)

# Grupların (her model) arasındaki boşluk ve bar genişliği
bar_width = 0.15
index = np.arange(len(models))

# Grafiği oluşturma
plt.figure(figsize=(12, 8))

# Her bir metrik için bir bar grubu oluşturma ve değerleri ekleme
for i, metric in enumerate(metrics):
    bars = plt.bar(index + i * bar_width, result_df_final[metric], width=bar_width, label=metric)
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, yval,
            f'{metric}\n{yval:.2f}', ha='center', va='bottom', fontsize=8
        )

# X eksenine model isimlerini yerleştirme
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance On the Test Set')
plt.xticks(index + bar_width * (num_metrics - 1) / 2, models, rotation=0)

# Gösterge ve açıklama ekleme
plt.legend(title="Metrics")
plt.tight_layout()
plt.show(block=True)








###### ENSEMBLE MODEL FEATURE IMPORTANCE ######

# Bu grafik, PCA ile kurdugumuz nsemble modelin  her bir orijinal özelliğe ne derece önem verdigini göstermektedir.
# Ancak bu önem dereceleri, doğrudan modelleme sürecindeki özelliklerin değil,
# PCA kullanılarak oluşturulmuş bileşenler üzerinden elde edilen ve bu bileşenlerin orijinal özelliklere geri dönüştürülmesiyle elde edilen bir özetidir.

# Modellerin ağırlıkları
weights = {'XGBoost': 0.7, 'CatBoost': 0.1, 'LightGBM': 0.1, 'Logistic Regression': 0.1}

# Modellerden alınan feature importance değerleri
xgb_importances = xgb_model.feature_importances_
cat_importances = catboost_model.get_feature_importance()
lgbm_importances = lgbm_model.feature_importances_
logreg_importances = np.abs(logreg_model.coef_[0])  # Logistic Regression için abs(coef_)

# Ensemble PCA feature importance
ensemble_pca_importances = (
    weights['XGBoost'] * xgb_importances +
    weights['CatBoost'] * cat_importances +
    weights['LightGBM'] * lgbm_importances +
    weights['Logistic Regression'] * logreg_importances
)

# PCA işlemini uygularken bileşen sayısını 400 olarak ayarlayın
pca = PCA(n_components=300)
pca_fit = pca.fit_transform(X)
components = pca.components_  # (400, orijinal özellik sayısı)

# Orijinal özelliklerin önem derecesini almak için bileşen yüklemeleriyle çarpma işlemi yapıyoruz
original_importances = np.abs(ensemble_pca_importances @ components)

# Orijinal değişken isimlerini 400 özellik ile sınırlıyoruz
original_features = original_features[:300]

# Orijinal özelliklere göre önem derecesini içeren DataFrame oluşturma
original_importance_df = pd.DataFrame({
    "Feature": original_features,
    "Importance": original_importances
}).sort_values(by="Importance", ascending=False)

# İlk 50 önemli değişken
top_50_features = original_importance_df.head(50)

print(top_50_features)

# İlk 20 değişken için bar grafiği
plt.figure(figsize=(12, 8))
plt.barh(top_50_features["Feature"], top_50_features["Importance"], color="skyblue")
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("Top 20 Features by Ensemble Model (Original Features)", fontsize=14)
plt.gca().invert_yaxis()  # En önemli değişken yukarıda görünsün
plt.tight_layout()
plt.show(block=True)




################# ACTIONABLE INSIGHTS ################
# modelimizden churn olasılık skorlarını aliyoruz
_, y_prob = blended_prediction(X_test_pca)

# Segmentasyon için eşik değerlerini belirle
df_test = pd.DataFrame({'Churn_Probability': y_prob, 'True_Label': y_test})  # Test setini bir DataFrame'e çevir
df_test.head()
df_test['Risk_Group'] = pd.cut(
    df_test['Churn_Probability'],
    bins=[0, 0.5, 0.75, 1],  # eşik değerleri: düşük, orta, yüksek risk
    labels=['Düşük Risk', 'Orta Risk', 'Yüksek Risk']
)

# Segment dağılımını kontrol et
print(df_test['Risk_Group'].value_counts())

#Gorsellestir
# Risk gruplarının dağılımını al
risk_counts = df_test['Risk_Group'].value_counts()

# Bar grafiği
colors = ['green', 'orange', 'red']
ax = risk_counts.plot(kind='bar', color=colors)

# Veri etiketlerini ekleme
for i, count in enumerate(risk_counts):
    ax.text(i, count + 5,  # X ve Y koordinatları
            str(count),    # Etiketin içeriği
            ha='center', va='bottom', fontsize=10, fontweight='bold')  # Stil ayarları

# Grafik başlık ve etiketler
plt.title('Risk Gruplarına Göre Müşteri Dağılımı')
plt.ylabel('Müşteri Sayısı')
plt.xlabel('Risk Grupları')
plt.tight_layout()
plt.show()





############ PIRAMIT ##########
# Churn olasılık skorlarından gelen müşteri segmentleri
risk_counts = df_test['Risk_Group'].value_counts()

# Funnel için müşteri sayıları
customer_counts = risk_counts.values
risk_groups = risk_counts.index

# Funnel grafiği
fig, ax = plt.subplots(figsize=(6, 8))
colors = ['green', 'orange', 'red']

for i, (label, count, color) in enumerate(zip(risk_groups[::-1], customer_counts[::-1], colors[::-1])):
    width = count / max(customer_counts)
    top = len(risk_groups) - i
    bottom = top - 1
    left = (1 - width) / 2
    right = (1 + width) / 2
    ax.fill_betweenx([bottom, top], [left, left], [right, right], color=color, edgecolor='black')
    ax.text(right + 0.02, top - 0.5, f"{label}\n{count}", ha='left', va='center', fontsize=10, color='black', fontweight='bold')  # Sağda etiket

ax.set_xlim(0, 1.2)  # Sağ tarafa yer açmak için limitleri genişlet
ax.set_ylim(0, len(risk_groups))
ax.axis('off')
ax.set_title("Risk Gruplarına Göre Funnel Analizi", fontsize=14, fontweight='bold')

plt.show(block=True)





################ BASKA GOSTERIM ##########

# Churn olasılık skorlarından gelen müşteri segmentleri
risk_counts = df_test['Risk_Group'].value_counts()

# Funnel için müşteri sayıları
customer_counts = risk_counts.values[::-1]
risk_groups = risk_counts.index[::-1]

# renkler
colors = ['red', 'orange', 'green']

# Grafik oluşturma
fig, ax = plt.subplots(figsize=(6, 8))
bars = plt.barh(risk_groups, customer_counts, color=colors)

# Etiketleri dışarı yazma
for bar, count in zip(bars, customer_counts):
    ax.text(count + 10, bar.get_y() + bar.get_height()/2,
            f"{count}", va='center', ha='left', fontsize=10)

# Başlık ve eksen isimleri
plt.title("Risk Gruplarına Göre Funnel Analizi", fontsize=14, weight='bold')
plt.xlabel("Müşteri Sayısı", fontsize=12)
plt.ylabel("Risk Grupları", fontsize=12)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show(block=True)

