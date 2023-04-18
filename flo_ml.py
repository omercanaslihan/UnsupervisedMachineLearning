##Gözetimsiz Öğrenme ile Müşteri Segmentasyonu##

##İş Problemi##
#FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor. Buna yönelik olarak
#müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.

#Veri Seti Hikayesi#
#Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
#olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

#Görev 1: Veriyi Hazırlama#
#Adım 1: flo_data_20K.csv verisini okutunuz.
df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
df.head()
#Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
#Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max() #2021-05-30
today_date = dt.datetime(2021, 6, 1)

df["tenure"] = (df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")
df["recency"] = (today_date - df["last_order_date"]).astype("timedelta64[D]")
df.head()

model_df = df[["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online", "tenure", "recency"]]
model_df.head()

#Görev 2: K-Means ile Müşteri Segmentasyonu#
#Adım 1: Değişkenleri standartlaştırınız.
ms = MinMaxScaler()
scaled_df = ms.fit_transform(model_df)
model_df = pd.DataFrame(scaled_df, columns=model_df.columns)
model_df.head()
#Adım 2: Optimum küme sayısını belirleyiniz.
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.elbow_value_ #6
#Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
kmeans_model = KMeans(n_clusters=6, random_state=42).fit(model_df)
segments = kmeans_model.labels_
segments

final_df = df[["master_id", "order_num_total_ever_offline", "order_num_total_ever_online", "customer_value_total_ever_online", "customer_value_total_ever_offline",
               'recency', "tenure"]]
final_df["segments"] = segments
final_df.head()
#Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.
final_df.groupby("segments").agg({"order_num_total_ever_online": ["mean", "min", "max"],
                                  "order_num_total_ever_offline": ["mean", "min", "max"],
                                  "customer_value_total_ever_offline": ["mean", "min", "max"],
                                  "customer_value_total_ever_online": ["mean", "min", "max"],
                                  "recency": ["mean", "min", "max"],
                                  "tenure": ["mean", "min", "max"]})
#Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu#
#Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
hc_complete = linkage(model_df, "complete")

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_complete,
                  truncate_mode="lastp",
                  p=10,
                  show_contracted=True,
                  leaf_font_size=10)
plt.axhline(y=1.2, color="r", linestyle="--")

#Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
hc = AgglomerativeClustering(n_clusters=5)
segments = hc.fit_predict(model_df)
segments

final_df = df[["master_id", "order_num_total_ever_offline", "order_num_total_ever_online", "customer_value_total_ever_online", "customer_value_total_ever_offline",
               'recency', "tenure"]]
final_df["segments"] = segments
final_df.head()
#Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.
final_df.groupby("segments").agg({"order_num_total_ever_online": ["mean", "min", "max"],
                                  "order_num_total_ever_offline": ["mean", "min", "max"],
                                  "customer_value_total_ever_offline": ["mean", "min", "max"],
                                  "customer_value_total_ever_online": ["mean", "min", "max"],
                                  "recency": ["mean", "min", "max"],
                                  "tenure": ["mean", "min", "max"]})