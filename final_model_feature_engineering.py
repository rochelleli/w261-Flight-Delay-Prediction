# Databricks notebook source
# Import PySpark feature engineering libraries
from pyspark.sql import Window
from pyspark.sql.functions import col, regexp_replace
from pyspark.sql.functions import trim
from pyspark.sql.functions import udf
from pyspark.sql.functions import split
from pyspark.sql.functions import create_map, lit
from pyspark.sql.types import FloatType
from pyspark.sql.functions import when
from pyspark.sql.functions import skewness, kurtosis
import pyspark.sql.functions as F
from pyspark.sql.functions import rand,col,when,concat,substring,lit,udf,lower,sum as ps_sum,count as ps_count,row_number
from pyspark.sql.window import *
from pyspark.sql.types import *
from pyspark.sql import Row

# Import general data processing libraries
import pandas as pd
import numpy as np
import ast
from itertools import chain
from functools import reduce
from math import radians, cos, sin, asin, sqrt
import pandas as pd

# Import visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
import random

# Import PySpark ML libraries and supporting libraries
from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.sql import DataFrame
from pyspark.mllib.stat import Statistics
from pyspark.ml.feature import VectorAssembler,BucketedRandomProjectionLSH,VectorSlicer, StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, Imputer
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.sql.functions import array, create_map, struct
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import PCA


spark.conf.set("spark.databricks.io.cache.enabled", "true")

# COMMAND ----------

sc = spark.sparkContext
spark

# COMMAND ----------

# ASHWINI STORAGE
blob_container = "w261team09container" # The name of your container created in https://portal.azure.com
storage_account = "w261team09storage" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261team09" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261team09-key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# MACKENZIE STORAGE
blob_container = "w261team09container" # The name of your container created in https://portal.azure.com
storage_account = "w261team09storage2" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261team09scope" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261team09-key" # The name of the secret key created in your local computer using the Databricks CLI
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)

# COMMAND ----------

# MAGIC %md
# MAGIC # This notebook is a continuation of the join notebook

# COMMAND ----------

# Must read from Ashwini's storage
# Read/write the other data from Mackenzie's storage
df_full = spark.read.parquet(f"{blob_url}/df_last/*")

# COMMAND ----------

print(f'Number of rows in the full data set: {df_full.count():,}')

# COMMAND ----------

display(df_full)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering: Add New Features

# COMMAND ----------

# Create to Hour and Minute Columns Based on Scheduled Departure time
df_full = df_full.withColumn('HOUR', F.hour('LOCAL_TIME')).withColumn('MINUTE', F.minute('LOCAL_TIME'))

# COMMAND ----------

# Create a rolling window column that shows the nth flight that flight is for the day
w = Window.partitionBy(['CALENDAR_DAY_AIRLINES']).orderBy('LOCAL_TIME')
df_full = df_full.withColumn('NTH_FLIGHTS', F.rank().over(w)).cache()

# number of delayed flights yesterday
windowval = (Window.partitionBy(['CALENDAR_DAY_AIRLINES']))
df_full = df_full.withColumn('NUM_DELAY', F.sum('DEP_DEL15').over(windowval)).withColumn('PRIOR_DAY_NUM_DELAY', F.lag('NUM_DELAY', 1).over(Window.partitionBy(['CALENDAR_DAY_AIRLINES']).orderBy('CALENDAR_DAY_AIRLINES'))).drop('NUM_DELAY')

# COMMAND ----------

display(df_full)

# COMMAND ----------

# Split out the weather column pieces into individual columns to get the separated values
split_WND =  F.split(df_full.WND, '\\,')
split_CIG =  F.split(df_full.CIG, '\\,')
split_VIS =  F.split(df_full.VIS, '\\,')
split_TMP =  F.split(df_full.TMP, '\\,')
split_SLP =  F.split(df_full.SLP, '\\,')
split_DEW =  F.split(df_full.DEW, '\\,')
split_AW1 =  F.split(df_full.AW1, '\\,')
split_GA1 =  F.split(df_full.GA1, '\\,')
split_GE1 =  F.split(df_full.GE1, '\\,')
split_KA1 =  F.split(df_full.KA1, '\\,')
split_MA1 =  F.split(df_full.MA1, '\\,')
split_MD1 =  F.split(df_full.MD1, '\\,')
split_OC1 =  F.split(df_full.OC1, '\\,')
split_GD1 =  F.split(df_full.GD1, '\\,')
split_CH1 =  F.split(df_full.CH1, '\\,')
split_CT1 =  F.split(df_full.CT1, '\\,')
split_AO1 =  F.split(df_full.AO1, '\\,')
split_AA1 =  F.split(df_full.AA1, '\\,')
split_AU1 =  F.split(df_full.AU1, '\\,')
split_CO1 =  F.split(df_full.CO1, '\\,')
split_CB1 =  F.split(df_full.CB1, '\\,')
split_CG1 =  F.split(df_full.CG1, '\\,')
split_CW1 =  F.split(df_full.CW1, '\\,')

df_split = df_full.withColumn('WND1', split_WND.getItem(0)).withColumn('WND2', split_WND.getItem(1)).withColumn('WND3', split_WND.getItem(2)).withColumn('WND4', split_WND.getItem(3)).withColumn('WND5', split_WND.getItem(4))

df_split = df_split.withColumn('CIG1', split_CIG.getItem(0)).withColumn('CIG2', split_CIG.getItem(1)).withColumn('CIG3', split_CIG.getItem(2)).withColumn('CIG4', split_CIG.getItem(3))

df_split = df_split.withColumn('VIS1', split_VIS.getItem(0)).withColumn('VIS2', split_VIS.getItem(1)).withColumn('VIS3', split_VIS.getItem(2)).withColumn('VIS4', split_VIS.getItem(3))

df_split = df_split.withColumn('TMP1', split_TMP.getItem(0)).withColumn('TMP2', split_TMP.getItem(1))
df_split = df_split.withColumn('DEW1', split_DEW.getItem(0)).withColumn('DEW2', split_DEW.getItem(1))
df_split = df_split.withColumn('SLP1', split_SLP.getItem(0)).withColumn('SLP2', split_SLP.getItem(1))
df_split = df_split.withColumn('AW11', split_AW1.getItem(0)).withColumn('AW12', split_AW1.getItem(1))

df_split = df_split.withColumn('GA11', split_GA1.getItem(0)).withColumn('GA12', split_GA1.getItem(1)).withColumn('GA13', split_GA1.getItem(2)).withColumn('GA14', split_GA1.getItem(3)).withColumn('GA15', split_GA1.getItem(4)).withColumn('GA16', split_GA1.getItem(5))

df_split = df_split.withColumn('GE11', split_GE1.getItem(0)).withColumn('GE12', split_GE1.getItem(1)).withColumn('GE13', split_GE1.getItem(2)).withColumn('GE14', split_GE1.getItem(3))
df_split = df_split.withColumn('KA11', split_KA1.getItem(0)).withColumn('KA12', split_KA1.getItem(1)).withColumn('KA13', split_KA1.getItem(2)).withColumn('KA14', split_KA1.getItem(3))
df_split = df_split.withColumn('MA11', split_MA1.getItem(0)).withColumn('MA12', split_MA1.getItem(1)).withColumn('MA13', split_MA1.getItem(2)).withColumn('MA14', split_MA1.getItem(3))

df_split = df_split.withColumn('MD11', split_MD1.getItem(0)).withColumn('MD12', split_MD1.getItem(1)).withColumn('MD13', split_MD1.getItem(2)).withColumn('MD14', split_MD1.getItem(3)).withColumn('MD15', split_MD1.getItem(4)).withColumn('MD16', split_MD1.getItem(5))

df_split = df_split.withColumn('OC11', split_OC1.getItem(0)).withColumn('OC12', split_OC1.getItem(1))

df_split = df_split.withColumn('GD11', split_GD1.getItem(0)).withColumn('GD12', split_GD1.getItem(1)).withColumn('GD13', split_GD1.getItem(2)).withColumn('GD14', split_GD1.getItem(3)).withColumn('GD15', split_GD1.getItem(4)).withColumn('GD16', split_GD1.getItem(5))

df_split = df_split.withColumn('CH11', split_CH1.getItem(0)).withColumn('CH12', split_CH1.getItem(1)).withColumn('CH13', split_CH1.getItem(2)).withColumn('CH14', split_CH1.getItem(3)).withColumn('CH15', split_CH1.getItem(4)).withColumn('CH16', split_CH1.getItem(5)).withColumn('CH17', split_CH1.getItem(6))

df_split = df_split.withColumn('CT11', split_CT1.getItem(0)).withColumn('CT12', split_CT1.getItem(1)).withColumn('CT13', split_CT1.getItem(1))

df_split = df_split.withColumn('AO11', split_AO1.getItem(0)).withColumn('AO12', split_AO1.getItem(1)).withColumn('AO13', split_AO1.getItem(2)).withColumn('AO14', split_AO1.getItem(3))

df_split = df_split.withColumn('AA11', split_AA1.getItem(0)).withColumn('AA12', split_AA1.getItem(1)).withColumn('AA13', split_AA1.getItem(2)).withColumn('AA14', split_AA1.getItem(3))

df_split = df_split.withColumn('AU11', split_AU1.getItem(0)).withColumn('AU12', split_AU1.getItem(1)).withColumn('AU13', split_AU1.getItem(2)).withColumn('AU14', split_AU1.getItem(3)).withColumn('AU15', split_AU1.getItem(4)).withColumn('AU16', split_AU1.getItem(5)).withColumn('AU17', split_AU1.getItem(6))

df_split = df_split.withColumn('CO11', split_CO1.getItem(0)).withColumn('CO12', split_CO1.getItem(1))

df_split = df_split.withColumn('CB11', split_CB1.getItem(0)).withColumn('CB12', split_CB1.getItem(1)).withColumn('CB13', split_CB1.getItem(2)).withColumn('CB14', split_CB1.getItem(3))

df_split = df_split.withColumn('CG11', split_CG1.getItem(0)).withColumn('CG12', split_CG1.getItem(1)).withColumn('CG13', split_CG1.getItem(2))

df_split = df_split.withColumn('CW11', split_CW1.getItem(0)).withColumn('CW12', split_CW1.getItem(1)).withColumn('CW13', split_CW1.getItem(2)).withColumn('CW14', split_CW1.getItem(3)).withColumn('CW15', split_CW1.getItem(4)).withColumn('CW16', split_CW1.getItem(5))

# COMMAND ----------

display(df_split)

# COMMAND ----------

print(f'There are now {len(df_split.columns)} columns')

# COMMAND ----------

# Based on the data dictionary for weather, https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf, we can 
# use the proper null representation for each of the split weather columns
mv_9 = ['WND2', 'WND3', 'WND5', 'CIG2', 'CIG3', 'CIG4', 'VIS2', 'VIS3', 'VIS4', 'GE11', 'KA12', 'MD11', 'GD11', 'GD16', 'CH13', 'CH16', 'CT12', 'AO13', 'AA13', 'AU11', 'AU12', 'AU14', 'AU15', 'AU16', 'CG12', 'CW12', 'CW15']
mv_99 = ['GA11', 'GA15', 'GD12', 'CH11', 'AO11', 'AA11', 'AU13', 'CO11']
mv_999 = ['WND1', 'KA11', 'MD13', 'MD15']
mv_9999 = ['WND4', 'TMP1', 'DEW1', 'KA13', 'OC11', 'CH12', 'CH15', 'CT11', 'AO12', 'AA12']
mv_99999 = ['CIG1', 'SLP1', 'GA13', 'GE13', 'GE14', 'MA11', 'MA13', 'GD14', 'CO12', 'CG11', 'CW11', 'CW14']
mv_999999 = ['VIS1', 'GE12']
mv_und = ['TMP2', 'DEW2', 'SLP2', 'AW11', 'AW12', 'GA12', 'GA14', 'KA14', 'MA12', 'MA14', 'MD12', 'MD14', 'MD16', 'OC12', 'GD13', 'GD15', 'CH14', 'CH17', 'CT13', 'AO14', 'AA14', 'AU17', 'CO14', 'CG13', 'CW13', 'CW16']

# COMMAND ----------

# Convert appropriate weather columns to numeric values
weatherCols_toConvertDouble = ['WND1', 'WND4', 'CIG1', 'VIS1', 'TMP1', 'DEW1', 'SLP1', 'GA13', 'GE13', 'GE14', 'KA11', 'KA13', 
                               'MA11', 'MA13', 'MD13', 'MD15', 'GD14', 'CH11', 'CH12', 'CH15', 'CT11', 'AO11', 'AO12', 
                               'AA11', 'AA12', 'CB11', 'CB12', 'CG11', 'CW11', 'CW14', 'OC11']  
weatherCols_toConvertInteger = ['CO11', 'CO12']

# COMMAND ----------

# Fill in null values for the weather data depending on the key
df_split = df_split.fillna('9', subset=mv_9)
df_split = df_split.fillna('99', subset=mv_99)
df_split = df_split.fillna('999', subset=mv_999)
df_split = df_split.fillna('9999', subset=mv_9999)
df_split = df_split.fillna('99999', subset=mv_99999)
df_split = df_split.fillna('999999', subset=mv_999999)

# COMMAND ----------

# Code to convert string to numeric cols
df_split_converted = df_split
for col_name in weatherCols_toConvertDouble:
  df_split_converted = df_split_converted.withColumn(col_name, df_split_converted[col_name].cast(DoubleType()))
  
for col_name in weatherCols_toConvertInteger:
  df_split_converted = df_split_converted.withColumn(col_name, df_split_converted[col_name].cast(IntegerType()))

# COMMAND ----------

# Sanity check to ensure the proper string columns are now numerics
df_split_converted.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop Future Flight Information and Unnecessary Columns (Duplicate data)

# COMMAND ----------

cols_to_remove1 = [ 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL','DEP_TIME', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID','DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'GA2', 'GA3', 'KA2', 'REM', 'EQD', 'GD2', 'CT3', 'GD3', 'CG3', 'CT2', 'CG2', 'COUNTRY', 'CLOSEST_CITY', 'CALENDAR_DAY_WEATHER','WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP', 'AW1', 'GA1', 'GE1', 'KA1', 'MA1', 'MD1', 'OC1', 'GD1', 'CH1', 'CT1', 'AO1', 'AA1', 'AU1', 'CO1',  'CB1', 'CG1', 'CW1', 'DEP_DELAY', 'DEP_DELAY_NEW','DEP_DELAY_GROUP']

cols_to_remove2 = ['ORIGIN_STATE_NM','CRS_DEP_TIME', 'TIMEZONE', 'LATITUDE', 'LONGITUDE', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'ORIGIN_CITY_NAME', 'DEP_TIME_BLK', 'ORIGIN_WAC', 'DEST_WAC', 'FLIGHTS','DATE', 'ORIGIN_SATE_FIPS', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 'FLIGHT_ID2', 'LATITUDE_ORIG', 'LONGITUDE_ORIG', 'OP_CARRIER_AIRLINE_ID', 'DEST_CITY_NAME', 'CALENDAR_DAY']

cols_to_remove = cols_to_remove1 + cols_to_remove2

df_split_full = df_split_converted.drop(*cols_to_remove)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split the Data into Train and Test (Time Series Method)
# MAGIC - Sequentially split the data (ordering by Calendar Day and time) 
# MAGIC - 80/20 split

# COMMAND ----------

yr_count = df_split_full.groupby(['YEAR']).count()
yr_count = yr_count.orderBy(yr_count.YEAR)
display(yr_count)

# COMMAND ----------

yrmnth_count = df_split_full.groupby(['YEAR', 'MONTH']).count()
yrmnth_count = yrmnth_count.orderBy(yrmnth_count.YEAR)
display(yrmnth_count)

# COMMAND ----------

df_split_full = df_split_full.withColumn("rank", F.percent_rank().over(Window.partitionBy().orderBy(["CALENDAR_DAY_AIRLINES", 'HOUR', 'MINUTE'])))

# COMMAND ----------

train_df = df_split_full.where("rank <= .8").drop("rank")
test_df = df_split_full.where("rank > .8").drop("rank")

# COMMAND ----------

display(test_df)

# COMMAND ----------

print(f'Number of rows in the train data: {train_df.count()}')
print(f'Number of rows in the test data: {test_df.count()}')

# COMMAND ----------

display(train_df)

# COMMAND ----------

# Save the  train/test splits with filled in nulls and converted columns to Azure
train_df.write.parquet(f"{blob_url}/df_train")
test_df.write.parquet(f"{blob_url}/df_test")

# COMMAND ----------

# Read in data from Azure
train_df = spark.read.parquet(f"{blob_url}/df_train/*")
test_df = spark.read.parquet(f"{blob_url}/df_test/*")

# COMMAND ----------

display(train_df)

# COMMAND ----------

display(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Target Variable: DEP_DEL15

# COMMAND ----------

display(train_df.select('DEP_DEL15'))

# COMMAND ----------

# MAGIC %md
# MAGIC Plot above shows the count of `No Delay`, `Cancelled`, and `Delay` flights. For this project, we are classifying cancelled flights as delayed flights.

# COMMAND ----------

# Classify cancelled flights as delayed
train_df_RC = train_df.withColumn("DEP_DEL15", \
              when(train_df["CANCELLED"] == 1, 1).otherwise(train_df["DEP_DEL15"]))

test_df_RC = test_df.withColumn("DEP_DEL15", \
              when(test_df["CANCELLED"] == 1, 1).otherwise(test_df["DEP_DEL15"]))

# COMMAND ----------

display(test_df_RC)

# COMMAND ----------

display(train_df_RC.select('DEP_DEL15'))

# COMMAND ----------

train_df.groupBy('DEP_DEL15').count().show()

# COMMAND ----------

train_df_RC.groupBy('DEP_DEL15').count().show()

# COMMAND ----------

tmp = train_df.filter((F.col('OP_CARRIER')=='9E'))
tmp = tmp.filter(F.col('DEP_DEL15').isNull())
tmp = tmp.filter(F.col('CANCELLED')==0)

# View one example
# 9E flights that aren't cancelled and not delayed need to be relabelled as 0
display(tmp.filter((F.col('CALENDAR_DAY_AIRLINES')=='2018-02-01') & (F.col('OP_CARRIER')=='9E') & (F.col('DEP_DEL15').isNull())))

# COMMAND ----------

# MAGIC %md
# MAGIC Some flights by 9E are missing a `DEP_DEL15` value even though they have an on-time departure. 

# COMMAND ----------

# fill in the remaining flights with a null label as not cancelled since they do not have a delay
train_df_RC = train_df_RC.fillna(0, subset='DEP_DEL15')
test_df_RC = test_df_RC.fillna(0, subset='DEP_DEL15')

# COMMAND ----------

train_df_RC.groupBy('DEP_DEL15').count().show()

# COMMAND ----------

# Before classifying cancellations as delayed
train_df.filter((F.col('OP_CARRIER')=='9E')).groupBy('DEP_DEL15').count().show()

# COMMAND ----------

train_df.filter((F.col('OP_CARRIER')=='9E')).groupBy('CANCELLED').count().show()

# COMMAND ----------

# After classifying cancellations as delayed
train_df_RC.filter((F.col('OP_CARRIER')=='9E')).groupBy('DEP_DEL15').count().show()

# COMMAND ----------

# scatter plot of all delay types
display(train_df.filter(train_df.DEP_DELAY > 14).select('DEP_DELAY', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY','SECURITY_DELAY','LATE_AIRCRAFT_DELAY'))

# COMMAND ----------

# Number of  Flights per State
train_df.registerTempTable("train_df")
display(sqlContext.sql("Select ORIGIN_STATE_ABR as state ,count(DEP_DEL15) as value from train_df  where ORIGIN_STATE_ABR in ('AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA','HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD','MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ','NM', 'NY', 'NC', 'ND', 'OH','OK', 'OR','PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY') group by ORIGIN_STATE_ABR"))


# COMMAND ----------

# MAGIC %md
# MAGIC #### Correlation Matrix

# COMMAND ----------

from pyspark.ml.feature import Imputer
# Subset and impute the numeric columns with the median to calculate the correlation matrix
num_cols = [f.name for f in train_df_RC.schema.fields if isinstance(f.dataType, DoubleType) or isinstance(f.dataType, IntegerType)]

imputer = Imputer(
    inputCols=num_cols, 
    outputCols=["{}".format(c) for c in num_cols]
    ).setStrategy("median")

train_num = train_df_RC[num_cols]
df_impute = imputer.fit(train_num).transform(train_num)

# COMMAND ----------

display(df_impute)

# COMMAND ----------

# Drop columns with dtype string, data, timestamp
notDate_train_df = df_impute.select([i[0] for i in df_impute.dtypes if not 'date' in i[1]])
notDateStr_train_df = notDate_train_df.select([i[0] for i in notDate_train_df.dtypes if not 'str' in i[1]])
notDateStrTimestamp_train_df = notDateStr_train_df.select([i[0] for i in notDateStr_train_df.dtypes if not 'timestamp' in i[1]])

# COMMAND ----------

# List of features
features = notDateStrTimestamp_train_df.rdd.map(lambda x: x[0:])
# Correlation Matrix
corr_mat = Statistics.corr(features, method = "pearson")
corr_mat_df = pd.DataFrame(corr_mat, columns = notDateStrTimestamp_train_df.columns, index = notDateStrTimestamp_train_df.columns)

# COMMAND ----------

# Plot Correlation Heatmap
f, ax = plt.subplots(figsize=(24, 18))
sns.heatmap(corr_mat_df, vmax=.8, square=True)

# COMMAND ----------

# Show columns that are at least 0.75 correlated with another column
m = (corr_mat_df.mask(np.eye(len(corr_mat_df), dtype=bool)).abs() >= 0.75).any()
correlated = corr_mat_df.loc[m, m]
correlated

# COMMAND ----------

# Drop highly correlated columns
drop_corr_cols = ['QUARTER', 'DISTANCE', 'DISTANCE_GROUP', 'NTH_FLIGHTS', 'VIS1', 'DEW1', 'GA13', 'KA13', 'CW14', 'CH12','CH15', 'CT11', 'AO11', 'AO12', 'AA12', 'CO12', 'CG11', 'CW11', 'CW14']

# COMMAND ----------

display(test_df_RC)

# COMMAND ----------

train_df_RC = train_df_RC.drop(*drop_corr_cols)
test_df_RC = test_df_RC.drop(*drop_corr_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filling String Null Values and Dropping Columns with 89%> nulls

# COMMAND ----------

# Count number of rows in the data
train_count = train_df_RC.count()

# COMMAND ----------

# Create a new dataframe that displays the percent of missing values per column
missing_train_RC = train_df_RC.agg(*[(F.count(F.when(F.isnull(c), c))/train_count).alias(c) for c in train_df_RC.columns])

# COMMAND ----------

display(missing_train_RC)

# COMMAND ----------

# drop the selected columns with missing values greater than the threshold
threshold = 0.89
null_cols_train = [key for (key,value) in missing_train_RC.collect()[0].asDict().items() if value >= threshold]
print(null_cols_train)

# COMMAND ----------

# These columns to be dropped also have info captured by other other columns since they came from the split weather columns so we can assume
# little info will be lost
null_cols_train = ['AW11', 'AW12', 'KA14', 'MD12', 'MD14', 'MD16', 'OC12', 'AU17']
train_df_RC = train_df_RC.drop(*null_cols_train)
test_df_RC = test_df_RC.drop(*null_cols_train)

# COMMAND ----------

# get the categorical and numeric columns
str_cols = [f.name for f in train_df_RC.schema.fields if isinstance(f.dataType, StringType)]
num_cols = [f.name for f in train_df_RC.schema.fields if isinstance(f.dataType, DoubleType) or isinstance(f.dataType, IntegerType)]

# COMMAND ----------

# find str columns that have missing values, we will impute numerical values later
str_cols_null = [key for (key,value) in missing_train_RC[str_cols].collect()[0].asDict().items() if value > 0]
print(str_cols_null)

# COMMAND ----------

str_fill = ['TAIL_NUM','STATION', 'GF1', 'TMP2', 'DEW2', 'SLP2', 'GA12', 'GA14', 'GA16', 'MA12', 'MA14', 'GD13', 'GD15', 'CH14', 'CH17', 'CT13', 'AO14', 'AA14', 'CB13', 'CB14', 'CG13', 'CW13', 'CW16']

# COMMAND ----------

# fill the string null values with the most common value
for col in str_fill:
  mode = train_df_RC.where(F.col(col).isNotNull()).groupby(col).count().orderBy(F.desc('count')).first()[col]
  train_df_RC = train_df_RC.fillna(str(mode), subset=col)

# COMMAND ----------

for col in str_fill:
  mode = test_df_RC.where(F.col(col).isNotNull()).groupby(col).count().orderBy(F.desc('count')).first()[col]
  test_df_RC = test_df_RC.fillna(str(mode), subset=col)

# COMMAND ----------

display(test_df_RC)

# COMMAND ----------

# Sanity check ensuring the missing values are now filled
missing_train_RC2 = train_df_RC.agg(*[(F.count(F.when(F.isnull(c), c))/train_count).alias(c) for c in train_df_RC.columns])
display(missing_train_RC2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find correlated Categorical Columns using Univariate Feature Selection

# COMMAND ----------

str_cols = [f.name for f in train_df_RC.schema.fields if isinstance(f.dataType, StringType)]
print(str_cols)

# COMMAND ----------

train_strs = train_df_RC[str_cols+['DEP_DEL15']]

# COMMAND ----------

# Convert categorical values to integer representation so feature selection can work
cat_cols = train_strs.columns + ['DEP_DEL15']
indexers = [StringIndexer(inputCol=column, outputCol=column+'_index', handleInvalid = 'keep') for column in list(set(cat_cols))]
pipeline = Pipeline(stages=indexers)
train_strs_ind = pipeline.fit(train_strs).transform(train_strs)

# COMMAND ----------

display(train_strs_ind)

# COMMAND ----------

cols = ['MA12_index',  'WND5_index',  'CIG4_index',  'GD12_index',  'AU12_index',  'CG13_index',  'WND2_index',  'ORIGIN_index',  'SLP2_index',  'GD11_index',  'DEST_index',  'STATION_index',  'MA14_index',  'AA14_index',  'DEP_DEL15_index',  'CH14_index',  'KA12_index',  'GE11_index',  'GE12_index',  'GA12_index',  'CH13_index',  'AU11_index',  'DEST_STATE_ABR_index',  'GD15_index',  'WND3_index',  'CT13_index',  'AA13_index',  'VIS4_index',  'AO13_index',  'GF1_index',  'AU15_index',  'ORIGIN_STATE_ABR_index',  'CB13_index',  'TMP2_index',  'CIG2_index',  'TAIL_NUM_index',  'GA16_index',  'DEW2_index',  'CG12_index',  'VIS3_index',  'AO14_index',  'CW12_index',  'OP_CARRIER_index',  'AU13_index',  'CW15_index',  'CB14_index',  'VIS2_index',  'AU16_index',  'CH17_index',  'MD11_index',  'GD13_index',  'GD16_index',  'GA15_index',  'CIG3_index',  'CW13_index',  'AU14_index',  'CT12_index',  'GA11_index',  'GA14_index',  'CH16_index',  'CW16_index']

# Vectorize the categorical values so that they can be used as input into the algorithm
vectorAssembler = VectorAssembler(inputCols = cols, outputCol = 'features')
v_str = vectorAssembler.transform(train_strs_ind)
v_str = v_str.select(['features', 'DEP_DEL15']).withColumnRenamed('DEP_DEL15','label')

# COMMAND ----------

# Use family-wise error rate to control for the number of columns to be dropped
selector = UnivariateFeatureSelector(outputCol="selectedFeatures", selectionMode='fwe')
selector.setFeatureType("categorical").setLabelType("categorical")
feat_sel = selector.fit(v_str)

# COMMAND ----------

# Show the featuers to be dropped after running the algorithm
str_selectedCols = [cols[i].replace('_index','') for i in feat_sel.selectedFeatures]
noInd = [col.replace('_index','') for col in cols]
drop_strs = set(noInd).difference(set(str_selectedCols))
print(drop_strs)

# COMMAND ----------

drop_strs = ['GE11', 'GD12', 'CW13', 'AO13', 'CH17', 'GD13', 'CW16']
train_df_RC = train_df_RC.drop(*drop_strs)
test_df_RC = test_df_RC.drop(*drop_strs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bucket Categorical Values to Reduce Cardinality

# COMMAND ----------

# Show cardinality for each string column
for col in train_strs.columns:
  row = train_strs.select(F.countDistinct(col)).collect()
  print(f'{col}: {row}')

# COMMAND ----------

# TAIL_NUM shows the biggest opportunity to cut down the number of distinct values along with STATION
# tail nums can be random so we can try to group them
# https://crankyflier.com/2014/10/16/everything-youve-ever-wanted-to-know-about-us-airline-tail-numbers-part-1/
tail_num = train_strs.select('TAIL_NUM')
udf_str = udf(lambda x:x[-2:],StringType())
tail_num = tail_num.withColumn('sub_tail', udf_str('TAIL_NUM'))
tail_num.show()

# COMMAND ----------

# reduced cardinality from 7648 to 425
tail_num.select(F.countDistinct('sub_tail')).show()

# COMMAND ----------

# Can assume that location data encapsulates station information so we don't need stations data
stations  = train_strs.select('STATION')
stations.distinct().show()

# COMMAND ----------

train_df_RC = train_df_RC.withColumn('TAIL_NUM', udf_str('TAIL_NUM')).drop('STATION')
test_df_RC = test_df_RC.withColumn('TAIL_NUM', udf_str('TAIL_NUM')).drop('STATION')

# COMMAND ----------

display(test_df_RC)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline Model - Predict No Delay for Every Row in the Test Data

# COMMAND ----------

# Get the baseline for if we predicted not delayed for every row
actuals_test = test_df_RC.select('DEP_DEL15').rdd.map(lambda r: r[0]).collect()
accuracy_mean = sum(np.array(actuals_test) == np.zeros(len(actuals_test)))/len(actuals_test)
print(f'Accuracy of predict delay model: {accuracy_mean:.3f}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline Logistic Regression
# MAGIC 
# MAGIC Run a simple logistic regression model in numerical columns with imputing to get a base idea on model performance

# COMMAND ----------

# Impute and prepare data to be vectorized
imputer = Imputer(
    inputCols=num_cols, 
    outputCols=["{}".format(c) for c in num_cols]
    ).setStrategy("median")

train_num = train_df_RC[num_cols]
train_fit = imputer.fit(train_num)
train_base = train_fit.transform(train_num)
train_base = train_base.withColumnRenamed('DEP_DEL15', 'label')

test_base = train_fit.transform(test_df_RC[train_num.columns]).withColumnRenamed('DEP_DEL15', 'label')
actuals_base = test_base.select('label').rdd.map(lambda r: r[0]).collect()

# COMMAND ----------

# Train a logistic regression on vectorized data
lr = LogisticRegression(maxIter=5)

cols = train_base.columns
cols.remove('label')

vectorAssembler = VectorAssembler(inputCols = cols, outputCol = 'features')
v_df = vectorAssembler.transform(train_base)
v_df = v_df.select(['features', 'label'])

base_model = lr.fit(v_df)

# COMMAND ----------

vectorAssemblertest = VectorAssembler(inputCols = cols, outputCol = 'features')
v_dftest = vectorAssembler.transform(test_base)
v_dftest = v_dftest.select(['features', 'label'])

# COMMAND ----------

# get the predictions
preds_base = base_model.transform(v_dftest)
predictions_base = preds_base.select('prediction').rdd.map(lambda r: r[0]).collect()

# COMMAND ----------

accuracy_base = sum(np.array(predictions_base) == np.array(actuals_base))/len(actuals_base)
print(f'Accuracy of baseline model: {accuracy_base:.3f}')

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print(f'Area under PR for baseline model: {evaluator.evaluate(preds_base, {evaluator.metricName: "areaUnderPR"})}')

# COMMAND ----------

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

conf_preds = preds_base.select('prediction').rdd.map(lambda r: r[0]).collect()
conf_mat = confusion_matrix(actuals_base, conf_preds)
conf_mat_normalized = conf_mat.astype('float') / len(conf_preds)
sns.heatmap(conf_mat_normalized, annot=True,fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preprocessing - Pipeline Creation
# MAGIC - Transforming variable
# MAGIC - Encoding variables

# COMMAND ----------

str_cols = [f.name for f in train_df_RC.schema.fields if isinstance(f.dataType, StringType)]
numeric_cols = [f.name for f in train_df_RC.schema.fields if isinstance(f.dataType, DoubleType) or isinstance(f.dataType, IntegerType)]

# COMMAND ----------

############################## spark smote oversampling ##########################
#for categorical columns, must take its stringIndexed form (smote should be after string indexing, default by frequency)
def pre_smote_df_process(df_train, df_test, num_cols,cat_cols,target_col,index_suffix="_index"):
    '''
    string indexer and vector assembler
    inputs:
    * df: spark df, original
    * num_cols: numerical cols to be assembled
    * cat_cols: categorical cols to be stringindexed
    * target_col: prediction target
    * index_suffix: will be the suffix after string indexing
    output:
    * vectorized_test: vectorized spark train df after processing
    * vectorized_train: vectorized spark test df after processing
    '''
    if(df_train.select(target_col).distinct().count() != 2):
        raise ValueError("Target col must have exactly 2 classes")
        
    if target_col in num_cols:
        num_cols.remove(target_col)
    
    # index the string cols, except possibly for the label col
    indexers = [StringIndexer(inputCol=column, outputCol=column+index_suffix, handleInvalid = 'keep') for column in list(set(cat_cols))]
    
    # one hot encoder
    ohes = [OneHotEncoder(inputCol=c+index_suffix, outputCol=c+"_class") for c in list(set(cat_cols))]

    
    # Impute numeric columns
    imputers = [Imputer(inputCols = num_cols, outputCols = ["{}".format(c) for c in num_cols]).setStrategy("median")]

    # Build the stage for the ML pipeline
    vector_cols = [c+"_class" for c in cat_cols] + ["{}".format(c) for c in num_cols]
    
    
    
    model_matrix_stages = indexers  + \
                          ohes + \
                          imputers + \
                          [VectorAssembler(inputCols=vector_cols, outputCol='features')]
#                           [VectorAssembler(inputCols=vector_cols, outputCol="features")]                   
    
    # scale the numeric columns
    scaler = [StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)]
  
    pipeline = Pipeline(stages=model_matrix_stages+scaler)
    pipeline_fit = pipeline.fit(df_train)
    pos_vectorized_train = pipeline_fit.transform(df_train)
    pos_vectorized_test = pipeline_fit.transform(df_test)

    
    # drop original num cols and cat cols
    drop_cols = num_cols+cat_cols
    
    keep_cols = [a for a in pos_vectorized_train.columns if a not in drop_cols]
    vectorized_train = pos_vectorized_train.select(*keep_cols).withColumn('label',pos_vectorized_train[target_col]).drop(target_col)
    vectorized_test = pos_vectorized_test.select(*keep_cols).withColumn('label',pos_vectorized_test[target_col]).drop(target_col)

    
    return vectorized_train, vectorized_test

# COMMAND ----------

vectorized_train, vectorized_test = pre_smote_df_process(train_df_RC, test_df_RC, numeric_cols, str_cols, 'DEP_DEL15')

# COMMAND ----------

#Keep local time for sorting purposes later and select just the scaled features column to reduce df size and overhead
vectorized_train_subset = vectorized_train.select(F.col('LOCAL_TIME'),F.col('scaledFeatures'),F.col('label')).cache()
vectorized_test_subset = vectorized_test.select(F.col('LOCAL_TIME'),F.col('scaledFeatures'),F.col('label')).cache()

# COMMAND ----------

# We see we have 2182 features after vectorization
display(vectorized_train_subset)

# COMMAND ----------

display(vectorized_test_subset)

# COMMAND ----------

vectorized_train_tosave.write.parquet(f"{blob_url}/vectorized_train")
vectorized_test_tosave.write.parquet(f"{blob_url}/vectorized_test")

# COMMAND ----------

train_vec = spark.read.parquet(f"{blob_url}/vectorized_train/*")

# COMMAND ----------

display(train_vec)

# COMMAND ----------

 test_vec = spark.read.parquet(f"{blob_url}/vectorized_test/*")

# COMMAND ----------

display(test_vec)

# COMMAND ----------

# MAGIC %md
# MAGIC ### PCA

# COMMAND ----------

# Create a sample of data to reduce training time
sample = vectorized_train_subset.sample(0.1, seed=123).cache()

# COMMAND ----------

# Run 1000 n_components of PCA on the sample
pca = PCA(k=1000, inputCol="scaledFeatures")
pca.setOutputCol("pca_features")
pca_fit = pca.fit(sample)
explained_var = sum(pca_fit.explainedVariance)
print(explained_var)

#sample of 0.0001 with k=100 had EV=0.56
# 0.5683409614878457 for 0.1 sample for k=500
# 10% sample, k=1000, EV=67%

# COMMAND ----------

# Start with 500 components and see what the explained variance is, adjust from there
pca = PCA(k=500, inputCol="scaledFeatures")
pca.setOutputCol("pca_features")
pca_fit_500 = pca.fit(vectorized_train_subset)
explained_var_500 = sum(pca_fit_500.explainedVariance)
print(explained_var_500)

# COMMAND ----------

(pca_fit_500.explainedVariance)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Univariate Feature Selection
# MAGIC 
# MAGIC PCA showed poor results so we can try Univariate Feature Selection again to find ways to reduce the ~2k features

# COMMAND ----------

display(vectorized_train)

# COMMAND ----------

# fwe chooses all features whose p-values are below a threshold. The threshold is scaled by 1 / numFeatures, thus controlling the family-wise error rate of selection.
selector = UnivariateFeatureSelector(featuresCol='scaledFeatures',outputCol="selectedFeatures", selectionMode='fwe')
selector.setFeatureType("continuous").setLabelType("categorical")
feat_sel = selector.fit(vectorized_train_subset)

# COMMAND ----------

vec_subset_train = feat_sel.transform(vectorized_train_subset)

# COMMAND ----------

display(vec_subset_train)

# COMMAND ----------

vec_subset_test = feat_sel.transform(vectorized_test_subset)

# COMMAND ----------

# Save the data
vec_subset_train_tosave = vec_subset_train.select(F.col('LOCAL_TIME'),F.col('selectedFeatures'),F.col('label'))
vec_subset_train_tosave.write.parquet(f"{blob_url}/train_selected")

# COMMAND ----------

#test Save the data
vec_subset_test_tosave = vec_subset_test.select(F.col('LOCAL_TIME'),F.col('selectedFeatures'),F.col('label'))
vec_subset_test_tosave.write.parquet(f"{blob_url}/test_selected")

# COMMAND ----------

# reduced features by ~50%
train_vec_subset = spark.read.parquet(f"{blob_url}/train_selected/*")
display(train_vec_subset)

# COMMAND ----------

test_vec_subset = spark.read.parquet(f"{blob_url}/test_selected/*")
display(test_vec_subset)

# COMMAND ----------

train_vec_subset = train_vec_subset .withColumnRenamed('selectedFeatures', 'features')
test_vec_subset = test_vec_subset.withColumnRenamed('selectedFeatures', 'features')

# COMMAND ----------

# MAGIC %md
# MAGIC ### SMOTE

# COMMAND ----------

# https://medium.com/@haoyunlai/smote-implementation-in-pyspark-76ec4ffa2f1d
def smote(vectorized_sdf,seed=48, bucketLength=5, k=3, multiplier=2):
    '''
    contains logic to perform smote oversampling, given a spark df with 2 classes
    inputs:
    * vectorized_sdf: cat cols are already stringindexed, num cols are assembled into 'features' vector
      df target col should be 'label'
    output:
    * oversampled_df: spark df after smote oversampling
    '''
    dataInput_min = vectorized_sdf[vectorized_sdf['label'] == 1]
    dataInput_maj = vectorized_sdf[vectorized_sdf['label'] == 0]
    
    # LSH, bucketed random projection
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes",seed=seed, bucketLength=bucketLength)
    # smote only applies on existing minority instances    
    model = brp.fit(dataInput_min)
    model.transform(dataInput_min)

    # here distance is calculated from brp's param inputCol
    self_join_w_distance = model.approxSimilarityJoin(dataInput_min, dataInput_min, float("inf"), distCol="EuclideanDistance")

    # remove self-comparison (distance 0)
    self_join_w_distance = self_join_w_distance.filter(self_join_w_distance.EuclideanDistance > 0)

    over_original_rows = Window.partitionBy("datasetA").orderBy("EuclideanDistance")

    self_similarity_df = self_join_w_distance.withColumn("r_num", F.row_number().over(over_original_rows))

    self_similarity_df_selected = self_similarity_df.filter(self_similarity_df.r_num <= k)

    over_original_rows_no_order = Window.partitionBy('datasetA')

    # list to store batches of synthetic data
    res = []
    
    # two udf for vector add and subtract, subtraction include a random factor [0,1]
    subtract_vector_udf = F.udf(lambda arr: random.uniform(0, 1)*(arr[0]-arr[1]), VectorUDT())
    add_vector_udf = F.udf(lambda arr: arr[0]+arr[1], VectorUDT())
    
    # retain original columns
    original_cols = dataInput_min.columns
    
    for i in range(multiplier):
        print("generating batch %s of synthetic instances"%i)
        # logic to randomly select neighbour: pick the largest random number generated row as the neighbour
        df_random_sel = self_similarity_df_selected.withColumn("rand", F.rand()).withColumn('max_rand', F.max('rand').over(over_original_rows_no_order))\
                            .where(F.col('rand') == F.col('max_rand')).drop(*['max_rand','rand','r_num'])
        # create synthetic feature numerical part
        df_vec_diff = df_random_sel.select('*', subtract_vector_udf(F.array('datasetA.features', 'datasetB.features')).alias('vec_diff'))
        df_vec_modified = df_vec_diff.select('*', add_vector_udf(F.array('datasetA.features', 'vec_diff')).alias('features'))
        
        # for categorical cols, either pick original or the neighbour's cat values
        for c in original_cols:
            # randomly select neighbour or original data
            col_sub = random.choice(['datasetA','datasetB'])
            val = "{0}.{1}".format(col_sub,c)
            if c != 'features':
                # do not unpack original numerical features
                df_vec_modified = df_vec_modified.withColumn(c,F.col(val))
        
        # this df_vec_modified is the synthetic minority instances,
        df_vec_modified = df_vec_modified.drop(*['datasetA','datasetB','vec_diff','EuclideanDistance'])
        
        res.append(df_vec_modified)
    
    dfunion = reduce(DataFrame.unionAll, res)
    # union synthetic instances with original full (both minority and majority) df
    oversampled_df = dfunion.union(vectorized_sdf.select(dfunion.columns))
    
    return oversampled_df

# COMMAND ----------

# Run Smote on only the train data
ss = vectorized_train_subset.sample(0.0001, seed=42).withColumnRenamed('scaledFeatures', 'features').cache()

# COMMAND ----------

train_smote = smote(ss)

# COMMAND ----------

print(ss.count())
print(train_smote.count())

# COMMAND ----------

ss.groupby('label').count().show()

# COMMAND ----------

train_smote.groupby('label').count().show()

# COMMAND ----------

display(train_smote)

# COMMAND ----------

display(train_vec_subset)

# COMMAND ----------

train_smote = smote(train_vec_subset)

# COMMAND ----------

# Save the smote train data 
train_smote.write.parquet(f"{blob_url}/train_smote")

# COMMAND ----------

# train_smote = spark.read.parquet(f"{blob_url}/train_smote/*")

# COMMAND ----------

display(train_smote)

# COMMAND ----------

train_vec_subset.groupby('label').count().show()

# COMMAND ----------

train_smote.groupby('label').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Go to Model Notebook named ______ to see the model training stages

# COMMAND ----------


