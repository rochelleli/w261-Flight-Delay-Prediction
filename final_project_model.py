# Databricks notebook source
from pyspark.sql.functions import col, regexp_replace
from pyspark.sql import Window

from pyspark.sql.functions import trim
from pyspark.sql.functions import udf
from pyspark.sql.functions import split
from pyspark.sql.functions import create_map, lit
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np

import ast
from itertools import chain
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import seaborn as sns
import math 

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import GBTClassifier
import statistics

sqlContext.setConf("spark.sql.adaptive.skewJoin.enabled", "true")

# COMMAND ----------

sc = spark.sparkContext
spark

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

from pyspark.sql.functions import col, regexp_replace
from pyspark.sql import Window

from pyspark.sql.functions import trim
from pyspark.sql.functions import udf
from pyspark.sql.functions import split
from pyspark.sql.functions import create_map, lit
from pyspark.sql.types import FloatType
import pyspark.sql.functions as F
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import ast
from itertools import chain
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import seaborn as sns

import seaborn as sns
from pyspark.sql.functions import skewness, kurtosis
import matplotlib.pyplot as plt
from pyspark.sql.functions import when
from pyspark.mllib.stat import Statistics

from pyspark.ml.feature import UnivariateFeatureSelector

import random
from functools import reduce
from pyspark.sql import Row
from pyspark.sql.functions import rand,col,when,concat,substring,lit,udf,lower,sum as ps_sum,count as ps_count,row_number
from pyspark.sql.window import *
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler,BucketedRandomProjectionLSH,VectorSlicer, StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler, Imputer
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.sql.functions import array, create_map, struct
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import PCA

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

spark.conf.set("spark.databricks.io.cache.enabled", "true")

# COMMAND ----------

train_df = spark.read.parquet(f"{blob_url}/df_train/*")
test_df = spark.read.parquet(f"{blob_url}/df_test/*")

cols_to_remove1 = [ 'NAME', 'REPORT_TYPE', 'CALL_SIGN', 'QUALITY_CONTROL','DEP_TIME', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'ARR_DEL15', 'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'FL_DATE', 'OP_UNIQUE_CARRIER', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID','DEST_AIRPORT_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_CITY_MARKET_ID', 'GA2', 'GA3', 'KA2', 'REM', 'EQD', 'GD2', 'CT3', 'GD3', 'CG3', 'CT2', 'CG2', 'COUNTRY', 'CLOSEST_CITY', 'CALENDAR_DAY_WEATHER','WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP', 'AW1', 'GA1', 'GE1', 'KA1', 'MA1', 'MD1', 'OC1', 'GD1', 'CH1', 'CT1', 'AO1', 'AA1', 'AU1', 'CO1',  'CB1', 'CG1', 'CW1', 'DEP_DELAY', 'DEP_DELAY_NEW','DEP_DELAY_GROUP']

cols_to_remove2 = ['ORIGIN_STATE_NM','CRS_DEP_TIME', 'TIMEZONE', 'LATITUDE', 'LONGITUDE', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'ORIGIN_CITY_NAME', 'DEP_TIME_BLK', 'ORIGIN_WAC', 'DEST_WAC', 'FLIGHTS','DATE', 'ORIGIN_SATE_FIPS', 'DEST_STATE_FIPS', 'DEST_STATE_NM', 'FLIGHT_ID2', 'LATITUDE_ORIG', 'LONGITUDE_ORIG', 'OP_CARRIER_AIRLINE_ID', 'DEST_CITY_NAME', 'CALENDAR_DAY']

cols_to_remove = cols_to_remove1 + cols_to_remove2

train_df = train_df.drop(*cols_to_remove)
test_df = test_df.drop(*cols_to_remove)

train_df_RC = train_df.withColumn("DEP_DEL15", \
              F.when(train_df["CANCELLED"] == 1, 1).otherwise(train_df["DEP_DEL15"]))
test_df_RC = test_df.withColumn("DEP_DEL15", \
              F.when(test_df["CANCELLED"] == 1, 1).otherwise(test_df["DEP_DEL15"]))

train_df_RC = train_df_RC.fillna(0, subset='DEP_DEL15')
test_df_RC = test_df_RC.fillna(0, subset='DEP_DEL15')
drop_corr_cols = ['QUARTER', 'DISTANCE', 'DISTANCE_GROUP', 'NTH_FLIGHTS', 'VIS1', 'DEW1', 'GA13', 'KA13', 'CW14', 'CH12','CH15', 'CT11', 'AO11', 'AO12', 'AA12', 'CO12', 'CG11', 'CW11', 'CW14']
train_df_RC = train_df_RC.drop(*drop_corr_cols)
test_df_RC = test_df_RC.drop(*drop_corr_cols)

null_cols_train = ['AW11', 'AW12', 'KA14', 'MD12', 'MD14', 'MD16', 'OC12', 'AU17']
train_df_RC = train_df_RC.drop(*null_cols_train)
test_df_RC = test_df_RC.drop(*null_cols_train)

str_fill = ['TAIL_NUM','STATION', 'GF1', 'TMP2', 'DEW2', 'SLP2', 'GA12', 'GA14', 'GA16', 'MA12', 'MA14', 'GD13', 'GD15', 'CH14', 'CH17', 'CT13', 'AO14', 'AA14', 'CB13', 'CB14', 'CG13', 'CW13', 'CW16']

for col in str_fill:
  mode = train_df_RC.where(F.col(col).isNotNull()).groupby(col).count().orderBy(F.desc('count')).first()[col]
  train_df_RC = train_df_RC.fillna(str(mode), subset=col)
  test_df_RC = test_df_RC.fillna(str(mode), subset=col)
  
drop_strs = ['GE11', 'GD12', 'CW13', 'AO13', 'CH17', 'GD13', 'CW16']
train_df_RC = train_df_RC.drop(*drop_strs)
test_df_RC = test_df_RC.drop(*drop_strs)

udf_str = udf(lambda x:x[-2:],StringType())
train_df_RC = train_df_RC.withColumn('TAIL_NUM', udf_str('TAIL_NUM')).drop('STATION')
test_df_RC = test_df_RC.withColumn('TAIL_NUM', udf_str('TAIL_NUM')).drop('STATION')

# COMMAND ----------

# MAGIC %md ### Preprocessing

# COMMAND ----------

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
    * vectorized: spark df, after stringindex and vector assemble, ready for smote
    '''
    if(df_train.select(target_col).distinct().count() != 2):
        raise ValueError("Target col must have exactly 2 classes")
        
    if target_col in num_cols:
        num_cols.remove(target_col)
    
    # index the string cols, except possibly for the label col
    indexers = [StringIndexer(inputCol=column, outputCol=column+index_suffix, handleInvalid = 'keep') for column in list(set(cat_cols))]
    
    # one hot encoder
#     ohes = [OneHotEncoder(inputCol=c + "_idx", outputCol=c+"_class").fit(df) for c in list(set(cat_cols))]
    ohes = [OneHotEncoder(inputCol=c+index_suffix, outputCol=c+"_class") for c in list(set(cat_cols))]

    
    # Impute numeric columns
    imputers = [Imputer(inputCols = num_cols, outputCols = ["{}".format(c) for c in num_cols]).setStrategy("median")]

    # Build the stage for the ML pipeline
    vector_cols = [c+"_class" for c in cat_cols] + ["{}".format(c) for c in num_cols]
    
    model_matrix_stages = indexers  + \
                          ohes + \
                          imputers + \
                          [VectorAssembler(inputCols=vector_cols, outputCol='features')]              
    
    # scale the numeric columns
    scaler = [StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)]

    pipeline = Pipeline(stages=model_matrix_stages+scaler)
    pipeline_fit = pipeline.fit(df_train)
    pos_vectorized_train = pipeline_fit.transform(df_train)
    pos_vectorized_test = pipeline_fit.transform(df_test)

    
    # drop original num cols and cat cols
    drop_cols = num_cols+cat_cols
    
    keep_cols = [a for a in pos_vectorized_train.columns if a not in drop_cols]
#     keep_cols = ['scaledFeatures']
    vectorized_train = pos_vectorized_train.select(*keep_cols).withColumn('label',pos_vectorized_train[target_col]).drop(target_col)
    vectorized_test = pos_vectorized_test.select(*keep_cols).withColumn('label',pos_vectorized_test[target_col]).drop(target_col)

    
#     return vectorized
    return vectorized_train, vectorized_test

# COMMAND ----------

str_cols = [f.name for f in train_df_RC.schema.fields if isinstance(f.dataType, StringType)]
numeric_cols = [f.name for f in train_df_RC.schema.fields if isinstance(f.dataType, DoubleType) or isinstance(f.dataType, IntegerType)]

# COMMAND ----------

vectorized_train, vectorized_test = pre_smote_df_process(train_df_RC, test_df_RC, numeric_cols, str_cols, 'DEP_DEL15')

# COMMAND ----------

vectorized_train_subset = vectorized_train.select(F.col('LOCAL_TIME'),F.col('scaledFeatures'),F.col('label'))
vectorized_test_subset = vectorized_test.select(F.col('LOCAL_TIME'),F.col('scaledFeatures'),F.col('label'))

# COMMAND ----------

selector = UnivariateFeatureSelector(featuresCol='scaledFeatures',outputCol="selectedFeatures", selectionMode='fwe')
selector.setFeatureType("continuous").setLabelType("categorical")
feat_sel = selector.fit(vectorized_train_subset)

# COMMAND ----------

vec_subset_train = feat_sel.transform(vectorized_train_subset)
vec_subset_test = feat_sel.transform(vectorized_test_subset)

# COMMAND ----------

train_vec_subset = vec_subset_train.select(F.col('LOCAL_TIME'),F.col('selectedFeatures'),F.col('label'))
test_vec_subset = vec_subset_test.select(F.col('LOCAL_TIME'),F.col('selectedFeatures'),F.col('label'))

# COMMAND ----------

train_vec_subset = train_vec_subset.withColumnRenamed('selectedFeatures', 'features')
test_vec_subset = test_vec_subset.withColumnRenamed('selectedFeatures', 'features')

# COMMAND ----------

display(train_vec_subset)

# COMMAND ----------

display(test_vec_subset.filter(F.col('label').isNull()))

# COMMAND ----------

# read data
train_vec_subset = spark.read.parquet(f"{blob_url}/train_selected/*")
test_vec_subset = spark.read.parquet(f"{blob_url}/test_selected/*")

# COMMAND ----------

train_vec_subset = train_vec_subset.withColumnRenamed('selectedFeatures', 'features').orderBy('LOCAL_TIME').withColumn('index', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)
test_vec_subset = test_vec_subset.withColumnRenamed('selectedFeatures', 'features').orderBy('LOCAL_TIME').withColumn('index', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)

# COMMAND ----------

display(train_vec_subset)

# COMMAND ----------

# Ensure that the code from the feature engineering notebook leaves no nulls as these were flights that needed imputing with 0
# test_vec_subset = test_vec_subset.fillna(0, subset='label')

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Cross Validation and Training

# COMMAND ----------

# 3 folds
n_rows = 19153585
end1 = math.ceil(n_rows/3)
end_train = math.ceil(end1*0.8)

end2 = end1 + math.ceil(n_rows/3)
end_train2 = math.ceil(end2*0.8)

end_train3 = end2+1 + math.ceil((n_rows-end2)*.8)

# COMMAND ----------

train_fold1 = train_vec_subset.where(F.col("index").between(0, end_train)).cache()
test_fold1 = train_vec_subset.where(F.col("index").between(end_train+1, end1)).cache()

train_fold2 = train_vec_subset.where(F.col("index").between(end1+1, end_train2)).cache()
test_fold2 = train_vec_subset.where(F.col("index").between(end_train2+1, end2)).cache()

train_fold3 = train_vec_subset.where(F.col("index").between(end2+1, end_train3)).cache()
test_fold3 = train_vec_subset.where(F.col("index").between(end_train3+1, n_rows)).cache()

# COMMAND ----------

trains = [train_fold1, train_fold2, train_fold3]
tests = [test_fold1, test_fold2, test_fold3]

# COMMAND ----------

# MAGIC %md
# MAGIC ### GBT

# COMMAND ----------

cv_scores = []
depth_params = [1, 5, 10]
best_depth = 0
best_PR = 0

for train, test, depth in zip(trains, tests, depth_params):
  gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=3, maxDepth=depth)
  gbt_model = gbt.fit(train)
  predictions = gbt_model.transform(test)
  evaluator = BinaryClassificationEvaluator()
  PR = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
  if PR >= best_PR:
    best_PR = PR
    best_depth = depth
  cv_scores.append(PR)
  print(f'maxDepth={depth} - Area Under PR: {PR}')

# COMMAND ----------

print(f'Average CV score for GBT: {statistics.mean(cv_scores)}')
print(f'Best PR score achieved with maxDepth={best_depth}')

# COMMAND ----------

# Use best maxDepth
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10, maxDepth=best_depth)
gbt_model = gbt.fit(train_vec_subset)

# COMMAND ----------

gbt_predictions = gbt_model.transform(test_vec_subset)

# COMMAND ----------

pred1 = gbt_predictions.select('prediction').rdd.map(lambda r: r[0]).collect()
label1 = gbt_predictions.select('label').rdd.map(lambda r: r[0]).collect()
accuracy1 = sum(np.array(pred1) == np.array(label1))/len(label1)
print(f'Accuracy of GBT model: {accuracy1:.3f}')

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print(f'Area under PR for GBT: {evaluator.evaluate(gbt_predictions, {evaluator.metricName: "areaUnderPR"})}')

# COMMAND ----------

gbt_conf_mat = confusion_matrix(label1, pred1)
gbt_conf_mat_normalized = gbt_conf_mat.astype('float') / len(pred1)
sns.heatmap(gbt_conf_mat_normalized, annot=True,fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# COMMAND ----------

from pyspark.mllib.evaluation import RankingMetrics
predictionAndLabel = [pred1, label1]
predictionAndLabels = sc.parallelize(predictionAndLabel)
metrics = RankingMetrics(predictionAndLabels)

# COMMAND ----------

metrics.meanAveragePrecision

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gaussian Naive Bayes

# COMMAND ----------

cv_scores2 = []
reg_smooth = [0.5, 1, 1.5]
best_PR_nb = 0
best_smooth = 0
for train, test, smooth in zip(trains, tests, reg_smooth):
  nb = NaiveBayes(smoothing=smooth, modelType="gaussian")
  nb_model = nb.fit(train)
  predictions = nb_model.transform(test)
  evaluator = BinaryClassificationEvaluator()
  PR = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
  if PR >= best_PR_nb:
    best_PR_nb = PR
    best_smooth = smooth
  cv_scores2.append(PR)
  print(f'smoothing={smooth} - Area Under PR: {PR}')

# COMMAND ----------

print(f'Average CV score for GNB: {statistics.mean(cv_scores2)}')
print(f'Best PR score achieved with smoothing={best_smooth}')

# COMMAND ----------

nb = NaiveBayes(smoothing=best_smooth, modelType="gaussian")
nb_model = nb.fit(train_vec_subset)

# COMMAND ----------

nb_predictions = nb_model.transform(test_vec_subset)

# COMMAND ----------

pred2 = nb_predictions.select('prediction').rdd.map(lambda r: r[0]).collect()
label2 = nb_predictions.select('label').rdd.map(lambda r: r[0]).collect()
accuracy2 = sum(np.array(pred2) == np.array(label2))/len(label2)
print(f'Accuracy of GNB model: {accuracy2:.3f}')

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print(f'Area under PR for LSVC: {evaluator.evaluate(nb_predictions, {evaluator.metricName: "areaUnderPR"})}')

# COMMAND ----------

nb_conf_mat = confusion_matrix(label2, pred2)
nb_conf_mat_normalized = nb_conf_mat.astype('float') / len(pred2)
sns.heatmap(nb_conf_mat_normalized, annot=True,fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# COMMAND ----------

sns.heatmap(nb_conf_mat, annot=True,fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# COMMAND ----------

# MAGIC %md
# MAGIC ### SVM

# COMMAND ----------

cv_scores3 = []
reg_params = [0.0, 0.5, 0.8]
best_PR_svm = 0
best_reg = 0
for train, test, reg in zip(trains, tests, reg_params):
  lsvc = LinearSVC(maxIter=2, regParam=reg)
  lsvc_model = lsvc.fit(train)
  predictions = lsvc_model.transform(test)
  evaluator = BinaryClassificationEvaluator()
  PR = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
  if PR >= best_PR_svm:
    best_PR_svm = PR
    best_reg = reg
  cv_scores3.append(PR)
  print(f'regParam={reg} - Area Under PR: {PR}')

# COMMAND ----------

print(f'Average CV score for SVM: {statistics.mean(cv_scores3)}')
print(f'Best PR score achieved with regParam={best_reg}')

# COMMAND ----------

lsvc = LinearSVC(maxIter=5, regParam=best_reg)
lsvc_model = lsvc.fit(train_vec_subset)

# COMMAND ----------

lsvc_predictions = lsvc_model.transform(test_vec_subset)

# COMMAND ----------

pred3 = lsvc_predictions.select('prediction').rdd.map(lambda r: r[0]).collect()
label3 = lsvc_predictions.select('label').rdd.map(lambda r: r[0]).collect()
accuracy3 = sum(np.array(pred3) == np.array(label3))/len(label3)
print(f'Accuracy of SVM model: {accuracy3:.3f}')

# COMMAND ----------

evaluator = BinaryClassificationEvaluator()
print(f'Area under PR for LSVC: {evaluator.evaluate(lsvc_predictions, {evaluator.metricName: "areaUnderPR"})}')

# COMMAND ----------

lsvc_conf_mat = confusion_matrix(label3, pred3)
lsvc_conf_mat_normalized = lsvc_conf_mat.astype('float') / len(pred3)
sns.heatmap(lsvc_conf_mat_normalized, annot=True,fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# COMMAND ----------

sns.heatmap(lsvc_conf_mat, annot=True,fmt='g')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# COMMAND ----------

# MAGIC %md
# MAGIC During our work on this project, we applied a number of concepts from the course. We learned how to perform EDA, feature engineering and run ML models on big data in databricks with built-in functions for plotting. Feature reduction and feature engineering were key in obtaining workable runtimes for joins, data preprocessing, and model training. Scalability was a key issue and knowing the impacts of things like one hot encoding to ensure that no single step took up too much time and proper insight into the algorithms and code was essential to avoid code that could take days to run like SMOTE . Dropping features after EDA and reducing the dimensionality of categorical columns was a key part of our pipeline. After the data was prepared, we tried algorithms like GBT and SVM, which took 2-4 hours to train depending on the cluster size. We learned the challenges of dealing with big data and gained familiarity with the Spark libraries for machine learning. 

# COMMAND ----------


