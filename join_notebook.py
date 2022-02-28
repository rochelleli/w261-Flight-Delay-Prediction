# Databricks notebook source
# MAGIC %md 
# MAGIC 
# MAGIC # w261 Final Project - Flight Delay Prediction
# MAGIC [Team 09]
# MAGIC 
# MAGIC [Simran Bhatia,Harinandan Srikanth, Mackenzie Lee,Rochelle Li, Ashwini Bhingare]
# MAGIC 
# MAGIC Summer 2021, section [w261_su21_section2]
# MAGIC 
# MAGIC 
# MAGIC The main goal of this project is to implement a model that predicts whether a flight will be delayed by 15 minutes or more two hours prior to the scheduled flight time.
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC ## Table of Contents
# MAGIC * [Section 1 - Question Formulation](#Section1)
# MAGIC * [Section 2 - Algorithm Explanation](#Section2)
# MAGIC * [Section 3 - EDA & Challenges](#Section3)
# MAGIC * [Section 4 - Algorithm Implementation](#Section4)
# MAGIC * [Section 5 - Course Concepts](#Section5)

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Packages

# COMMAND ----------

!pip install geopy
!pip install timezonefinder

# COMMAND ----------

import pyspark.sql.functions as f
from pyspark.sql.types import *
from pyspark.sql import Window
import numpy as np
import ast
import pandas as pd
from math import radians, cos, sin, asin, sqrt

from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder

spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.databricks.io.cache.enabled", "true")

# COMMAND ----------

# Azure Information for writing and reading data from the cloud
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

# MAGIC %md
# MAGIC ### Read in Initial Datasets
# MAGIC 
# MAGIC There are 3 datasets available:
# MAGIC - **Airlines table**: This is a subset of the passenger flight's on-time performance data taken from the TranStats data collection available from the [U.S. Department of Transportation (DOT)](http://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236&DB_Short_Name=On-Time)
# MAGIC   - See also: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ 
# MAGIC   - This data consists of flight data departing from all major US airports for the 2015-2019 timeframe
# MAGIC   - A Data Dictionary for this dataset is located here: https://www.transtats.bts.gov/Glossary.asp?index=C 
# MAGIC - **Weather table**: A weather table  has been pre-downloaded from the [National Oceanic and Atmospheric Administration repository](https://www.ncdc.noaa.gov/orders/qclcd/) to S3 in the form of  parquet files (thereby enabling pushdown querying and efficient joins).
# MAGIC   - The data consists of various weather metrics gathered from various weather stations in the US and internationally from 2015-2019
# MAGIC - **Stations table**: This is a table consisting of details of the stations used to gather weather data

# COMMAND ----------

# Load Data for all Flights
df_airlines = spark.read.parquet("/mnt/mids-w261/datasets_final_project/parquet_airlines_data/*")
display(df_airlines)

# COMMAND ----------

# Load the all data for Weather
df_weather = spark.read.parquet("/mnt/mids-w261/datasets_final_project/weather_data/*")
display(df_weather)

# COMMAND ----------

# Load the stations data
df_stations = spark.read.parquet("/mnt/mids-w261/datasets_final_project/stations_data/*")
display(df_stations)

# COMMAND ----------

print(f'Number of rows in the Weather data (All countries): {df_weather.count()}')
print(f'Number of rows in the Airlines data: {df_airlines.count()}')
print(f'Number of rows in the Stations data: {df_stations.count()}')

# The Airlines data only has flights originating in the US
df_weather = df_weather.withColumn('COUNTRY', f.substring(f.regexp_replace(f.col("NAME"), " ", ""), -2, 2)).filter(f.upper(f.col('COUNTRY')) == "US")
print(f'Number of rows in the Weather data (Only US): {df_weather.count()}')

# COMMAND ----------

# MAGIC %md
# MAGIC The stations data provides very little new information as many of the columns are already captured in the weather data like latitude, longitude, and neighbor state. The other columns are ID representations and provide little new information so we will not use the stations data keeping performance and scalability in mind.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert Departure Times from Local Time to UTC Time in Airlines Data
# MAGIC 
# MAGIC The weather data has date timestamps in UTC format, while the airlines data has scheduled flight times (_CRS_DEP_TIME_ column) in local time depending on where the airport is located. We will convert the airlines flight time to UTC time to ensure matching times when we join.

# COMMAND ----------

display(df_airlines)

# COMMAND ----------

# Select all origin city names in the airlines data
city_dict = {}
cities = df_airlines.select('ORIGIN_CITY_NAME').distinct().rdd.map(lambda r: r[0]).collect()

# Map each city to a timezone
geolocator = Nominatim(user_agent="W261-team09")
def map_timezone(location):
  '''
  Input: city name
  Output: timezone  
  '''
  location = location.split('/')[-1]
  coords = geolocator.geocode(location)
  obj = TimezoneFinder()
  timezone = obj.timezone_at(lng=coords.longitude, lat=coords.latitude)
  return timezone

# Create a dictionary of cities in the airlines data to a timezone to be used to map to UTC time
for city in cities:
  alt_city = city
  if city=='Iron Mountain/Kingsfd, MI':
    alt_city = 'Iron Mountain, MI'
  tz = map_timezone(alt_city)
  city_dict[city] = tz

# COMMAND ----------

def map_time(time):
  '''Formats the time to be spark friendly by ensuring the time is in the form: hh:mm:ss'''
  time = str(time)
  temp = time[:-2] +':' + time[-2:] +':' + '00'
  return temp.zfill(8)


def map_timezone(location):
  '''Return the timezone of a given city'''
  return city_dict.get(location, 'None')

# Create UDFs to be used to make new columns in spark
udfTimeMap = f.udf(map_time, StringType())
udfTimezoneMap = f.udf(map_timezone, StringType())

# Create a temp_time column that is the formatted time string
# Create a flight timestamp column by concatenating the date column with the temp_time column created that can be converted to a timestamp type
df_airlines = df_airlines.withColumn('temp_time', udfTimeMap(f.col("CRS_DEP_TIME"))).withColumn('FLIGHT_TIMESTAMP',f.concat('FL_DATE', f.lit('T'), f.col('temp_time'))).withColumn("FLIGHT_TIMESTAMP",f.to_timestamp(f.col("FLIGHT_TIMESTAMP"))).withColumn("FL_DATE",f.to_date(f.col("FL_DATE"))).drop(*['temp_time'])

# Create a timezone column using the UDF created above by mapping a city to a timestamp using the dictionary from the cell above
# Create a LOCAL_TIME column that is the UTC time of the flight
df_airlines = df_airlines.withColumn('TIMEZONE', udfTimezoneMap(f.col('ORIGIN_CITY_NAME'))).withColumn('LOCAL_TIME', f.from_utc_timestamp(f.split(df_airlines.FLIGHT_TIMESTAMP,'\+')[0],f.col('TIMEZONE')))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Map the Cities in the Airlines Data to a Latitude and Longitude
# MAGIC 
# MAGIC The join is dependent on both time and location since weather is only relevant to certain areas at certain times. The _LOCAL_TIME_ column handles the time aspect, but we will map the cities to a latitude and longitude. The weather data already has latitude and longitude columns and with these two extra columns in the airlines data, we will be able to calculate the distance from the airport city to the weather data location. We can then select the row of the weather data that is most relevant to the flight based on time and distance.

# COMMAND ----------

loc_dict = {}

def get_latlong(location):
  '''Return the latitude and longitude given a city'''
  location = location.split('/')[-1]
  location = geolocator.geocode(location)
  return (location.latitude, location.longitude)

# Map the cities in the airlines data to a latitude and longitude in a dictionary
for city in cities:
  alt_city = city
  # Some cities need to be specially handled 
  if city=='Iron Mountain/Kingsfd, MI':
    alt_city = 'Iron Mountain, MI'
  tz = get_latlong(alt_city)
  loc_dict[city] = tz

# COMMAND ----------

def map_lat(city):
  '''Return the latitude of a city'''
  return loc_dict[city][0]

def map_long(city):
    '''Return the longitude of a city'''
  return loc_dict[city][1]

# Create UDFs for spark to use
udfLatMap = F.udf(map_lat, StringType())
udfLongMap = F.udf(map_long, StringType())

# Create a latitude column and a longitude column based on the ORIGIN_CITY_NAME column
airlines_prejoin = df_airlines.withColumn('LATITUDE_ORIG', udfLatMap(f.col("ORIGIN_CITY_NAME"))).withColumn('LONGITUDE_ORIG', udfLongMap(f.col("ORIGIN_CITY_NAME")))

# COMMAND ----------

display(airlines_prejoin)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop Columns in the Weather and Airlines Data that are >=95% Null
# MAGIC 
# MAGIC We will make the assumption that columns that are >=95% null contain little to no useful data and can be dropped. We do this to improve performance during the join so there is less data to shuffle around. Based on research, >60% missing values seemed to be a common threshold to drop missing columns. A more important consideration is the domain knowledge and knowing the importance of the actual column. The columns with the most missing values come from the weather data and seem to be metrics gathered from the stations that are not particularly more valuable information-wise than other columns. 
# MAGIC 
# MAGIC Sources:
# MAGIC - https://discuss.analyticsvidhya.com/t/what-should-be-the-allowed-percentage-of-missing-values/2456/7
# MAGIC - https://stats.stackexchange.com/questions/149140/how-much-missing-data-is-too-much-multiple-imputation-mice-r
# MAGIC - https://www.mastersindatascience.org/learning/how-to-deal-with-missing-data/

# COMMAND ----------

# Get the count of values in each dataset as we will use this to calculate the percent of missing values in each column
airlines_count = airlines_prejoin.count()
weather_count = weather_prejoin.count()

# COMMAND ----------

# Fill in empty rows with None so they get counted later on 
airlines_prejoin = airlines_prejoin.select([f.when(f.col(c)=="",None).otherwise(f.col(c)).alias(c) for c in airlines_prejoin.columns])
weather_prejoin = weather_prejoin.select([f.when(f.col(c)=="",None).otherwise(f.col(c)).alias(c) for c in weather_prejoin.columns])

# COMMAND ----------

# Create a df that shows the percent of missing values in each column for the airlines and weather data
frac_missing_airlines = airlines_prejoin.agg(*[(f.count(f.when(f.isnull(c), c))/airlines_count).alias(c) for c in airlines_prejoin.columns])
display(frac_missing_airlines)

# COMMAND ----------

frac_missing_weather = weather_prejoin.agg(*[(f.count(f.when(f.isnull(c), c))/weather_count).alias(c) for c in weather_prejoin.columns])
display(frac_missing_weather)

# COMMAND ----------

# Get the list of columns that have >95% null values in the airlines and weather data
threshold = 0.95
null_cols_air = [key for (key,value) in frac_missing_airlines.collect()[0].asDict().items() if value >= threshold]
null_cols_weather = [key for (key,value) in frac_missing_weather.collect()[0].asDict().items() if value >= threshold]

print(f'{len(null_cols_air)} columns with >={threshold*100}% null values')
print(f'{len(null_cols_weather)} columns with >={threshold*100}% null values')

# COMMAND ----------

# Drop the columns from the airlines and weather data and create new dfs to be saved to the cloud
airlines_prejoin_tosave = airlines_prejoin.drop(*null_cols_air)
weather_prejoin_tosave = weather_prejoin.drop(*null_cols_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the Prejoin Weather and Airlines Data with Dropped Null Columns to Cloud Storage for Quick Access

# COMMAND ----------

# Save the dfs to the cloud for easy access later
airlines_prejoin_tosave.write.parquet(f"{blob_url}/airlines_prejoin")
weather_prejoin_tosave.write.parquet(f"{blob_url}/weather_prejoin")

# COMMAND ----------

airlines_prejoin = spark.read.parquet(f"{blob_url}/airlines_prejoin/*")
weather_prejoin = spark.read.parquet(f"{blob_url}/weather_prejoin/*")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Map Each Weather Row to the Closest Origin City from the List of Cities in the Airlines Data
# MAGIC 
# MAGIC Now that we have latitude and longitude in both datasets, we can use these columns to calculate the distance (in KM) for each weather row to each city in the list of available cities in the airlines data. We will then select the closest city as this will be used as a join field. The distance metric will be Haversine distance that [determines the great-circle distance between two points on a sphere given their longitudes and latitudes](https://en.wikipedia.org/wiki/Haversine_formula).

# COMMAND ----------

display(weather_prejoin)

# COMMAND ----------

def map_closest(lat, long):
  '''Given a latitude and longitude in the weather data, return the closest city in the airlines data based on Haversine distance'''
  def haversine(lat1, lon1, lat2, lon2):
    '''Return the haversine distance in KM given two latitude and longitude coordinates'''
    R = 6372.8 # KM

    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))

    return R * c
  closest_city = ''
  closest_distance = 100000
  for city, loc in loc_dict.items():
    dist = haversine(float(lat), float(long), loc[0], loc[1])
    if dist <= closest_distance:
      closest_city = city
      closest_distance = dist
  return closest_city
    
# Create a new column in the weather data that represents the closest city based on haversine distance
udfCityMap = f.udf(map_closest, StringType())
weather_prejoin = weather_prejoin.withColumn("CLOSEST_CITY", udfCityMap(f.col("LATITUDE"), f.col('LONGITUDE')))

# COMMAND ----------

display(weather_prejoin)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Drop Unneeded Created Columns Before Joining
# MAGIC 
# MAGIC Unneeded columns mean columns that provide no extra data and were typically created during data processing/feature engineering to map the join.

# COMMAND ----------

# These columns were created as temporary columns to create the important UTC time and closest city columns
to_drop_airlines = ['FLIGHT_TIMESTAMP']
to_drop_weather = ['FORMATTED_DATE', 'MAPPED_CITY']

airlines_prejoin = airlines_prejoin.drop(*to_drop_airlines)
weather_prejoin = weather_prejoin.drop(*to_drop_weather)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join the Weather and Airlines by Calendar Date and Closest City
# MAGIC 
# MAGIC Join the weather and airlines data based on the day and closest city. 

# COMMAND ----------

# Extract Calendar Day from the date column to use as the join 
# Run it on the airlines data in case the local timestamp may have changed the calendar day
airlines_prejoin = airlines_prejoin.withColumn("CALENDAR_DAY_AIRLINES",f.to_date(f.col("LOCAL_TIME")))
weather_prejoin = weather_prejoin.withColumn("CALENDAR_DAY_WEATHER",f.to_date(f.col("DATE")))

# COMMAND ----------

display(airlines_prejoin)

# COMMAND ----------

display(weather_prejoin)

# COMMAND ----------

# Join the airlines and weather data by the same calendar day and city
# Use a left join on airlines data to keep all the flight data as this is the most important data
join_all = airlines_prejoin.join(weather_prejoin,
                                 (airlines_prejoin.CALENDAR_DAY_AIRLINES==weather_prejoin.CALENDAR_DAY_WEATHER) & (airlines_prejoin.ORIGIN_CITY_NAME==weather_prejoin.CLOSEST_CITY),
                                 how='left')

# COMMAND ----------

# Save the full join data to cloud
join_all.write.parquet(f"{blob_url}/df_joined")

# COMMAND ----------

df_joined = spark.read.parquet(f"{blob_url}/df_joined/*")
display(df_joined)

# COMMAND ----------

print(f'Number of rows in fully joined Weather+Airlines data by origin city: {df_joined.count()}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate the Difference in Time between the Weather and Airlines Timestamps in Hours
# MAGIC 
# MAGIC Since we will be predicting flight delay>15min two hours before scheduled flight time, we only want to keep rows that fit in the interval 2-4 hours before scheduled flight time. We make the latest cutoff 4 hours as weather can be fickle and it is very rare for weather to be the same 6 hours later. 4 hours before scheduled flight time was a reasonable cutoff time. We will calculate the difference in hours between the weather data and flight data and then used this field to filter the data.

# COMMAND ----------

df_joined = df_joined.withColumn('FLIGHT_HOUR_DIFF', (f.unix_timestamp(f.col('LOCAL_TIME')) - f.unix_timestamp(f.col('DATE')))/3600)
#Filter rows that are between 2-4 hours before the scheduled flight time (OLD)
df_filt = df_joined.filter((f.col('FLIGHT_HOUR_DIFF') <= 4) & (f.col('FLIGHT_HOUR_DIFF') >= 2))

#Filter rows that are 2 hours before the scheduled flight time
# df_filt = df_joined.filter(f.col('FLIGHT_HOUR_DIFF') >= 2)

# COMMAND ----------

display(df_joined)

# COMMAND ----------

df_joined.select('FLIGHT_ID').distinct().count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Filter Rows in the Joined Data to those with a Time Difference Between 2-4 Hours Before the Scheduled Flight Departure

# COMMAND ----------

# Create a unique flight identifier that will be used later to select the most recent weather row
df_filt = df_filt.withColumn('FLIGHT_ID', f.concat(f.col('OP_CARRIER'), f.col('OP_CARRIER_FL_NUM')))

# COMMAND ----------

display(airlines_prejoin)

# COMMAND ----------

# We only need to keep one row per flight that has the most relevant weather data that fits 2-4 hours before scheduled flight time
num_flights_original = airlines_prejoin.withColumn('FLIGHT_ID', f.concat(f.col('OP_CARRIER'), f.col('OP_CARRIER_FL_NUM'))).dropDuplicates(['CALENDAR_DAY_AIRLINES', 'FLIGHT_ID']).count()
print(f'Number of flights in the original airlines data: {num_flights_original}')

# COMMAND ----------

num_flights_filt = df_filt.dropDuplicates(['CALENDAR_DAY_AIRLINES', 'FLIGHT_ID']).count()
print(f'Number of flights in the filtered data: {num_flights_filt}')

# COMMAND ----------

# MAGIC %md
# MAGIC Since the number of flights in the filtered data is less than the number of flights in the original data, we know some flights do not have weather data available at least 2 hours before the scheduled flight time.

# COMMAND ----------

# keep the most recent row where the weather time is within 2-4 hours before flight time
w = Window.partitionBy(['CALENDAR_DAY_AIRLINES', 'FLIGHT_ID'])
df_min = df_filt.withColumn('minWeather', f.min('FLIGHT_HOUR_DIFF').over(w)).where(f.col('FLIGHT_HOUR_DIFF')==f.col('minWeather')).drop('minWeather').dropDuplicates(['CALENDAR_DAY_AIRLINES', 'FLIGHT_ID'])
display(df_min)

# COMMAND ----------

# Create a unique flight identifier that will be used later to select the most recent weather row
df_filt = df_filt.withColumn('FLIGHT_ID', f.concat(f.col('OP_CARRIER'), f.col('OP_CARRIER_FL_NUM')))

# keep the most recent row where the weather time is within 2-4 hours before flight time
w = Window.partitionBy(['CALENDAR_DAY_AIRLINES', 'FLIGHT_ID'])
df_min = df_filt.withColumn('minWeather',
                            f.min('FLIGHT_HOUR_DIFF').over(w)).where(f.col('FLIGHT_HOUR_DIFF')==f.col('minWeather')).drop('minWeather').dropDuplicates(['CALENDAR_DAY_AIRLINES',
                            'FLIGHT_ID'])
display(df_min)

# COMMAND ----------

num_rows = df_min.count()
print(f'Number of Rows in the filtered, joined Data: {num_rows}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Write and Read Fully Joined and Filtered Data in the Cloud

# COMMAND ----------

df_min.write.parquet(f"{blob_url}/df_final")

# COMMAND ----------

df_final = spark.read.parquet(f"{blob_url}/df_final/*")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check for Any Flights that Don't have Weather data
# MAGIC 
# MAGIC There are flights based on the day + city that do not have any weather data 2-4 hours before scheduled flight time. These flights need to be added back to the data.

# COMMAND ----------

df_final_flights = df_final.withColumn('FLIGHT_ID2', f.concat(f.col('OP_CARRIER'), f.col('OP_CARRIER_FL_NUM'), f.col('FL_DATE'))).select('FLIGHT_ID2').distinct().rdd.map(lambda r: r[0]).collect()

# COMMAND ----------

display(df_final)

# COMMAND ----------

airlines_prejoin_flights = airlines_prejoin.withColumn('FLIGHT_ID2', f.concat(f.col('OP_CARRIER'), f.col('OP_CARRIER_FL_NUM'), f.col('FL_DATE'))).select('FLIGHT_ID2').distinct().rdd.map(lambda r: r[0]).collect()

# COMMAND ----------

# yields flights in df_final_flights that aren't in airlines_prejoin_flights
missing_flights = np.setdiff1d(airlines_prejoin_flights,df_final_flights)

# COMMAND ----------

print(f'{len(missing_flights)} flights missing in the filtered data that need to be added back to the final data')

# COMMAND ----------

missing = list(missing_flights)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find the Rows of the Missing Flights in the Pre-filtered Data to see if there is any weather data available

# COMMAND ----------

df_joined = df_joined.withColumn('FLIGHT_ID2', f.concat(f.col('OP_CARRIER'), f.col('OP_CARRIER_FL_NUM'), f.col('FL_DATE')))

# COMMAND ----------

df_missing = df_joined.filter(f.col('FLIGHT_ID2').isin(missing)) 

# COMMAND ----------

df_missing.write.parquet(f"{blob_url}/df_missing2")

# COMMAND ----------

df_missing = spark.read.parquet(f"{blob_url}/df_missing2/*")

# COMMAND ----------

display(df_missing)

# COMMAND ----------

# Keep one unique row per flight
df_missing_temp = df_missing.dropDuplicates([ 'FLIGHT_ID2'])

# COMMAND ----------

# Fill in null for weather info for missing airlines data
# These values can be imputed later on
weather_cols = ['STATION',  'DATE',  'SOURCE',  'LATITUDE',  'LONGITUDE',  'ELEVATION',  'NAME',  'REPORT_TYPE',  'CALL_SIGN',  'QUALITY_CONTROL',  'WND',  'CIG',  'VIS',  'TMP',  'DEW',  'SLP',  'AW1',  'GA1',  'GA2',  'GA3',  'GE1',  'GF1',  'KA1',  'KA2',  'MA1',  'MD1',  'OC1',  'REM',  'EQD',  'GD1',  'CH1',  'CT1',  'AO1',  'GD2',  'AA1',  'CT3',  'AU1',  'GD3',  'CG3',  'CT2',  'CG2',  'CO1',  'CB1',  'CG1',  'CW1',  'CALENDAR_DAY',  'COUNTRY',  'CLOSEST_CITY',  'CALENDAR_DAY_WEATHER',  'FLIGHT_HOUR_DIFF']
for col in weather_cols:
  df_missing_temp = df_missing_temp.withColumn(col, f.lit(None))


# COMMAND ----------

display(df_missing_temp)

# COMMAND ----------

# Ensure that there are the same number of columns and that they are in order before combining the two dataframes
df_final= df_final.withColumnRenamed('FLIGHT_ID', 'FLIGHT_ID2')
print(f'The columns in the joined data and missing data are the same length and in the same order: {df_missing_temp.columns == df_final.columns}')

# COMMAND ----------

df_full_joined = df_final.union(df_missing_temp)

# COMMAND ----------

df_full_joined.write.parquet(f"{blob_url}/df_last")

# COMMAND ----------

print(f'Number of rows added (i.e. number of missing flights added): {df_missing_temp.count()}')
print(f'Number of rows in full data set: {df_full_joined.count()}')

# COMMAND ----------

display(df_full_joined)
