# Databricks notebook source
# MAGIC %pip install /dbfs/FileStore/sdk/dev/MLCoreSDK-0.4.5-py3-none-any.whl --force-reinstall
# MAGIC %pip install sparkmeasure

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## INSTALL MLCORE SDK

# COMMAND ----------

from sparkmeasure import StageMetrics
from sparkmeasure import TaskMetrics
taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

# DBTITLE 1,Load the YAML config
import yaml
import ast
from MLCORE_SDK import mlclient
from pyspark.sql import functions as F
import pickle

try:
    solution_config = (dbutils.widgets.get("solution_config"))
    solution_config = ast.literal_eval(solution_config)
except Exception as e:
    with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
        solution_config = yaml.safe_load(solution_config)  

# COMMAND ----------

# MAGIC %md
# MAGIC ## PERFORM MODEL TRAINING 

# COMMAND ----------

# DBTITLE 1,Imports
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import time
from sklearn.metrics import *
import json
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from prophet import Prophet
import logging
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import time, json
from utils import utils
from sklearn.metrics import *
logger = spark._jvm.org.apache.log4j
logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# DBTITLE 1,Input from the user
try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

tracking_url = solution_config["general_configs"].get("tracking_url", None)
tracking_url = f"https://{tracking_url}" if tracking_url else None

# JOB SPECIFIC PARAMETERS
input_table_configs = solution_config["train"]["datalake_configs"]["input_tables"]
output_table_configs = solution_config["train"]["datalake_configs"]['output_tables']
model_configs = solution_config["train"]["model_configs"]
feature_columns = solution_config['train']["feature_columns"]
target_columns = solution_config['train']["target_columns"]
test_size = solution_config['train']["test_size"]
horizon = solution_config['train']["horizon"]
frequency = solution_config['train']["frequency"]

# COMMAND ----------

def get_name_space(table_config):
    data_objects = {}
    for table_name, config in table_config.items() : 
        catalog_name = config.get("catalog_name", None)
        schema = config.get("schema", None)
        table = config.get("table", None)

        if catalog_name and catalog_name.lower() != "none":
            table_path = f"{catalog_name}.{schema}.{table}"
        else :
            table_path = f"{schema}.{table}"

        data_objects[table_name] = table_path
    
    return data_objects

# COMMAND ----------

model_configs

# COMMAND ----------

# DBTITLE 1,Update the table paths as needed.
input_table_paths = get_name_space(input_table_configs)
output_table_paths = get_name_space(output_table_configs)

# COMMAND ----------

ft_data = spark.sql(f"SELECT * FROM {input_table_paths['input_1']}")
gt_data = spark.sql(f"SELECT * FROM {input_table_paths['input_2']}")

# COMMAND ----------

ft_data.display()

# COMMAND ----------

gt_data.display()

# COMMAND ----------

try : 
    date_filters = dbutils.widgets.get("date_filters")
    print(f"Input date filter : {date_filters}")
    date_filters = json.loads(date_filters)
except :
    date_filters = {}

try : 
    hyperparameters = dbutils.widgets.get("hyperparameters")
    print(f"Input hyper parameters : {hyperparameters}")
    hyperparameters = json.loads(hyperparameters)
except :
    hyperparameters = {}

print(f"Data filters used in model train : {date_filters}, hyper parameters : {hyperparameters}")

# COMMAND ----------

if date_filters and date_filters['feature_table_date_filters'] and date_filters['feature_table_date_filters'] != {} :   
    ft_start_date = date_filters.get('feature_table_date_filters', {}).get('start_date',None)
    ft_end_date = date_filters.get('feature_table_date_filters', {}).get('end_date',None)
    if ft_start_date not in ["","0",None] and ft_end_date not in  ["","0",None] : 
        print(f"Filtering the feature data")
        ft_data = ft_data.filter(F.col("timestamp") >= int(ft_start_date)).filter(F.col("timestamp") <= int(ft_end_date))

if date_filters and date_filters['ground_truth_table_date_filters'] and date_filters['ground_truth_table_date_filters'] != {} : 
    gt_start_date = date_filters.get('ground_truth_table_date_filters', {}).get('start_date',None)
    gt_end_date = date_filters.get('ground_truth_table_date_filters', {}).get('end_date',None)
    if gt_start_date not in ["","0",None] and gt_end_date not in ["","0",None] : 
        print(f"Filtering the ground truth data")
        gt_data = gt_data.filter(F.col("timestamp") >= int(gt_start_date)).filter(F.col("timestamp") <= int(gt_end_date))

# COMMAND ----------

ft_data.count(), gt_data.count()

# COMMAND ----------

ground_truth_data = gt_data.select([input_table_configs["input_2"]["primary_keys"]] + target_columns)
features_data = ft_data.select([input_table_configs["input_1"]["primary_keys"]] + feature_columns )

# COMMAND ----------

final_df = features_data.join(ground_truth_data, on = input_table_configs["input_1"]["primary_keys"])

# COMMAND ----------

# DBTITLE 1,Converting the Spark df to Pandas df
final_df_pandas = final_df.toPandas()
final_df_pandas.head()

# COMMAND ----------



# COMMAND ----------

final_df_pandas.shape

# COMMAND ----------

# DBTITLE 1,Dropping the null rows in the final df
final_df_pandas.dropna(inplace=True)

# COMMAND ----------

final_df_pandas.display()

# COMMAND ----------

# DBTITLE 1,Spliting the Final df to test and train dfs
# Split the Data to Train and Test
# X_train, X_test, y_train, y_test = train_test_split(final_df_pandas[feature_columns], final_df_pandas[target_columns], test_size=test_size, random_state = 0)


# Split the Data to Train and Test
traindf = final_df_pandas.iloc[int(final_df_pandas.shape[0] * test_size):]
testdf = final_df_pandas.iloc[:int(final_df_pandas.shape[0] * test_size)]
lr = Prophet()
lr.fit(traindf)
X_train= traindf["epoc"]
y_train= traindf["avg_heart_rate"]
X_test= testdf["epoc"]
y_test= testdf["avg_heart_rate"]

# #Fetching train and test predictions from model
# train_pred = model.predict(traindf)
# test_pred = model.predict(testdf)

# COMMAND ----------

# # Build a Scikit learn pipeline
# pipe = Pipeline([
#     ('regressor',LinearRegression())
# ])
X_train_np = X_train.to_numpy()
X_test_np = X_test.to_numpy()

# model = Prophet()
# model.fit(traindf)

# COMMAND ----------

# Fit the pipeline
# lr = pipe.fit(X_train_np, y_train)

# COMMAND ----------

# Predict it on Test and calculate metrics
y_pred = lr.predict(X_test_np)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred,squared = False)

# COMMAND ----------

test_metrics = {"mae":mae, "mse":mse, "r2":r2,"rmse":rmse}
test_metrics

# COMMAND ----------

# Predict it on Test and calculate metrics
y_pred_train = lr.predict(X_train_np)
mae = mean_absolute_error(y_train, y_pred_train)
mse = mean_squared_error(y_train, y_pred_train)
r2 = r2_score(y_train, y_pred_train)
rmse = mean_squared_error(y_train, y_pred_train,squared = False)

# COMMAND ----------

train_metrics = {"mae":mae, "mse":mse, "r2":r2,"rmse":rmse}
train_metrics

# COMMAND ----------

pred_train = pd.concat([X_train, y_train], axis = 1)
pred_test = pd.concat([X_test, y_test], axis = 1)

# COMMAND ----------

y_pred_train = lr.predict(X_train_np)
y_pred = lr.predict(X_test_np)

# COMMAND ----------

# pred_train

# COMMAND ----------

# MAGIC %md
# MAGIC ## SAVE PREDICTIONS TO HIVE

# COMMAND ----------

pred_train["prediction"] = y_pred_train
pred_train["dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE"] = "train"
pred_test["prediction"] = y_pred
pred_test["dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE"] = "test"

# COMMAND ----------

final_train_output_df = pd.concat([pred_train, pred_test])
train_output_df = spark.createDataFrame(final_train_output_df)

# COMMAND ----------

columns_to_include = feature_columns 

# COMMAND ----------

first_row_dict = X_train_np[:5]

# COMMAND ----------

def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats = (
        "MM-dd-yyyy",
        "dd-MM-yyyy",
        "MM/dd/yyyy",
        "yyyy-MM-dd",
        "M/d/yyyy",
        "M/dd/yyyy",
        "MM/dd/yy",
        "MM.dd.yyyy",
        "dd.MM.yyyy",
        "yyyy-MM-dd",
        "yyyy-dd-MM",
    )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

now = datetime.now()
date = now.strftime("%m-%d-%Y")
train_output_df = train_output_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
train_output_df = train_output_df.withColumn("date", F.lit(date))
train_output_df = train_output_df.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in train_output_df.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  train_output_df = train_output_df.withColumn("id", F.row_number().over(window))

# COMMAND ----------

db_name = output_table_configs["output_1"]["schema"]
table_name = output_table_configs["output_1"]["table"]
catalog_name = output_table_configs["output_1"]["catalog_name"]
output_path = output_table_paths["output_1"]

# Get the catalog name from the table name
if catalog_name and catalog_name.lower() != "none":
  spark.sql(f"USE CATALOG {catalog_name}")
else:
  spark.sql(f"USE CATALOG hive_metastore")

# Create the database if it does not exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
print(f"HIVE METASTORE DATABASE NAME : {db_name}")

train_output_df.createOrReplaceTempView(table_name)

feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]

if not any(feature_table_exist):
  print(f"CREATING SOURCE TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} AS SELECT * FROM {table_name}")
else :
  print(F"UPDATING SOURCE TABLE")
  spark.sql(f"INSERT INTO {output_path} SELECT * FROM {table_name}")

if catalog_name and catalog_name.lower() != "none":
  output_1_table_path = output_path
else:
  output_1_table_path = spark.sql(f"desc {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

print(f"Features Hive Path : {output_1_table_path}")

# COMMAND ----------

if input_table_configs["input_1"]["catalog_name"]:
    feature_table_path = input_table_paths["input_1"]
else:
    feature_table_path = spark.sql(f"desc formatted {input_table_paths['input_1']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

if input_table_configs["input_2"]["catalog_name"]:
    gt_table_path = input_table_paths["input_2"]
else:
    gt_table_path = spark.sql(f"desc formatted {input_table_paths['input_2']}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]


print(feature_table_path, gt_table_path)

# COMMAND ----------

stagemetrics.end()
taskmetrics.end()

stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## REGISTER MODEL IN MLCORE

# COMMAND ----------

from MLCORE_SDK import mlclient

# COMMAND ----------

train_data_date_dict = {
    "feature_table" : {
        "ft_start_date" : ft_data.select(F.min("timestamp")).collect()[0][0],
        "ft_end_date" : ft_data.select(F.max("timestamp")).collect()[0][0]
    },
    "gt_table" : {
        "gt_start_date" : gt_data.select(F.min("timestamp")).collect()[0][0],
        "gt_end_date" : gt_data.select(F.max("timestamp")).collect()[0][0]        
    }
}

# COMMAND ----------



# COMMAND ----------

model_name = f"{model_configs.get('model_registry_params').get('catalog_name')}.{model_configs.get('model_registry_params').get('schema_name')}.{model_configs.get('model_params').get('model_name')}"
model_name

if(flag):
    flag = False
    mlclient.log(
        operation_type="job_run_add", 
        session_id = sdk_session_id, 
        dbutils = dbutils, 
        request_type = "train", 
        job_config = 
        {
            "table_name" : output_table_configs["output_1"]["table"],
            "model_name" : model_name,
            "feature_table_path" : feature_table_path,
            "ground_truth_table_path" : gt_table_path,
            "feature_columns" : feature_columns,
            "target_columns" : target_columns,
            "model" : Prophet,
            "model_runtime_env" : "python",
            "reuse_train_session" : False
        },
        tracking_env = env,
        tracking_url = tracking_url,
        spark = spark,
        verbose = True,
        )
mlflow.end_run()

mlclient.log(operation_type = "register_model",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    model = lr,
    model_name = model_name, #model_configs["model_params"]["model_name"],
    model_runtime_env = "python",
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    feature_table_path = feature_table_path,
    ground_truth_table_path = gt_table_path,
    train_output_path = output_1_table_path,
    train_output_rows = train_output_df.count(),
    train_output_cols = train_output_df.columns,
    table_schema=train_output_df.schema,
    column_datatype = train_output_df.dtypes,
    feature_columns = feature_columns,
    target_columns = target_columns,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal",
    train_data_date_dict = train_data_date_dict,
    compute_usage_metrics = compute_metrics,
    taskmetrics = taskmetrics,
    stagemetrics = stagemetrics,
    tracking_env = env,
    horizon=horizon,
    frequency=frequency,
    example_input = first_row_dict,
    # register_in_feature_store=True,
    model_configs = model_configs,
    tracking_url = tracking_url,
    verbose = True)

# COMMAND ----------

# DBTITLE 1,Registering the model in MLCore


# COMMAND ----------

# try :
#     media_artifacts_path = mlclient.log(operation_type = "get_media_artifact_path",
#         sdk_session_id = sdk_session_id,
#         dbutils = dbutils)
#     dbutils.notebook.run(
#         "Model_Test", 
#         timeout_seconds = 5000, 
#         arguments = 
#         {
#             "feature_columns" : ",".join(map(str,feature_columns)),
#             "target_columns" : ",".join(map(str,target_columns)), #json dumps
#             "model_data_path" : train_output_dbfs_path,
#             "model_name": model_name,
#             "media_artifacts_path" : media_artifacts_path,
#         })
# except Exception as e:
#     print(f"Exception while triggering model testing notebook : {e}")

# COMMAND ----------

# try: 
#     #define media artifacts path
#     media_artifacts_path = mlclient.log(operation_type = "get_media_artifact_path",
#         sdk_session_id = sdk_session_id,
#         dbutils = dbutils)
    
#     print(media_artifacts_path)

#     custom_notebook_result = dbutils.notebook.run(
#         "Model_eval",
#         timeout_seconds = 0,
#         arguments = 
#         {
#             "date_column" : date_column,
#             "feature_columns" : ",".join(map(str,feature_columns)),
#             "target_columns" : ",".join(map(str,target_columns)), #json dumps
#             "model_data_path" : train_output_dbfs_path,
#             "model_name": model_name,
#             "media_artifacts_path" : media_artifacts_path,
#         })
# except Exception as e:
#     print(f"Exception while triggering model eval notebook : {e}")
