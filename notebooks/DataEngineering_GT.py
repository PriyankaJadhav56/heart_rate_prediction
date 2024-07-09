# Databricks notebook source
# MAGIC %pip install /dbfs/FileStore/sdk/dev/MLCoreSDK-0.4.5-py3-none-any.whl --force-reinstall
# MAGIC %pip install sparkmeasure

# COMMAND ----------

from sparkmeasure import StageMetrics
from sparkmeasure import TaskMetrics
taskmetrics = TaskMetrics(spark)
stagemetrics = StageMetrics(spark)

taskmetrics.begin()
stagemetrics.begin()

# COMMAND ----------

try :
    env = dbutils.widgets.get("env")
except :
    env = "dev"
print(f"Input environment : {env}")

# COMMAND ----------

# MAGIC %md <b> User Inputs

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
    print(e)
    with open('../data_config/SolutionConfig.yaml', 'r') as solution_config:
        solution_config = yaml.safe_load(solution_config)  

# COMMAND ----------

try :
    sdk_session_id = dbutils.widgets.get("sdk_session_id")
except :
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

if sdk_session_id.lower() == "none":
    sdk_session_id = solution_config["general_configs"]["sdk_session_id"][env]

tracking_url = solution_config["general_configs"].get("tracking_url", None)
tracking_url = f"https://{tracking_url}" if tracking_url else None

# DE SPECIFIC PARAMETERS
batch_size = int(solution_config["data_engineering_gt"].get("batch_size",500))
input_table_configs = solution_config["data_engineering_gt"]["datalake_configs"]["input_tables"]
output_table_configs = solution_config["data_engineering_gt"]["datalake_configs"]['output_tables']

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

input_table_paths = get_name_space(input_table_configs)
output_table_paths = get_name_space(output_table_configs)

# COMMAND ----------

source_1_df = spark.sql(f"SELECT * FROM {input_table_paths['source_1']}")

# COMMAND ----------

print(f"{input_table_paths['source_1']}")

# COMMAND ----------

mlclient.log(
    operation_type="job_run_add", 
    session_id = sdk_session_id, 
    dbutils = dbutils, 
    request_type = "de", 
    job_config = 
    {
        "table_name" : input_table_configs["source_1"]["table"],
        "table_type" : "Ground_Truth",
        "batch_size" : batch_size
    },
    tracking_env = env,
    tracking_url = tracking_url,
    verbose = True,
    spark = spark
    )

# COMMAND ----------

output_1_df = source_1_df.drop('date','id','timestamp')

# COMMAND ----------

output_1_df = output_1_df.select('index','avg_heart_rate')

# COMMAND ----------

output_1_df.display()

# COMMAND ----------

# DBTITLE 1,Creating hive database if it doesn't exist
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def to_date_(col):
    """
    Checks col row-wise and returns first date format which returns non-null output for the respective column value
    """
    formats=(
             "MM-dd-yyyy", "dd-MM-yyyy",
             "MM/dd/yyyy", "yyyy-MM-dd", 
             "M/d/yyyy", "M/dd/yyyy",
             "MM/dd/yy", "MM.dd.yyyy",
             "dd.MM.yyyy", "yyyy-MM-dd",
             "yyyy-dd-MM"
            )
    return F.coalesce(*[F.to_date(col, f) for f in formats])

# COMMAND ----------

# DBTITLE 1,ADD A MONOTONICALLY INREASING COLUMN - "id"
now = datetime.now()
date = now.strftime("%m-%d-%Y")
output_1_df = output_1_df.withColumn(
    "timestamp",
    F.expr("reflect('java.lang.System', 'currentTimeMillis')").cast("long"),
)
output_1_df = output_1_df.withColumn("date", F.lit(date))
output_1_df = output_1_df.withColumn("date", to_date_(F.col("date")))

# ADD A MONOTONICALLY INREASING COLUMN
if "id" not in output_1_df.columns : 
  window = Window.orderBy(F.monotonically_increasing_id())
  output_1_df = output_1_df.withColumn("id", F.row_number().over(window))

# COMMAND ----------

# DBTITLE 1,Create/Update table in hive database
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

output_1_df.createOrReplaceTempView(table_name)

feature_table_exist = [True for table_data in spark.catalog.listTables(db_name) if table_data.name.lower() == table_name.lower() and not table_data.isTemporary]

if not any(feature_table_exist):
  print(f"CREATING SOURCE TABLE")
  spark.sql(f"CREATE TABLE IF NOT EXISTS {output_path} AS SELECT * FROM {table_name}")
else :
  print(F"UPDATING SOURCE TABLE")
  spark.sql(f"INSERT INTO {output_path} SELECT * FROM {table_name}")

# if catalog_name and catalog_name.lower() != "none": 
output_1_table_path = output_path
# else:
  # output_1_table_path = spark.sql(f"desc {output_path}").filter(F.col("col_name") == "Location").select("data_type").collect()[0][0]

print(f"Features Hive Path : {output_1_table_path}")

# COMMAND ----------

stagemetrics.end()
taskmetrics.end()

stage_Df = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")
task_Df = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

compute_metrics = stagemetrics.aggregate_stagemetrics_DF().select("executorCpuTime", "peakExecutionMemory","memoryBytesSpilled","diskBytesSpilled").collect()[0].asDict()

compute_metrics['executorCpuTime'] = compute_metrics['executorCpuTime']/1000
compute_metrics['peakExecutionMemory'] = float(compute_metrics['peakExecutionMemory']) /(1024*1024)

# COMMAND ----------

# MAGIC %md <b> Use MLCore SDK to register Features and Ground Truth Tables

# COMMAND ----------

from MLCORE_SDK import mlclient

# COMMAND ----------

mlclient.log(operation_type = "register_table",
    sdk_session_id = sdk_session_id,
    dbutils = dbutils,
    spark = spark,
    table_name = output_table_configs["output_1"]["table"],
    num_rows = output_1_df.count(),
    cols = output_1_df.columns,
    column_datatype = output_1_df.dtypes,
    table_schema = output_1_df.schema,
    primary_keys = output_table_configs["output_1"]["primary_keys"],
    table_path = output_1_table_path,
    table_type="unitycatalog" if output_table_configs["output_1"]["catalog_name"] else "internal" ,
    table_sub_type="Ground_Truth",
    tracking_env = env,
    compute_usage_metrics = compute_metrics,
    taskmetrics=taskmetrics,
    stagemetrics=stagemetrics,
    tracking_url = tracking_url,
    # register_in_feature_store=True,
    verbose=True,)
