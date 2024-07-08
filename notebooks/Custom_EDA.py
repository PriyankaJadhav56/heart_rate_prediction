# Databricks notebook source
# MAGIC %pip install /dbfs/FileStore/sdk/Vamsi/tigerml.core-0.4.4-py3-none-any.whl --force-reinstall
# MAGIC %pip install /dbfs/FileStore/sdk/Vamsi/tigerml.eda-0.4.4-py3-none-any.whl
# MAGIC # %pip install /dbfs/FileStore/sdk/Vamsi/tigerml.model_eval-0.4.4-py3-none-any.whl --force-reinstall
# MAGIC
# MAGIC %pip install numpy==1.22

# COMMAND ----------

from MLCORE_SDK import mlclient
from tigerml.eda import EDAReport

# COMMAND ----------

input_table_path = dbutils.widgets.get("input_table_path")

# COMMAND ----------

pd_df = spark.read.load(input_table_path).toPandas()
pd_df.drop(columns = ["id","date","timestamp"],inplace = True)
pd_df.display()

# COMMAND ----------

tigermleda = EDAReport(pd_df)

# COMMAND ----------

mlclient.log(operation_type = "register_plots",
    dbutils = dbutils, 
    figure_to_save=tigermleda, 
    plot_name='Tiger_ML_EDA',
    lib="tigerml",
    folder_name = "custom_reports",
    request_type="push_plot")
