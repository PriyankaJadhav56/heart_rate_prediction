# Databricks notebook source
# MAGIC %md
# MAGIC #### Install MLCORE SDK

# COMMAND ----------

#%pip install /dbfs/FileStore/sdk/dev/MLCoreSDK-0.0.1-py3-none-any.whl --force-reinstall

# COMMAND ----------

# %pip install databricks-feature-store

# COMMAND ----------

# MAGIC %md
# MAGIC #### Install Deep Checks, MLFlow, Pandas and Numpy to specific version

# COMMAND ----------

# MAGIC %pip install deepchecks

# COMMAND ----------

# %pip install mlflow

# COMMAND ----------

# MAGIC %pip install pandas==1.0.5

# COMMAND ----------

# MAGIC %pip install numpy==1.19.1

# COMMAND ----------

# MAGIC %pip install matplotlib==3.3.2

# COMMAND ----------

# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Import libraries

# COMMAND ----------

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, model_evaluation
from io import StringIO
from pyspark.sql import functions as F
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read inputs parameters provided to the notebook.

# COMMAND ----------

model_name = dbutils.widgets.get("model_name")
model_data_path = dbutils.widgets.get("model_data_path")
feature_columns = dbutils.widgets.get("feature_columns").split(",")
target_columns = dbutils.widgets.get("target_columns").split(",")
media_artifacts_path = dbutils.widgets.get("media_artifacts_path")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Read the train data

# COMMAND ----------

data_to_check = spark.read.load(model_data_path)

# COMMAND ----------

def take_random_rows(df, n=20000):
    """
    Samples n rows randomly from the input dataframe
    """
    fraction_value = n / df.count()
    if fraction_value > 1:
        return df
    else:
        return df.sample(withReplacement=False, fraction=fraction_value, seed=2023) 

# COMMAND ----------

data_to_check = take_random_rows(data_to_check)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Fetch the Train and Test data and select the categorical columns

# COMMAND ----------

trainDF = data_to_check.filter(F.col("dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE") == "train").select(feature_columns + target_columns)
testDF = data_to_check.filter(F.col("dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE") == "test").select(feature_columns + target_columns)

# COMMAND ----------

def detect_categorical_cols(df, threshold=5):
    """
    Get the Categorical columns with greater than threshold percentage of unique values.

    This function returns the Categorical columns with the unique values in the column
    greater than the threshold percentage.

    Parameters
    ----------
    df: pyspark.sql.DataFrame
    threshold : int , default = 5
        threshold value in percentage

    Returns
    -------
    report_data : dict
        dictionary containing the Numeric column data.

    """
    df = df.toPandas()
    no_of_rows = df.shape[0]
    possible_cat_cols = (
        df.convert_dtypes()
        .select_dtypes(exclude=[np.datetime64, "float", "float64"])
        .columns.values.tolist()
    )
    temp_series = df[possible_cat_cols].apply(
        lambda col: (len(col.unique()) / no_of_rows) * 100 > threshold
    )
    cat_cols = temp_series[temp_series == False].index.tolist()
    return cat_cols

# COMMAND ----------

categorial_columns = detect_categorical_cols(trainDF.select(feature_columns))

# COMMAND ----------

pd_train = trainDF.toPandas()
pd_test = testDF.toPandas()

ds_train = Dataset(pd_train, label=target_columns[0], cat_features=categorial_columns)
ds_test = Dataset(pd_test, label=target_columns[0], cat_features=categorial_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Perform Data Integrity Test on Train and Test data.

# COMMAND ----------

train_res = data_integrity().run(ds_train)
test_res = data_integrity().run(ds_test)

# COMMAND ----------

from MLCORE_SDK import mlclient

# COMMAND ----------

# DBTITLE 1,Show the report for train data
# Using helper method save_fig to save the plot

mlclient.log(operation_type = "register_plots",
    dbutils = dbutils, 
    figure_to_save=train_res, 
    plot_name="Train_DeepCheck_Report",
    lib='deepchecks',
    folder_name = "Test_Validation",
    request_type="push_plot")

# COMMAND ----------

# DBTITLE 1,Show the report for test data
# Using helper method save_fig to save the plot

mlclient.log(operation_type = "register_plots",
    dbutils = dbutils, 
    figure_to_save=test_res, 
    plot_name="Test_DeepCheck_Report",
    lib='deepchecks',
    folder_name = "Test_Validation",
    request_type="push_plot")
