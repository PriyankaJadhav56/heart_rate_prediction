# Databricks notebook source
model_name = dbutils.widgets.get("model_name")
train_output_path = dbutils.widgets.get("model_data_path")
date_column = dbutils.widgets.get("date_column")
feature_columns = dbutils.widgets.get("feature_columns").split(",")
target_columns = dbutils.widgets.get("target_columns")
media_artifacts_path = dbutils.widgets.get("media_artifacts_path")

# COMMAND ----------

if date_column in feature_columns:
    feature_columns.remove(date_column)

# Now, 'date_column' has been removed from feature_columns
print(feature_columns)

# COMMAND ----------

import sklearn
import mlflow
from sklearn.linear_model import LinearRegression
# from tigerml.model_eval import RegressionReport,ClassificationReport
import warnings
from MLCORE_SDK import mlclient
warnings.filterwarnings("ignore")
#logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
model_versions = client.get_latest_versions(model_name)
model_version = model_versions[0].version

# COMMAND ----------

train_output_df = spark.read.load(train_output_path).toPandas()
train_output_df.display()

# COMMAND ----------

loaded_model = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{model_version}")
model = loaded_model
type(model)

# COMMAND ----------

# MAGIC %md ## Model Evaluation Template Logic
# MAGIC

# COMMAND ----------

# MAGIC %md ### Option 1 - with model
# MAGIC

# COMMAND ----------

train_df = train_output_df[train_output_df['dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE'] == "train"]
test_df = train_output_df[train_output_df['dataset_type_71E4E76EB8C12230B6F51EA2214BD5FE'] == "test"]

# COMMAND ----------

import pandas as pd
import plotly.express as px

# Create a line plot using Plotly
fig_train = px.line(train_df, x=date_column, y=[target_columns, 'prediction'])

fig_test = px.line(test_df, x=date_column, y=[target_columns, 'prediction'])

fig_train.update_traces(mode='markers+lines', marker=dict(size=5))
fig_test.update_traces(mode='markers+lines', marker=dict(size=5))

fig_train.update_layout(
    title='Train Actual & Train Predicted',
    xaxis_title='Timestamp',
    yaxis_title='Sales Volume',
    legend_title='Legend',
    xaxis=dict(tickangle=-45),
    width=1000,
    height=500,
)

fig_test.update_layout(
    title='Test Actual & Test Predicted',
    xaxis_title='Timestamp',
    yaxis_title='Sales Volume',
    legend_title='Legend',
    xaxis=dict(tickangle=-45),
    width=1000,
    height=500,
)

# Show the plot
fig_train.show()
fig_test.show()

# COMMAND ----------

mlclient.log(operation_type = "register_plots",
    dbutils = dbutils, 
    figure_to_save = fig_train, 
    plot_name='Actual_vs_Forecast',
    lib="plotly",
    ext = "html",
    include_shap= True,
    folder_name = "Model_Evaluation",
    request_type="push_plot")

# COMMAND ----------

mlclient.log(operation_type = "register_plots",
    dbutils = dbutils, 
    figure_to_save=fig_test, 
    plot_name='Actual_vs_Forecast_',
    lib="plotly",
    ext = "html",
    include_shap= True,
    folder_name = "Model_Evaluation",
    request_type="push_plot")

# COMMAND ----------

train_residuals = train_df[target_columns] - train_df['prediction']
test_residuals = test_df[target_columns] - test_df['prediction']

# Create a DataFrame for the residuals
train_residual_df = pd.DataFrame({'Date': train_df[date_column], 'Residuals': train_residuals})
test_residual_df = pd.DataFrame({'Date': test_df[date_column], 'Residuals': test_residuals})

# Create a residual plot using Plotly
fig_train_residual = px.scatter(train_residual_df, x='Date', y='Residuals', title='Train Residual Plot')
fig_train_residual.add_shape(type='line', x0=min(train_residual_df['Date']), x1=max(train_residual_df['Date']), y0=0, y1=0, line=dict(color='red', dash='dash'))

# Create a residual plot using Plotly
fig_test_residual = px.scatter(test_residual_df, x='Date', y='Residuals', title='Test Residual Plot')
fig_test_residual.add_shape(type='line', x0=min(test_residual_df['Date']), x1=max(test_residual_df['Date']), y0=0, y1=0, line=dict(color='red', dash='dash'))

# Update axis labels
fig_train_residual.update_xaxes(title_text='Date')
fig_train_residual.update_yaxes(title_text='Residuals')

# Update axis labels
fig_test_residual.update_xaxes(title_text='Date')
fig_test_residual.update_yaxes(title_text='Residuals')

# Adjust the figure size
fig_train_residual.update_layout(width=800, height=600)
fig_test_residual.update_layout(width=800, height=600)

# Show the plot
fig_train_residual.show()
fig_test_residual.show()

# COMMAND ----------

mlclient.log(operation_type = "register_plots",
    dbutils = dbutils, 
    figure_to_save=fig_train_residual, 
    plot_name='Residual_Plot_for_Train_Data',
    lib="plotly",
    ext = "html",
    include_shap= True,
    folder_name = "Model_Evaluation",
    request_type="push_plot")

# COMMAND ----------

mlclient.log(operation_type = "register_plots",
    dbutils = dbutils, 
    figure_to_save=fig_test_residual, 
    plot_name='Residual_Plot_for_Test_Data',
    lib="plotly",
    ext = "html",
    include_shap= True,
    folder_name = "Model_Evaluation",
    request_type="push_plot")

# COMMAND ----------

from prophet.utilities import regressor_coefficients
regressor_coef = regressor_coefficients(model)

coefficients = regressor_coef["coef"]

feature_importance_df = pd.DataFrame({'Feature': feature_columns , 'Coefficient': coefficients})

# Sort the DataFrame by coefficient magnitude in descending order
feature_importance_df = feature_importance_df.reindex(feature_importance_df['Coefficient'].abs().sort_values(ascending=False).index)

# Create a bar chart of feature importances using Plotly Express with swapped axes
fig1 = px.bar(feature_importance_df, x='Coefficient', y='Feature', title='Feature Importance  for Linear Regression', orientation='h')
fig1.update_traces(marker_color='darkblue')
fig1.update_xaxes(title='Coefficient Value')
fig1.update_yaxes(categoryorder='total ascending')  # Sort features by coefficient magnitude

# Show the plot
fig1.show()

# COMMAND ----------

# Using helper method save_fig to save the plot
mlclient.log(operation_type = "register_plots",
    dbutils = dbutils, 
    figure_to_save=fig1, 
    plot_name="Feature_Importance",
    lib="plotly",
    ext = "html",
    folder_name = "Model_Evaluation",
    request_type="push_plot")
