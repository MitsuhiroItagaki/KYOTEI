# Databricks notebook source
# MAGIC %sql
# MAGIC use catalog main;
# MAGIC use kyotei_db;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select Place,count(1) from  training group by 1 order by 1

# COMMAND ----------

# MAGIC %md ## 全データ

# COMMAND ----------

#トレーニングデータを取得
train_df = sql("select * from  training order by 1")

# COMMAND ----------

from databricks import automl

#現在時刻を取得
CURRENT_TIME=sql("SELECT replace(from_utc_timestamp(current_timestamp(), 'Asia/Tokyo'),' ','-')").toPandas().to_string(index=False, header=False)

# コマンドラインでのAutoMLを実行
summary = automl.classify(train_df, target_col="TAN", primary_metric="roc_auc", exclude_frameworks=['sklearn', 'xgboost'], exclude_cols=['PLACE', 'TANK', 'RENTAN2', 'RENTAN2K', 'RENTAN3', 'RENTAN3K'], timeout_minutes=10, experiment_name=f"kyotei_{CURRENT_TIME}")

# COMMAND ----------

# MAGIC %md ## 三国

# COMMAND ----------

#トレーニングデータを取得
train_df = sql("select * from  training where place = '三国' order by 1")

# COMMAND ----------

from databricks import automl

#現在時刻を取得
CURRENT_TIME=sql("SELECT replace(from_utc_timestamp(current_timestamp(), 'Asia/Tokyo'),' ','-')").toPandas().to_string(index=False, header=False)

# コマンドラインでのAutoMLを実行
summary = automl.classify(train_df, target_col="TAN", primary_metric="roc_auc", exclude_frameworks=['sklearn', 'xgboost'], exclude_cols=['PLACE', 'TANK', 'RENTAN2', 'RENTAN2K', 'RENTAN3', 'RENTAN3K'], timeout_minutes=10, experiment_name=f"kyotei_{CURRENT_TIME}")

# COMMAND ----------

# MAGIC %md ## 下関

# COMMAND ----------

#トレーニングデータを取得
train_df = sql("select * from  training where place = '下関' order by 1")

# COMMAND ----------

from databricks import automl

#現在時刻を取得
CURRENT_TIME=sql("SELECT replace(from_utc_timestamp(current_timestamp(), 'Asia/Tokyo'),' ','-')").toPandas().to_string(index=False, header=False)

# コマンドラインでのAutoMLを実行
summary = automl.classify(train_df, target_col="TAN", primary_metric="roc_auc", exclude_frameworks=['sklearn', 'xgboost'], exclude_cols=['PLACE', 'TANK', 'RENTAN2', 'RENTAN2K', 'RENTAN3', 'RENTAN3K'], timeout_minutes=10, experiment_name=f"kyotei_{CURRENT_TIME}")
