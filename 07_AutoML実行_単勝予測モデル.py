# Databricks notebook source
# MAGIC %sql
# MAGIC use catalog main;
# MAGIC use kyotei_db;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select Place,count(1) from model_training group by 1 order by 1

# COMMAND ----------

# MAGIC %md ## トレーニングデータを取得

# COMMAND ----------

#
train_df = sql("select * from model_training order by 1")

# COMMAND ----------

# MAGIC %md ## トレーニング実行

# COMMAND ----------

from databricks import automl

# COMMAND ----------

from databricks import automl

#現在時刻を取得
CURRENT_TIME=sql("SELECT replace(from_utc_timestamp(current_timestamp(), 'Asia/Tokyo'),' ','-')").toPandas().to_string(index=False, header=False)

# AUTOML実行
summary = automl.classify(train_df,time_col="RACEDATE",target_col="TAN", primary_metric="roc_auc", exclude_frameworks=['sklearn', 'xgboost'], exclude_cols=['RACEDATE','RACE','TAN','TANK','FUKU1','FUKU1K','FUKU2','FUKU2K','WIDE1','WIDE1K','WIDE2','WIDE2K','WIDE3','WIDE3K','RENTAN2', 'RENTAN2K', 'RENFUKU2','RENFUKU2K','RENFUKU3','RENFUKU3K','RENTAN3', 'RENTAN3K'], timeout_minutes=300, experiment_name=f"kyotei_tan_{CURRENT_TIME}")
