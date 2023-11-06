# Databricks notebook source
# MAGIC %sql
# MAGIC use catalog main;
# MAGIC use kyotei_db;

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables;

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -l /Workspace/Repos/mitsuhiro.itagaki@databricks.com/KYOTEI/Backup_files/bangumi20220101_20230903.csv.zip

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -l /Workspace/Repos/mitsuhiro.itagaki@databricks.com/KYOTEI/Backup_files/result20220101_20230903.csv.zip

# COMMAND ----------

import pandas as pd

# COMMAND ----------

bangumi_df = spark.createDataFrame(pd.read_csv('/Workspace/Repos/mitsuhiro.itagaki@databricks.com/KYOTEI/Backup_files/bangumi20220101_20230903.csv.zip'))
bangumi_df.createOrReplaceTempView('bangumi_tmp')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bangumi_tmp

# COMMAND ----------

result_df = spark.createDataFrame(pd.read_csv('/Workspace/Repos/mitsuhiro.itagaki@databricks.com/KYOTEI/Backup_files/result20220101_20230903.csv.zip'))
result_df.createOrReplaceTempView('result_tmp')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from result_tmp

# COMMAND ----------

# MAGIC %sql
# MAGIC insert into bangumi select * from bangumi_tmp;

# COMMAND ----------

# MAGIC %sql
# MAGIC insert into result select * from result_tmp;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bangumi order by RACEDATE desc ,place,int(RACE);

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from result order by RACEDATE desc,place,int(RACE);
