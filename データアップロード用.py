# Databricks notebook source
# MAGIC %sql
# MAGIC use catalog main;
# MAGIC use kyotei_db;

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables;

# COMMAND ----------

# MAGIC %sh
# MAGIC ls -l /Workspace/Repos/mitsuhiro.itagaki@databricks.com/KYOTEI/bangumi20220101_20230903.csv.zip

# COMMAND ----------

# MAGIC %sh
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bangumi order by RACEDATE desc ,place,int(RACE);

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from result order by RACEDATE desc,place,int(RACE);
