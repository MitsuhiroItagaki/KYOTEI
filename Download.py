# Databricks notebook source
# MAGIC %sql
# MAGIC use catalog main;
# MAGIC use kyotei_db;

# COMMAND ----------

# MAGIC %sql
# MAGIC show tables;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bangumi order by RACEDATE desc ,place,int(RACE);

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from result order by RACEDATE desc,place,int(RACE);
