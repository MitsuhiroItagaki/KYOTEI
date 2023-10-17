-- Databricks notebook source
-- MAGIC %sql
-- MAGIC use catalog main;
-- MAGIC use kyotei_db;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #　累積テーブル削除
-- MAGIC renfuku = sqlContext.sql(\
-- MAGIC "         select '123' as renfuku \
-- MAGIC union all select '124' \
-- MAGIC ").rdd.map(lambda row : row[0]).collect()
-- MAGIC
-- MAGIC #変数取得
-- MAGIC dbutils.widgets.dropdown("RENFUKU", "123", [str(x) for x in renfuku],"3連複")
-- MAGIC RENFUKU=dbutils.widgets.get("RENFUKU")
-- MAGIC print(RENFUKU)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC #　累積テーブル削除
-- MAGIC isDrop = sqlContext.sql(\
-- MAGIC "         select 'YES' as isDrop \
-- MAGIC union all select 'NO' \
-- MAGIC ").rdd.map(lambda row : row[0]).collect()
-- MAGIC
-- MAGIC #変数取得
-- MAGIC dbutils.widgets.dropdown("DROPTABLE", "NO", [str(x) for x in isDrop],"累積テーブル削除")
-- MAGIC DROPTABLE=dbutils.widgets.get("DROPTABLE")
-- MAGIC print(DROPTABLE)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC kekka_table = "kekka_" + RENFUKU
-- MAGIC if DROPTABLE == "YES":
-- MAGIC   sql("drop table if exists " + kekka_table)
-- MAGIC   print("drop:" , "drop table if exists " + kekka_table)
-- MAGIC else:
-- MAGIC   print("keep:" , kekka_table)

-- COMMAND ----------

-- DBTITLE 1, データチェック用
select racedate,count(1)
from training_base2
group by racedate order by racedate;

-- COMMAND ----------

-- DBTITLE 1,データチェック用
select racedate,count(1)
from training_base2
where rentan3 is null
group by racedate order by racedate;

-- COMMAND ----------

-- DBTITLE 1,トレーニング用のベーステーブル
-- MAGIC %sql
-- MAGIC create or replace table training_base3 as
-- MAGIC select 
-- MAGIC distinct 
-- MAGIC  RACEDATE -- 特徴量にはしない
-- MAGIC ,PLACE -- 特徴量にはしない
-- MAGIC ,RACE -- 特徴量にはしない
-- MAGIC
-- MAGIC -- 特徴量にはしない
-- MAGIC --,PLAYERID1
-- MAGIC --,PLAYERID2
-- MAGIC --,PLAYERID3
-- MAGIC --,PLAYERID4
-- MAGIC --,PLAYERID5
-- MAGIC --,PLAYERID6
-- MAGIC
-- MAGIC ,CLASS1
-- MAGIC ,CLASS2
-- MAGIC ,CLASS3
-- MAGIC ,CLASS4
-- MAGIC ,CLASS5
-- MAGIC ,CLASS6
-- MAGIC
-- MAGIC -- 特徴量にはしない
-- MAGIC --,CLUB1
-- MAGIC --,CLUB2
-- MAGIC --,CLUB3
-- MAGIC --,CLUB4
-- MAGIC --,CLUB5
-- MAGIC --,CLUB6
-- MAGIC --,AGE1
-- MAGIC --,AGE2
-- MAGIC --,AGE3
-- MAGIC --,AGE4
-- MAGIC --,AGE5
-- MAGIC --,AGE6
-- MAGIC --,WEIGHT1
-- MAGIC --,WEIGHT2
-- MAGIC --,WEIGHT3
-- MAGIC --,WEIGHT4
-- MAGIC --,WEIGHT5
-- MAGIC --,WEIGHT6
-- MAGIC
-- MAGIC -- 特徴量にはしない
-- MAGIC --,F1
-- MAGIC --,F2
-- MAGIC --,F3
-- MAGIC --,F4
-- MAGIC --,F5
-- MAGIC --,F6
-- MAGIC --,L1
-- MAGIC --,L2
-- MAGIC --,L3
-- MAGIC --,L4
-- MAGIC --,L5
-- MAGIC --,L6
-- MAGIC
-- MAGIC ,WIN1RATE1
-- MAGIC ,WIN1RATE2
-- MAGIC ,WIN1RATE3
-- MAGIC ,WIN1RATE4
-- MAGIC ,WIN1RATE5
-- MAGIC ,WIN1RATE6
-- MAGIC
-- MAGIC ,WIN2RATE1
-- MAGIC ,WIN2RATE2
-- MAGIC ,WIN2RATE3
-- MAGIC ,WIN2RATE4
-- MAGIC ,WIN2RATE5
-- MAGIC ,WIN2RATE6
-- MAGIC
-- MAGIC -- 特徴量にはしない?
-- MAGIC --,LOCALWIN1RATE1
-- MAGIC --,LOCALWIN1RATE2
-- MAGIC --,LOCALWIN1RATE3
-- MAGIC --,LOCALWIN1RATE4
-- MAGIC --,LOCALWIN1RATE5
-- MAGIC --,LOCALWIN1RATE6
-- MAGIC
-- MAGIC -- 特徴量にはしない?
-- MAGIC --,LOCALWIN2RATE1
-- MAGIC --,LOCALWIN2RATE2
-- MAGIC --,LOCALWIN2RATE3
-- MAGIC --,LOCALWIN2RATE4
-- MAGIC --,LOCALWIN2RATE5
-- MAGIC --,LOCALWIN2RATE6
-- MAGIC
-- MAGIC ,MOTORWIN2RATE1
-- MAGIC ,MOTORWIN2RATE2
-- MAGIC ,MOTORWIN2RATE3
-- MAGIC ,MOTORWIN2RATE4
-- MAGIC ,MOTORWIN2RATE5
-- MAGIC ,MOTORWIN2RATE6
-- MAGIC
-- MAGIC -- 特徴量にはしない
-- MAGIC --,MOTORWIN3RATE1
-- MAGIC --,MOTORWIN3RATE2
-- MAGIC --,MOTORWIN3RATE3
-- MAGIC --,MOTORWIN3RATE4
-- MAGIC --,MOTORWIN3RATE5
-- MAGIC --,MOTORWIN3RATE6
-- MAGIC
-- MAGIC -- 特徴量にはしない
-- MAGIC --,BOATWIN2RATE1
-- MAGIC --,BOATWIN2RATE2
-- MAGIC --,BOATWIN2RATE3
-- MAGIC --,BOATWIN2RATE4
-- MAGIC --,BOATWIN2RATE5
-- MAGIC --,BOATWIN2RATE6
-- MAGIC
-- MAGIC --,BOATWIN3RATE1
-- MAGIC --,BOATWIN3RATE2
-- MAGIC --,BOATWIN3RATE3
-- MAGIC --,BOATWIN3RATE4
-- MAGIC --,BOATWIN3RATE5
-- MAGIC --,BOATWIN3RATE6
-- MAGIC
-- MAGIC ,ST_AVG1
-- MAGIC ,ST_AVG2
-- MAGIC ,ST_AVG3
-- MAGIC ,ST_AVG4
-- MAGIC ,ST_AVG5
-- MAGIC ,ST_AVG6
-- MAGIC
-- MAGIC --,COURCE1_RACE_COUNT1
-- MAGIC --,COURCE1_WIN1_RATE1
-- MAGIC --,COURCE1_WIN12_RATE1
-- MAGIC ,COURCE1_WIN123_RATE1
-- MAGIC
-- MAGIC --,COURCE2_RACE_COUNT2
-- MAGIC --,COURCE2_WIN1_RATE2
-- MAGIC --,COURCE2_WIN12_RATE2
-- MAGIC ,COURCE2_WIN123_RATE2
-- MAGIC
-- MAGIC --,COURCE3_RACE_COUNT3
-- MAGIC --,COURCE3_WIN1_RATE3
-- MAGIC --,COURCE3_WIN12_RATE3
-- MAGIC ,COURCE3_WIN123_RATE3
-- MAGIC
-- MAGIC --,COURCE4_RACE_COUNT4
-- MAGIC --,COURCE4_WIN1_RATE4
-- MAGIC --,COURCE4_WIN12_RATE4
-- MAGIC ,COURCE4_WIN123_RATE4
-- MAGIC
-- MAGIC --,COURCE5_RACE_COUNT5
-- MAGIC --,COURCE5_WIN1_RATE5
-- MAGIC --,COURCE5_WIN12_RATE5
-- MAGIC ,COURCE5_WIN123_RATE5
-- MAGIC
-- MAGIC --,COURCE6_RACE_COUNT6
-- MAGIC --,COURCE6_WIN1_RATE6
-- MAGIC --,COURCE6_WIN12_RATE6
-- MAGIC ,COURCE6_WIN123_RATE6
-- MAGIC
-- MAGIC --,COURCE1_LOCAL_RACE_COUNT1
-- MAGIC --,COURCE1_LOCAL_WIN1_RATE1
-- MAGIC --,COURCE1_LOCAL_WIN12_RATE1
-- MAGIC ,COURCE1_LOCAL_WIN123_RATE1
-- MAGIC
-- MAGIC --,COURCE2_LOCAL_RACE_COUNT2
-- MAGIC --,COURCE2_LOCAL_WIN1_RATE2
-- MAGIC --,COURCE2_LOCAL_WIN12_RATE2
-- MAGIC ,COURCE2_LOCAL_WIN123_RATE2
-- MAGIC
-- MAGIC --,COURCE3_LOCAL_RACE_COUNT3
-- MAGIC --,COURCE3_LOCAL_WIN1_RATE3
-- MAGIC --,COURCE3_LOCAL_WIN12_RATE3
-- MAGIC ,COURCE3_LOCAL_WIN123_RATE3
-- MAGIC
-- MAGIC --,COURCE4_LOCAL_RACE_COUNT4
-- MAGIC --,COURCE4_LOCAL_WIN1_RATE4
-- MAGIC --,COURCE4_LOCAL_WIN12_RATE4
-- MAGIC ,COURCE4_LOCAL_WIN123_RATE4
-- MAGIC
-- MAGIC --,COURCE5_LOCAL_RACE_COUNT5
-- MAGIC --,COURCE5_LOCAL_WIN1_RATE5
-- MAGIC --,COURCE5_LOCAL_WIN12_RATE5
-- MAGIC ,COURCE5_LOCAL_WIN123_RATE5
-- MAGIC
-- MAGIC --,COURCE6vRACE_COUNT6
-- MAGIC --,COURCE6_LOCAL_WIN1_RATE6
-- MAGIC --,COURCE6_LOCAL_WIN12_RATE6
-- MAGIC ,COURCE6_LOCAL_WIN123_RATE6
-- MAGIC
-- MAGIC --,TAN
-- MAGIC --,TANK
-- MAGIC --,WIDE1
-- MAGIC --,WIDE1K
-- MAGIC --,WIDE2
-- MAGIC --,WIDE2K
-- MAGIC --,WIDE3
-- MAGIC --,WIDE3K
-- MAGIC
-- MAGIC ,RENTAN2
-- MAGIC ,RENTAN2K
-- MAGIC ,RENFUKU2
-- MAGIC ,RENFUKU2K
-- MAGIC ,RENTAN3
-- MAGIC ,RENTAN3K
-- MAGIC ,RENFUKU3K
-- MAGIC ,RENFUKU3
-- MAGIC from training_base2
-- MAGIC ;

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC select racedate,count(1) from  training_base3 group by 1 order by 1
-- MAGIC ;

-- COMMAND ----------

-- MAGIC %md ## 一年前から予測対象の明日までのデータセットを取得

-- COMMAND ----------

-- MAGIC %python
-- MAGIC START_DATE = sql("SELECT date_format(date_sub(current_date(), 12 * 30), 'yyyyMMdd') as start").toPandas().to_string(index=False, header=False)
-- MAGIC print(START_DATE)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC END_DATE = sql("SELECT date_format(DATE_ADD(current_date(),1), 'yyyyMMdd') as tomorrow_date").toPandas().to_string(index=False, header=False)
-- MAGIC print(END_DATE)

-- COMMAND ----------

-- MAGIC %python
-- MAGIC sqltext=f"""
-- MAGIC create or replace table training_123f
-- MAGIC as 
-- MAGIC select 
-- MAGIC *,
-- MAGIC case when RENFUKU3 = '1=2=3'  then 1 else 0 end as kekka 
-- MAGIC from training_base3 
-- MAGIC --where racedate between '20220708' and '20230702' -- 明示的に指定する場合
-- MAGIC where racedate between '{START_DATE}' and '{END_DATE}' -- トレーニングのための期間(12month)　
-- MAGIC and   rentan3 is not null -- NULLデータは除外、且つ当日データは含まれない
-- MAGIC """
-- MAGIC print(sqltext)
-- MAGIC
-- MAGIC # 実行
-- MAGIC sql(sqltext)

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC select racedate,count(1) from  training_123f group by 1 order by 1
-- MAGIC ;

-- COMMAND ----------

-- MAGIC %md ##  予測データ作成

-- COMMAND ----------

-- MAGIC %python
-- MAGIC sqltext=f"""
-- MAGIC create or replace table predict_123f
-- MAGIC as 
-- MAGIC select 
-- MAGIC *,
-- MAGIC case when RENFUKU3 = '1=2=3'  then 1 else 0 end as kekka 
-- MAGIC from training_base3 
-- MAGIC --where racedate between '20230717' and '20230717' -- 明示的に指定する場合
-- MAGIC --where racedate = '{END_DATE}' 
-- MAGIC where racedate = ( select max(racedate) from training_base3) -- 取得済みデータの最終日のみ指定する場合
-- MAGIC """
-- MAGIC print(sqltext)
-- MAGIC
-- MAGIC # 実行
-- MAGIC sql(sqltext)

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC select racedate,count(1) from  predict_123f group by 1 order by 1
-- MAGIC ;

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC select * from  predict_123f;

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC select min(racedate), max(racedate),count(1) from  training_base3;

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC select min(racedate), max(racedate),count(1) from  training_123f;

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC select min(racedate), max(racedate),count(1) from  predict_123f;

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC select * from  predict_123f;