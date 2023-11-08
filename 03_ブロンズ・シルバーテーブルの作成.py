# Databricks notebook source
# MAGIC %md
# MAGIC #ブロンズ・シルバーテーブルの作成
# MAGIC
# MAGIC - 特徴量を計算し、書き出す。
# MAGIC - これらの特徴量を用いて、運賃を予測するモデルを学習する。
# MAGIC - Feature Storeに保存されている既存の特徴量を用いて、新しいデータのバッチでモデルを評価する。
# MAGIC
# MAGIC ## Requirements
# MAGIC - 機械学習のためのDatabricksランタイム (ML) 
# MAGIC   - Alternatively, you may use Databricks Runtime by running `%pip install databricks-feature-store` at the start of this notebook.
# MAGIC
# MAGIC **Note:** このノートブックは、Feature Storeクライアントv0.3.6以降で動作するように記述されています。v0.3.5以下をお使いの場合は、Cmd 19を削除またはコメントアウトし、Cmd 20をアンコメントしてください。

# COMMAND ----------

# MAGIC %md ## Compute features

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog main;
# MAGIC use kyotei_db;

# COMMAND ----------

# DBTITLE 1,結果表
# MAGIC %sql
# MAGIC select * from result 
# MAGIC --where place is null
# MAGIC order by racedate desc ,place,cast(race as int);

# COMMAND ----------

# MAGIC %sql
# MAGIC select RACEDATE,count(1) from result 
# MAGIC group by RACEDATE
# MAGIC order by RACEDATE

# COMMAND ----------

# DBTITLE 1,番組表
# MAGIC %sql
# MAGIC select * from bangumi 
# MAGIC --where place is null
# MAGIC order by racedate desc ,place,cast(race as int);

# COMMAND ----------

# MAGIC %sql
# MAGIC select RACEDATE,count(1) from bangumi 
# MAGIC group by RACEDATE
# MAGIC order by RACEDATE

# COMMAND ----------

# DBTITLE 1,ブロンズテーブル：番組表と結果を結合
# MAGIC %sql
# MAGIC create or replace table TRAINING_BRONZE as
# MAGIC select 
# MAGIC T1.RACEDATE        ,
# MAGIC case 
# MAGIC         when T1.place = "01" or T1.place = "1" then "桐生"
# MAGIC         when T1.place = "02" or T1.place = "2" then "戸田"
# MAGIC         when T1.place = "03" or T1.place = "3" then "江戸川"
# MAGIC         when T1.place = "04" or T1.place = "4" then "平和島"
# MAGIC         when T1.place = "05" or T1.place = "5" then "多摩川"
# MAGIC         when T1.place = "06" or T1.place = "6" then "浜名湖"
# MAGIC         when T1.place = "07" or T1.place = "7" then "蒲郡"
# MAGIC         when T1.place = "08" or T1.place = "8" then "常滑"
# MAGIC         when T1.place = "09" or T1.place = "9" then "津"
# MAGIC         when T1.place = "10" then "三国"
# MAGIC         when T1.place = "11" then "琵琶湖"
# MAGIC         when T1.place = "12" then "住之江"
# MAGIC         when T1.place = "13" then "尼崎"
# MAGIC         when T1.place = "14" then "鳴門"
# MAGIC         when T1.place = "15" then "丸亀"
# MAGIC         when T1.place = "16" then "児島"
# MAGIC         when T1.place = "17" then "宮島"
# MAGIC         when T1.place = "18" then "徳山"
# MAGIC         when T1.place = "19" then "下関"
# MAGIC         when T1.place = "20" then "若松"
# MAGIC         when T1.place = "21" then "芦屋"
# MAGIC         when T1.place = "22" then "福岡"
# MAGIC         when T1.place = "23" then "唐津"
# MAGIC         when T1.place = "24" then "大村"
# MAGIC       end as PLACE,
# MAGIC
# MAGIC T1.RACE    ,
# MAGIC
# MAGIC max(T1.PLAYERID1 ) as PLAYERID1 ,
# MAGIC max(T1.PLAYERID2 ) as PLAYERID2 ,
# MAGIC max(T1.PLAYERID3 ) as PLAYERID3 ,
# MAGIC max(T1.PLAYERID4 ) as PLAYERID4 ,
# MAGIC max(T1.PLAYERID5 ) as PLAYERID5 ,
# MAGIC max(T1.PLAYERID6 ) as PLAYERID6 ,
# MAGIC
# MAGIC max(T1.CLASS1    ) as CLASS1,
# MAGIC max(T1.CLASS2    ) as CLASS2,
# MAGIC max(T1.CLASS3    ) as CLASS3,
# MAGIC max(T1.CLASS4    ) as CLASS4,
# MAGIC max(T1.CLASS5    ) as CLASS5,
# MAGIC max(T1.CLASS6    ) as CLASS6,
# MAGIC
# MAGIC max(CLUB1        ) as CLUB1,
# MAGIC max(CLUB2        ) as CLUB2,
# MAGIC max(CLUB3        ) as CLUB3,
# MAGIC max(CLUB4        ) as CLUB4,
# MAGIC max(CLUB5        ) as CLUB5,
# MAGIC max(CLUB6        ) as CLUB6,
# MAGIC
# MAGIC max(CAST(AGE1 as double)) as AGE1  ,
# MAGIC max(CAST(AGE2 as double)) as AGE2  ,
# MAGIC max(CAST(AGE3 as double)) as AGE3  ,
# MAGIC max(CAST(AGE4 as double)) as AGE4  ,
# MAGIC max(CAST(AGE5 as double)) as AGE5  ,
# MAGIC max(CAST(AGE6 as double)) as AGE6  ,
# MAGIC
# MAGIC max(CAST(WEIGHT1 as double)) as WEIGHT1 ,
# MAGIC max(CAST(WEIGHT2 as double)) as WEIGHT2 ,
# MAGIC max(CAST(WEIGHT3 as double)) as WEIGHT3 ,
# MAGIC max(CAST(WEIGHT4 as double)) as WEIGHT4 ,
# MAGIC max(CAST(WEIGHT5 as double)) as WEIGHT5 ,
# MAGIC max(CAST(WEIGHT6 as double)) as WEIGHT6 ,
# MAGIC
# MAGIC max(CAST(F1 as double)) as F1  ,
# MAGIC max(CAST(F2 as double)) as F2  ,
# MAGIC max(CAST(F3 as double)) as F3  ,
# MAGIC max(CAST(F4 as double)) as F4  ,
# MAGIC max(CAST(F5 as double)) as F5  ,
# MAGIC max(CAST(F6 as double)) as F6  ,
# MAGIC
# MAGIC max(CAST(L1 as double)) as L1  ,
# MAGIC max(CAST(L2 as double)) as L2  ,
# MAGIC max(CAST(L3 as double)) as L3  ,
# MAGIC max(CAST(L4 as double)) as L4  ,
# MAGIC max(CAST(L5 as double)) as L5  ,
# MAGIC max(CAST(L6 as double)) as L6  ,
# MAGIC
# MAGIC max(CAST(WIN1RATE1 as double)) as WIN1RATE1      ,
# MAGIC max(CAST(WIN1RATE2 as double)) as WIN1RATE2      ,
# MAGIC max(CAST(WIN1RATE3 as double)) as WIN1RATE3      ,
# MAGIC max(CAST(WIN1RATE4 as double)) as WIN1RATE4      ,
# MAGIC max(CAST(WIN1RATE5 as double)) as WIN1RATE5      ,
# MAGIC max(CAST(WIN1RATE6 as double)) as WIN1RATE6      ,
# MAGIC
# MAGIC max(CAST(WIN2RATE1 as double)) as WIN2RATE1      ,
# MAGIC max(CAST(WIN2RATE2 as double)) as WIN2RATE2      ,
# MAGIC max(CAST(WIN2RATE3 as double)) as WIN2RATE3      ,
# MAGIC max(CAST(WIN2RATE4 as double)) as WIN2RATE4      ,
# MAGIC max(CAST(WIN2RATE5 as double)) as WIN2RATE5      ,
# MAGIC max(CAST(WIN2RATE6 as double)) as WIN2RATE6      ,
# MAGIC
# MAGIC max(CAST(LOCALWIN1RATE1 as double)) as LOCALWIN1RATE1  ,
# MAGIC max(CAST(LOCALWIN1RATE2 as double)) as LOCALWIN1RATE2  ,
# MAGIC max(CAST(LOCALWIN1RATE3 as double)) as LOCALWIN1RATE3  ,
# MAGIC max(CAST(LOCALWIN1RATE4 as double)) as LOCALWIN1RATE4  ,
# MAGIC max(CAST(LOCALWIN1RATE5 as double)) as LOCALWIN1RATE5  ,
# MAGIC max(CAST(LOCALWIN1RATE6 as double)) as LOCALWIN1RATE6  ,
# MAGIC
# MAGIC max(CAST(LOCALWIN2RATE1 as double)) as LOCALWIN2RATE1 ,
# MAGIC max(CAST(LOCALWIN2RATE2 as double)) as LOCALWIN2RATE2 ,
# MAGIC max(CAST(LOCALWIN2RATE3 as double)) as LOCALWIN2RATE3 ,
# MAGIC max(CAST(LOCALWIN2RATE4 as double)) as LOCALWIN2RATE4 ,
# MAGIC max(CAST(LOCALWIN2RATE5 as double)) as LOCALWIN2RATE5 ,
# MAGIC max(CAST(LOCALWIN2RATE6 as double)) as LOCALWIN2RATE6 ,
# MAGIC
# MAGIC max(CAST(MOTORWIN2RATE1 as double)) as MOTORWIN2RATE1 ,
# MAGIC max(CAST(MOTORWIN2RATE2 as double)) as MOTORWIN2RATE2 ,
# MAGIC max(CAST(MOTORWIN2RATE3 as double)) as MOTORWIN2RATE3 ,
# MAGIC max(CAST(MOTORWIN2RATE4 as double)) as MOTORWIN2RATE4 ,
# MAGIC max(CAST(MOTORWIN2RATE5 as double)) as MOTORWIN2RATE5 ,
# MAGIC max(CAST(MOTORWIN2RATE6 as double)) as MOTORWIN2RATE6 ,
# MAGIC
# MAGIC max(CAST(MOTORWIN3RATE1 as double)) as MOTORWIN3RATE1  ,
# MAGIC max(CAST(MOTORWIN3RATE2 as double)) as MOTORWIN3RATE2  ,
# MAGIC max(CAST(MOTORWIN3RATE3 as double)) as MOTORWIN3RATE3  ,
# MAGIC max(CAST(MOTORWIN3RATE4 as double)) as MOTORWIN3RATE4  ,
# MAGIC max(CAST(MOTORWIN3RATE5 as double)) as MOTORWIN3RATE5  ,
# MAGIC max(CAST(MOTORWIN3RATE6 as double)) as MOTORWIN3RATE6  ,
# MAGIC
# MAGIC max(CAST(BOATWIN2RATE1  as double)) as BOATWIN2RATE1,
# MAGIC max(CAST(BOATWIN2RATE2  as double)) as BOATWIN2RATE2,
# MAGIC max(CAST(BOATWIN2RATE3  as double)) as BOATWIN2RATE3,
# MAGIC max(CAST(BOATWIN2RATE4  as double)) as BOATWIN2RATE4,
# MAGIC max(CAST(BOATWIN2RATE5  as double)) as BOATWIN2RATE5,
# MAGIC max(CAST(BOATWIN2RATE6  as double)) as BOATWIN2RATE6,
# MAGIC
# MAGIC max(CAST(BOATWIN3RATE1  as double)) as BOATWIN3RATE1,
# MAGIC max(CAST(BOATWIN3RATE2  as double)) as BOATWIN3RATE2,
# MAGIC max(CAST(BOATWIN3RATE3  as double)) as BOATWIN3RATE3,
# MAGIC max(CAST(BOATWIN3RATE4  as double)) as BOATWIN3RATE4,
# MAGIC max(CAST(BOATWIN3RATE5  as double)) as BOATWIN3RATE5,
# MAGIC max(CAST(BOATWIN3RATE6  as double)) as BOATWIN3RATE6,
# MAGIC
# MAGIC max(CAST(ST_AVG1 as double)) as ST_AVG1 ,
# MAGIC max(CAST(ST_AVG2 as double)) as ST_AVG2 ,
# MAGIC max(CAST(ST_AVG3 as double)) as ST_AVG3 ,
# MAGIC max(CAST(ST_AVG4 as double)) as ST_AVG4 ,
# MAGIC max(CAST(ST_AVG5 as double)) as ST_AVG5 ,
# MAGIC max(CAST(ST_AVG6 as double)) as ST_AVG6 ,
# MAGIC
# MAGIC max(T2.TAN) as TAN,
# MAGIC max(T2.TANK) as TANK,
# MAGIC max(T2.RENTAN2) as RENTAN2,
# MAGIC max(T2.RENTAN2K) as RENTAN2K,
# MAGIC max(T2.RENFUKU2) as RENFUKU2,
# MAGIC max(T2.RENFUKU2K) as RENFUKU2K,
# MAGIC max(T2.RENTAN3) as  RENTAN3, 
# MAGIC max(T2.RENTAN3K) as RENTAN3K,
# MAGIC max(T2.RENFUKU3) as RENFUKU3,
# MAGIC max(T2.RENFUKU3K) as RENFUKU3K,
# MAGIC max(T2.WIDE1)   as WIDE1,
# MAGIC max(T2.WIDE1K)  as WIDE1K,
# MAGIC max(T2.WIDE2)   as WIDE2,
# MAGIC max(T2.WIDE2K)  as WIDE2K,
# MAGIC max(T2.WIDE3)   as WIDE3,
# MAGIC max(T2.WIDE3K)  as WIDE3K,
# MAGIC max(T2.FUKU1)   as FUKU1,
# MAGIC max(T2.FUKU1K)  as FUKU1K,
# MAGIC max(T2.FUKU2)   as FUKU2,
# MAGIC max(T2.FUKU2K)  as FUKU2K
# MAGIC
# MAGIC from
# MAGIC (select distinct * from BANGUMI) as T1 -- distinctで結合
# MAGIC LEFT OUTER join 
# MAGIC (select distinct * from RESULT)  as T2 -- distinctで結合
# MAGIC on 
# MAGIC T1.RACEDATE = T2.RACEDATE
# MAGIC -- フォーマットが異なるため数値に変換しないと結合できなかった
# MAGIC and cast(T1.PLACE as int)  = cast(T2.PLACE as int) and
# MAGIC T1.RACE = T2.RACE
# MAGIC -- データ取得日によって値の異なるデータがあったのでMAX値で一意にする
# MAGIC GROUP BY 1,2,3
# MAGIC ;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN RACEDATE COMMENT "レース開催日";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN PLACE COMMENT "レース開催地の名称";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN RACE COMMENT "レース番号";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN PLAYERID1 COMMENT "１号艇のレーサーID";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN PLAYERID2 COMMENT "２号艇のレーサーID";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN PLAYERID3 COMMENT "３号艇のレーサーID";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN PLAYERID4 COMMENT "4号艇のレーサーID";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN PLAYERID5 COMMENT "５号艇のレーサーID";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN PLAYERID6 COMMENT "６号艇のレーサーID";
# MAGIC
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN CLASS1 COMMENT "１号艇のレーサーグレード（A1:400,A2:300,B1:200,B2:100)";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN CLASS2 COMMENT "2号艇のレーサーグレード（A1:400,A2:300,B1:200,B2:100)";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN CLASS3 COMMENT "3号艇のレーサーグレード（A1:400,A2:300,B1:200,B2:100)";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN CLASS4 COMMENT "4号艇のレーサーグレード（A1:400,A2:300,B1:200,B2:100)";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN CLASS5 COMMENT "5号艇のレーサーグレード（A1:400,A2:300,B1:200,B2:100)";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN CLASS6 COMMENT "6号艇のレーサーグレード（A1:400,A2:300,B1:200,B2:100)";
# MAGIC
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN CLUB1 COMMENT "１号艇のレーサー所属地";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN CLUB2 COMMENT "２号艇のレーサー所属地";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN CLUB3 COMMENT "３号艇のレーサー所属地";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN CLUB4 COMMENT "４号艇のレーサー所属地";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN CLUB5 COMMENT "５号艇のレーサー所属地";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN CLUB6 COMMENT "６号艇のレーサー所属地";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN AGE1 COMMENT "１号艇レーサーの年齢";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN AGE2 COMMENT "２号艇レーサーの年齢";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN AGE3 COMMENT "３号艇レーサーの年齢";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN AGE4 COMMENT "４号艇レーサーの年齢";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN AGE5 COMMENT "５号艇レーサーの年齢";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN AGE6 COMMENT "６号艇レーサーの年齢";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WEIGHT1 COMMENT "１号艇レーサーの体重";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WEIGHT2 COMMENT "２号艇レーサーの体重";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WEIGHT3 COMMENT "３号艇レーサーの体重";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WEIGHT4 COMMENT "４号艇レーサーの体重";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WEIGHT5 COMMENT "５号艇レーサーの体重";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WEIGHT6 COMMENT "６号艇レーサーの体重";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN F1 COMMENT "１号艇レーサーのスタートフライング回数";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN F2 COMMENT "２号艇レーサーのスタートフライング回数";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN F3 COMMENT "３号艇レーサーのスタートフライング回数";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN F4 COMMENT "４号艇レーサーのスタートフライング回数";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN F5 COMMENT "５号艇レーサーのスタートフライング回数";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN F6 COMMENT "６号艇レーサーのスタートフライング回数";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN L1 COMMENT "１号艇レーサーのスタート遅れ回数";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN L2 COMMENT "２号艇レーサーのスタート遅れ回数";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN L3 COMMENT "３号艇レーサーのスタート遅れ回数";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN L4 COMMENT "４号艇レーサーのスタート遅れ回数";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN L5 COMMENT "５号艇レーサーのスタート遅れ回数";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN L6 COMMENT "６号艇レーサーのスタート遅れ回数";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN1RATE1 COMMENT "１号艇レーサーの全レース場での１着勝率（過去６ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN1RATE2 COMMENT "２号艇レーサーの全レース場での１着勝率（過去６ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN1RATE3 COMMENT "３号艇レーサーの全レース場での１着勝率（過去６ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN1RATE4 COMMENT "４号艇レーサーの全レース場での１着勝率（過去６ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN1RATE5 COMMENT "５号艇レーサーの全レース場での１着勝率（過去６ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN1RATE6 COMMENT "６号艇レーサーの全レース場での１着勝率（過去６ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN2RATE1 COMMENT "１号艇レーサーの全レース場での２着以内勝率（過去６ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN2RATE2 COMMENT "２号艇レーサーの全レース場での２着以内勝率（過去６ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN2RATE3 COMMENT "３号艇レーサーの全レース場での２着以内勝率（過去６ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN2RATE4 COMMENT "４号艇レーサーの全レース場での２着以内勝率（過去６ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN2RATE5 COMMENT "５号艇レーサーの全レース場での２着以内勝率（過去６ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN2RATE6 COMMENT "６号艇レーサーの全レース場での２着以内勝率（過去６ヶ月以内）";
# MAGIC
# MAGIC --ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN3RATE1 COMMENT "１号艇レーサーの全レース場での３着以内勝率（過去６ヶ月以内）";
# MAGIC --ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN3RATE2 COMMENT "２号艇レーサーの全レース場での３着以内勝率（過去６ヶ月以内）";
# MAGIC --ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN3RATE3 COMMENT "３号艇レーサーの全レース場での３着以内勝率（過去６ヶ月以内）";
# MAGIC --ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN3RATE4 COMMENT "４号艇レーサーの全レース場での３着以内勝率（過去６ヶ月以内）";
# MAGIC --ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN3RATE5 COMMENT "５号艇レーサーの全レース場での３着以内勝率（過去６ヶ月以内）";
# MAGIC --ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIN3RATE6 COMMENT "６号艇レーサーの全レース場での３着以内勝率（過去６ヶ月以内）";
# MAGIC
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN1RATE1 COMMENT "１号艇レーサーのレース開催地レース場での１着勝率（過去２４ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN1RATE2 COMMENT "２号艇レーサーのレース開催地レース場での１着勝率（過去２４ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN1RATE3 COMMENT "３号艇レーサーのレース開催地レース場での１着勝率（過去２４ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN1RATE4 COMMENT "４号艇レーサーのレース開催地レース場での１着勝率（過去２４ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN1RATE5 COMMENT "５号艇レーサーのレース開催地レース場での１着勝率（過去２４ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN1RATE6 COMMENT "６号艇レーサーのレース開催地レース場での１着勝率（過去２４ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN2RATE1 COMMENT "１号艇レーサーのレース開催地レース場での２着以内勝率（過去２４ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN2RATE2 COMMENT "２号艇レーサーのレース開催地レース場での２着以内勝率（過去２４ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN2RATE3 COMMENT "３号艇レーサーのレース開催地レース場での２着以内勝率（過去２４ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN2RATE4 COMMENT "４号艇レーサーのレース開催地レース場での２着以内勝率（過去２４ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN2RATE5 COMMENT "５号艇レーサーのレース開催地レース場での２着以内勝率（過去２４ヶ月以内）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN2RATE6 COMMENT "６号艇レーサーのレース開催地レース場での２着以内勝率（過去２４ヶ月以内）";
# MAGIC
# MAGIC --ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN3RATE1 COMMENT "１号艇レーサーのレース開催地レース場での３着以内勝率（過去２４ヶ月以内）";
# MAGIC --ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN3RATE2 COMMENT "２号艇レーサーのレース開催地レース場での３着以内勝率（過去２４ヶ月以内）";
# MAGIC --ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN3RATE3 COMMENT "３号艇レーサーのレース開催地レース場での３着以内勝率（過去２４ヶ月以内）";
# MAGIC --ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN3RATE4 COMMENT "４号艇レーサーのレース開催地レース場での３着以内勝率（過去２４ヶ月以内）";
# MAGIC --ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN3RATE5 COMMENT "５号艇レーサーのレース開催地レース場での３着以内勝率（過去２４ヶ月以内）";
# MAGIC --ALTER TABLE TRAINING_BRONZE ALTER COLUMN LOCALWIN3RATE6 COMMENT "６号艇レーサーのレース開催地レース場での３着以内勝率（過去２４ヶ月以内）";
# MAGIC
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN MOTORWIN2RATE1 COMMENT "１号艇のモーター２着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN MOTORWIN2RATE2 COMMENT "２号艇のモーター２着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN MOTORWIN2RATE3 COMMENT "３号艇のモーター２着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN MOTORWIN2RATE4 COMMENT "４号艇のモーター２着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN MOTORWIN2RATE5 COMMENT "５号艇のモーター２着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN MOTORWIN2RATE6 COMMENT "６号艇のモーター２着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN MOTORWIN3RATE1 COMMENT "１号艇のモーター３着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN MOTORWIN3RATE2 COMMENT "２号艇のモーター３着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN MOTORWIN3RATE3 COMMENT "３号艇のモーター３着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN MOTORWIN3RATE4 COMMENT "４号艇のモーター３着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN MOTORWIN3RATE5 COMMENT "５号艇のモーター３着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN MOTORWIN3RATE6 COMMENT "６号艇のモーター３着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN BOATWIN2RATE1 COMMENT "１号艇のボート２着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN BOATWIN2RATE2 COMMENT "２号艇のボート２着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN BOATWIN2RATE3 COMMENT "３号艇のボート２着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN BOATWIN2RATE4 COMMENT "４号艇のボート２着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN BOATWIN2RATE5 COMMENT "５号艇のボート２着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN BOATWIN2RATE6 COMMENT "６号艇のボート２着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN BOATWIN3RATE1 COMMENT "１号艇のボート３着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN BOATWIN3RATE2 COMMENT "2号艇のボート３着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN BOATWIN3RATE3 COMMENT "3号艇のボート３着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN BOATWIN3RATE4 COMMENT "4号艇のボート３着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN BOATWIN3RATE5 COMMENT "5号艇のボート３着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN BOATWIN3RATE6 COMMENT "6号艇のボート３着以内勝率";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN ST_AVG1 COMMENT "１号艇レーサーのスタートタイミング（０に近いほど良い）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN ST_AVG2 COMMENT "２号艇レーサーのスタートタイミング（０に近いほど良い）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN ST_AVG3 COMMENT "３号艇レーサーのスタートタイミング（０に近いほど良い）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN ST_AVG4 COMMENT "４号艇レーサーのスタートタイミング（０に近いほど良い）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN ST_AVG5 COMMENT "５号艇レーサーのスタートタイミング（０に近いほど良い）";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN ST_AVG6 COMMENT "６号艇レーサーのスタートタイミング（０に近いほど良い）";

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN RACEDATE COMMENT "レース開催日";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN PLACE COMMENT "レース開催地の名称";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN RACE COMMENT "レース番号";
# MAGIC
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN RENTAN3 COMMENT "３連単";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN RENTAN3K COMMENT "3連単払い戻し金額";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN RENFUKU3 COMMENT "３連複";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN RENFUKU3K COMMENT "3連複払い戻し金額";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN RENTAN2 COMMENT "２連単";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN RENTAN2K COMMENT "２連単払い戻し金額";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN RENFUKU2 COMMENT "２連複";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN RENFUKU2K COMMENT "２連複払い戻し金額";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIDE1 COMMENT "拡連複１";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIDE1K COMMENT "拡連複１払い戻し金額";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIDE2 COMMENT "拡連複２";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIDE2K COMMENT "拡連複２払い戻し金額";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIDE3 COMMENT "拡連複３";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN WIDE3K COMMENT "拡連複３払い戻し金額";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN TAN COMMENT "単勝";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN TANK COMMENT "単勝払い戻し金額";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN FUKU1 COMMENT "複勝１";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN FUKU1K COMMENT "複勝１払い戻し金額";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN FUKU2 COMMENT "複勝２";
# MAGIC ALTER TABLE TRAINING_BRONZE ALTER COLUMN FUKU2K COMMENT "複勝２払い戻し金額";

# COMMAND ----------

# MAGIC %sql
# MAGIC select racedate,count(1) from training_bronze group by 1 order by 1
# MAGIC ;

# COMMAND ----------

# DBTITLE 1,データ確認用
# MAGIC %sql
# MAGIC select * from training_bronze order by racedate,place,cast(race as int);

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT DATE_FORMAT(current_date(), 'yyyyMMdd');

# COMMAND ----------

# DBTITLE 1,シルバーテーブル：データ整形＋フィルタリング
# MAGIC %sql
# MAGIC create or replace table TRAINING_SILVER as
# MAGIC select 
# MAGIC distinct
# MAGIC RACEDATE
# MAGIC ,PLACE
# MAGIC ,RACE
# MAGIC ,PLAYERID1
# MAGIC ,PLAYERID2
# MAGIC ,PLAYERID3
# MAGIC ,PLAYERID4
# MAGIC ,PLAYERID5
# MAGIC ,PLAYERID6
# MAGIC -- このままCLASSはカテゴリで持たせても同じだと思うが。
# MAGIC ,case 
# MAGIC         when CLASS1 = "A1" then cast(400 as double)
# MAGIC         when CLASS1 = "A2" then cast(300 as double)
# MAGIC         when CLASS1 = "B1" then cast(200 as double)
# MAGIC         when CLASS1 = "B2" then cast(100 as double)
# MAGIC end as CLASS1
# MAGIC ,case 
# MAGIC         when CLASS2 = "A1" then cast(400 as double)
# MAGIC         when CLASS2 = "A2" then cast(300 as double)
# MAGIC         when CLASS2 = "B1" then cast(200 as double)
# MAGIC         when CLASS2 = "B2" then cast(100 as double)
# MAGIC end as CLASS2 
# MAGIC ,case 
# MAGIC         when CLASS3 = "A1" then cast(400 as double)
# MAGIC         when CLASS3 = "A2" then cast(300 as double)
# MAGIC         when CLASS3 = "B1" then cast(200 as double)
# MAGIC         when CLASS3 = "B2" then cast(100 as double)
# MAGIC end as CLASS3  
# MAGIC ,case 
# MAGIC         when CLASS4 = "A1" then cast(400 as double)
# MAGIC         when CLASS4 = "A2" then cast(300 as double)
# MAGIC         when CLASS4 = "B1" then cast(200 as double)
# MAGIC         when CLASS4 = "B2" then cast(100 as double)
# MAGIC end as CLASS4 
# MAGIC ,case 
# MAGIC         when CLASS5 = "A1" then cast(400 as double)
# MAGIC         when CLASS5 = "A2" then cast(300 as double)
# MAGIC         when CLASS5 = "B1" then cast(200 as double)
# MAGIC         when CLASS5 = "B2" then cast(100 as double)
# MAGIC end as CLASS5  
# MAGIC ,case 
# MAGIC         when CLASS6 = "A1" then cast(400 as double)
# MAGIC         when CLASS6 = "A2" then cast(300 as double)
# MAGIC         when CLASS6 = "B1" then cast(200 as double)
# MAGIC         when CLASS6 = "B2" then cast(100 as double)
# MAGIC end as CLASS6 
# MAGIC ,CLUB1
# MAGIC ,CLUB2
# MAGIC ,CLUB3
# MAGIC ,CLUB4
# MAGIC ,CLUB5
# MAGIC ,CLUB6
# MAGIC ,AGE1
# MAGIC ,AGE2
# MAGIC ,AGE3
# MAGIC ,AGE4
# MAGIC ,AGE5
# MAGIC ,AGE6
# MAGIC ,WEIGHT1
# MAGIC ,WEIGHT2
# MAGIC ,WEIGHT3
# MAGIC ,WEIGHT4
# MAGIC ,WEIGHT5
# MAGIC ,WEIGHT6
# MAGIC ,F1
# MAGIC ,F2
# MAGIC ,F3
# MAGIC ,F4
# MAGIC ,F5
# MAGIC ,F6
# MAGIC ,L1
# MAGIC ,L2
# MAGIC ,L3
# MAGIC ,L4
# MAGIC ,L5
# MAGIC ,L6
# MAGIC ,WIN1RATE1
# MAGIC ,WIN1RATE2
# MAGIC ,WIN1RATE3
# MAGIC ,WIN1RATE4
# MAGIC ,WIN1RATE5
# MAGIC ,WIN1RATE6
# MAGIC ,WIN2RATE1
# MAGIC ,WIN2RATE2
# MAGIC ,WIN2RATE3
# MAGIC ,WIN2RATE4
# MAGIC ,WIN2RATE5
# MAGIC ,WIN2RATE6
# MAGIC ,LOCALWIN1RATE1
# MAGIC ,LOCALWIN1RATE2
# MAGIC ,LOCALWIN1RATE3
# MAGIC ,LOCALWIN1RATE4
# MAGIC ,LOCALWIN1RATE5
# MAGIC ,LOCALWIN1RATE6
# MAGIC ,LOCALWIN2RATE1
# MAGIC ,LOCALWIN2RATE2
# MAGIC ,LOCALWIN2RATE3
# MAGIC ,LOCALWIN2RATE4
# MAGIC ,LOCALWIN2RATE5
# MAGIC ,LOCALWIN2RATE6
# MAGIC ,MOTORWIN2RATE1
# MAGIC ,MOTORWIN2RATE2
# MAGIC ,MOTORWIN2RATE3
# MAGIC ,MOTORWIN2RATE4
# MAGIC ,MOTORWIN2RATE5
# MAGIC ,MOTORWIN2RATE6
# MAGIC ,MOTORWIN3RATE1
# MAGIC ,MOTORWIN3RATE2
# MAGIC ,MOTORWIN3RATE3
# MAGIC ,MOTORWIN3RATE4
# MAGIC ,MOTORWIN3RATE5
# MAGIC ,MOTORWIN3RATE6
# MAGIC ,BOATWIN2RATE1
# MAGIC ,BOATWIN2RATE2
# MAGIC ,BOATWIN2RATE3
# MAGIC ,BOATWIN2RATE4
# MAGIC ,BOATWIN2RATE5
# MAGIC ,BOATWIN2RATE6
# MAGIC ,BOATWIN3RATE1
# MAGIC ,BOATWIN3RATE2
# MAGIC ,BOATWIN3RATE3
# MAGIC ,BOATWIN3RATE4
# MAGIC ,BOATWIN3RATE5
# MAGIC ,BOATWIN3RATE6
# MAGIC ,ST_AVG1
# MAGIC ,ST_AVG2
# MAGIC ,ST_AVG3
# MAGIC ,ST_AVG4
# MAGIC ,ST_AVG5
# MAGIC ,ST_AVG6
# MAGIC ,TAN
# MAGIC ,TANK
# MAGIC ,FUKU1
# MAGIC ,FUKU1K
# MAGIC ,FUKU2
# MAGIC ,FUKU2K
# MAGIC ,WIDE1
# MAGIC ,WIDE1K
# MAGIC ,WIDE2
# MAGIC ,WIDE2K
# MAGIC ,WIDE3
# MAGIC ,WIDE3K
# MAGIC ,RENTAN2
# MAGIC ,RENTAN2K
# MAGIC ,RENFUKU2
# MAGIC ,RENFUKU2K
# MAGIC ,RENFUKU3
# MAGIC ,RENFUKU3K
# MAGIC ,RENTAN3
# MAGIC ,RENTAN3K
# MAGIC from TRAINING_BRONZE
# MAGIC -- 統計取得の開始日
# MAGIC where 
# MAGIC (
# MAGIC         -- 直近の１年間のデータのみ使用
# MAGIC         racedate >= '20220101' 
# MAGIC         -- 実行日以前のデータはnullを含むレコードは削除
# MAGIC         and ( racedate is not null and racedate < DATE_FORMAT(current_date(), 'yyyyMMdd') ) 
# MAGIC         and ( rentan3 is not null and racedate < DATE_FORMAT(current_date(), 'yyyyMMdd') ) 
# MAGIC ) 
# MAGIC or
# MAGIC -- 実行日以降のデータは予測対象として残しておきたいのでnullでも許可する
# MAGIC racedate >= DATE_FORMAT(current_date(), 'yyyyMMdd')
# MAGIC ;
# MAGIC
# MAGIC select racedate,count(1) from TRAINING_SILVER group by racedate order by racedate ;

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC ALTER TABLE TRAINING_SILVER ALTER COLUMN CLASS1 COMMENT "１号艇のレーサーグレード（A1:400,A2:300,B1:200,B2:100)";
# MAGIC ALTER TABLE TRAINING_SILVER ALTER COLUMN CLASS2 COMMENT "2号艇のレーサーグレード（A1:400,A2:300,B1:200,B2:100)";
# MAGIC ALTER TABLE TRAINING_SILVER ALTER COLUMN CLASS3 COMMENT "3号艇のレーサーグレード（A1:400,A2:300,B1:200,B2:100)";
# MAGIC ALTER TABLE TRAINING_SILVER ALTER COLUMN CLASS4 COMMENT "4号艇のレーサーグレード（A1:400,A2:300,B1:200,B2:100)";
# MAGIC ALTER TABLE TRAINING_SILVER ALTER COLUMN CLASS5 COMMENT "5号艇のレーサーグレード（A1:400,A2:300,B1:200,B2:100)";
# MAGIC ALTER TABLE TRAINING_SILVER ALTER COLUMN CLASS6 COMMENT "6号艇のレーサーグレード（A1:400,A2:300,B1:200,B2:100)";

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from TRAINING_SILVER 
# MAGIC order by racedate desc ,place,cast(race as int);
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select racedate,count(1) from TRAINING_SILVER 
# MAGIC where rentan3 = '不成立' group by 1 order by racedate
