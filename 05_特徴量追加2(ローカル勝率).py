# Databricks notebook source
# MAGIC %md
# MAGIC # 特徴量作成
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

# MAGIC %md ## 特徴量の追加

# COMMAND ----------

# DBTITLE 0,1.番組表データから、コース1の１着率、２着率、３着率の実績テーブルを作成する。
# MAGIC %python
# MAGIC #1.番組表データから、コース1の１着率、２着率、３着率の実績テーブルを作成する。
# MAGIC df_COURCE1_LOCAL_WINRATE = sql(
# MAGIC """
# MAGIC select * from (
# MAGIC select 
# MAGIC   RACEDATE as RACEDATE_LATEST,
# MAGIC   --LEADで日付をずらし前日までの集計でトレーニングができるようにする
# MAGIC   LEAD(RACEDATE) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACEDATE,
# MAGIC   PLAYERID as PLAYERID1,
# MAGIC   PLACE,
# MAGIC   CAST(RACE_COUNT as double) as COURCE1_LOCAL_RACE_COUNT1,
# MAGIC   ROUND(CAST(( WIN1_COUNT / RACE_COUNT) as double),3) COURCE1_LOCAL_WIN1_RATE1,
# MAGIC   ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT)/ RACE_COUNT) as double),3) COURCE1_LOCAL_WIN12_RATE1,
# MAGIC   ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT+WIN3_COUNT)/ RACE_COUNT) as double),3) COURCE1_LOCAL_WIN123_RATE1
# MAGIC   from
# MAGIC   (
# MAGIC select
# MAGIC   RACEDATE,
# MAGIC   PLAYERID,
# MAGIC   PLACE,
# MAGIC   SUM(RACE_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as RACE_COUNT,
# MAGIC   SUM(WIN1_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN1_COUNT,
# MAGIC   SUM(WIN2_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN2_COUNT,
# MAGIC   SUM(WIN3_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN3_COUNT
# MAGIC from(
# MAGIC     select
# MAGIC       RACEDATE,
# MAGIC       PLAYERID1 as PLAYERID ,
# MAGIC       PLACE,
# MAGIC       COUNT(1) as RACE_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 1, 1) = 1 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN1_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 3, 1) = 1 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN2_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 5, 1) = 1 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN3_COUNT
# MAGIC     from
# MAGIC       TRAINING_SILVER
# MAGIC       --training_base1
# MAGIC     GROUP BY
# MAGIC       RACEDATE,
# MAGIC       PLAYERID,
# MAGIC       PLACE
# MAGIC   )
# MAGIC GROUP BY
# MAGIC   RACEDATE,
# MAGIC   PLAYERID,
# MAGIC   PLACE,
# MAGIC   RACE_COUNT,
# MAGIC   WIN1_COUNT,
# MAGIC   WIN2_COUNT,
# MAGIC   WIN3_COUNT
# MAGIC   )
# MAGIC ) as T1 WHERE RACEDATE IS NOT NULL
# MAGIC   """)
# MAGIC
# MAGIC display(df_COURCE1_LOCAL_WINRATE)
# MAGIC df_COURCE1_LOCAL_WINRATE.createOrReplaceTempView('local1')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from local1 where playerid1 = 'p2014' order by racedate;

# COMMAND ----------

# MAGIC %python
# MAGIC #2.番組表データから、コース2の１着率、２着率、３着率の実績テーブルを作成する。
# MAGIC df_COURCE2_LOCAL_WINRATE = sql(
# MAGIC """
# MAGIC select * from (
# MAGIC select 
# MAGIC   RACEDATE as RACEDATE_LATEST,
# MAGIC   --LEADで日付をずらし前日までの集計でトレーニングができるようにする
# MAGIC   LEAD(RACEDATE) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACEDATE,
# MAGIC   PLAYERID as PLAYERID2,
# MAGIC   PLACE,
# MAGIC   CAST(RACE_COUNT as double) as COURCE2_LOCAL_RACE_COUNT2,
# MAGIC   ROUND(CAST(( WIN1_COUNT / RACE_COUNT) as double),3) COURCE2_LOCAL_WIN1_RATE2,
# MAGIC   ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT)/ RACE_COUNT) as double),3) COURCE2_LOCAL_WIN12_RATE2,
# MAGIC   ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT+WIN3_COUNT)/ RACE_COUNT) as double),3) COURCE2_LOCAL_WIN123_RATE2
# MAGIC   from
# MAGIC   (
# MAGIC select
# MAGIC   RACEDATE,
# MAGIC   PLAYERID,
# MAGIC   PLACE,
# MAGIC   SUM(RACE_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as RACE_COUNT,
# MAGIC   SUM(WIN1_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN1_COUNT,
# MAGIC   SUM(WIN2_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN2_COUNT,
# MAGIC   SUM(WIN3_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN3_COUNT
# MAGIC from(
# MAGIC     select
# MAGIC       RACEDATE,
# MAGIC       PLAYERID2 as PLAYERID ,
# MAGIC       PLACE,
# MAGIC       COUNT(1) as RACE_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 1, 1) = 2 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN1_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 3, 1) = 2 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN2_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 5, 1) = 2 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN3_COUNT
# MAGIC     from
# MAGIC       TRAINING_SILVER
# MAGIC       --training_base1
# MAGIC     GROUP BY
# MAGIC       RACEDATE,
# MAGIC       PLAYERID,
# MAGIC       PLACE
# MAGIC   )
# MAGIC GROUP BY
# MAGIC   RACEDATE,
# MAGIC   PLAYERID,
# MAGIC   PLACE,
# MAGIC   RACE_COUNT,
# MAGIC   WIN1_COUNT,
# MAGIC   WIN2_COUNT,
# MAGIC   WIN3_COUNT
# MAGIC   )
# MAGIC ) as T1 WHERE RACEDATE IS NOT NULL
# MAGIC   """)
# MAGIC
# MAGIC display(df_COURCE2_LOCAL_WINRATE)
# MAGIC df_COURCE2_LOCAL_WINRATE.createOrReplaceTempView('local2')

# COMMAND ----------

# DBTITLE 1,チェック用
# MAGIC %sql
# MAGIC select * from local2 where playerid2 = 'p2014' order by racedate;

# COMMAND ----------

# MAGIC %python
# MAGIC #3.番組表データから、コース3の１着率、２着率、３着率の実績テーブルを作成する。
# MAGIC df_COURCE3_LOCAL_WINRATE = sql(
# MAGIC """
# MAGIC select * from (
# MAGIC select 
# MAGIC   RACEDATE as RACEDATE_LATEST,
# MAGIC   --LEADで日付をずらし前日までの集計でトレーニングができるようにする
# MAGIC   LEAD(RACEDATE) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACEDATE,
# MAGIC   PLAYERID as PLAYERID3,
# MAGIC   PLACE,
# MAGIC   CAST(RACE_COUNT as double) as COURCE3_LOCAL_RACE_COUNT3,
# MAGIC   ROUND(CAST(( WIN1_COUNT / RACE_COUNT) as double),3) COURCE3_LOCAL_WIN1_RATE3,
# MAGIC   ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT)/ RACE_COUNT) as double),3) COURCE3_LOCAL_WIN12_RATE3,
# MAGIC   ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT+WIN3_COUNT)/ RACE_COUNT) as double),3) COURCE3_LOCAL_WIN123_RATE3
# MAGIC   from
# MAGIC   (
# MAGIC select
# MAGIC   RACEDATE,
# MAGIC   PLAYERID,
# MAGIC   PLACE,
# MAGIC   SUM(RACE_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as RACE_COUNT,
# MAGIC   SUM(WIN1_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN1_COUNT,
# MAGIC   SUM(WIN2_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN2_COUNT,
# MAGIC   SUM(WIN3_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN3_COUNT
# MAGIC from(
# MAGIC     select
# MAGIC       RACEDATE,
# MAGIC       PLAYERID3 as PLAYERID ,
# MAGIC       PLACE,
# MAGIC       COUNT(1) as RACE_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 1, 1) = 3 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN1_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 3, 1) = 3 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN2_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 5, 1) = 3 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN3_COUNT
# MAGIC     from
# MAGIC       TRAINING_SILVER
# MAGIC       --training_base1
# MAGIC     GROUP BY
# MAGIC       RACEDATE,
# MAGIC       PLAYERID,
# MAGIC       PLACE
# MAGIC   )
# MAGIC GROUP BY
# MAGIC   RACEDATE,
# MAGIC   PLAYERID,
# MAGIC   PLACE,
# MAGIC   RACE_COUNT,
# MAGIC   WIN1_COUNT,
# MAGIC   WIN2_COUNT,
# MAGIC   WIN3_COUNT
# MAGIC   )
# MAGIC ) as T1 WHERE RACEDATE IS NOT NULL
# MAGIC   """)
# MAGIC
# MAGIC display(df_COURCE3_LOCAL_WINRATE)
# MAGIC df_COURCE3_LOCAL_WINRATE.createOrReplaceTempView('local3')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from local3 where playerid3 = 'p2014' order by racedate;

# COMMAND ----------

# MAGIC %python
# MAGIC #4.番組表データから、コース4の１着率、２着率、３着率の実績テーブルを作成する。
# MAGIC df_COURCE4_LOCAL_WINRATE = sql(
# MAGIC """
# MAGIC select * from (
# MAGIC select 
# MAGIC   RACEDATE as RACEDATE_LATEST,
# MAGIC   --LEADで日付をずらし前日までの集計でトレーニングができるようにする
# MAGIC   LEAD(RACEDATE) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACEDATE,
# MAGIC   PLAYERID as PLAYERID4,
# MAGIC   PLACE,
# MAGIC   CAST(RACE_COUNT as double) as COURCE4_LOCAL_RACE_COUNT4,
# MAGIC   ROUND(CAST(( WIN1_COUNT / RACE_COUNT) as double),3) COURCE4_LOCAL_WIN1_RATE4,
# MAGIC   ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT)/ RACE_COUNT) as double),3) COURCE4_LOCAL_WIN12_RATE4,
# MAGIC   ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT+WIN3_COUNT)/ RACE_COUNT) as double),3) COURCE4_LOCAL_WIN123_RATE4
# MAGIC   from
# MAGIC   (
# MAGIC select
# MAGIC   RACEDATE,
# MAGIC   PLAYERID,
# MAGIC   PLACE,
# MAGIC   SUM(RACE_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as RACE_COUNT,
# MAGIC   SUM(WIN1_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN1_COUNT,
# MAGIC   SUM(WIN2_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN2_COUNT,
# MAGIC   SUM(WIN3_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN3_COUNT
# MAGIC from(
# MAGIC     select
# MAGIC       RACEDATE,
# MAGIC       PLAYERID4 as PLAYERID ,
# MAGIC       PLACE,
# MAGIC       COUNT(1) as RACE_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 1, 1) = 4 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN1_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 3, 1) = 4 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN2_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 5, 1) = 4 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN3_COUNT
# MAGIC     from
# MAGIC       TRAINING_SILVER
# MAGIC       --training_base1
# MAGIC     GROUP BY
# MAGIC       RACEDATE,
# MAGIC       PLAYERID,
# MAGIC       PLACE
# MAGIC   )
# MAGIC GROUP BY
# MAGIC   RACEDATE,
# MAGIC   PLAYERID,
# MAGIC   PLACE,
# MAGIC   RACE_COUNT,
# MAGIC   WIN1_COUNT,
# MAGIC   WIN2_COUNT,
# MAGIC   WIN3_COUNT
# MAGIC   )
# MAGIC ) as T1 WHERE RACEDATE IS NOT NULL
# MAGIC   """)
# MAGIC
# MAGIC display(df_COURCE4_LOCAL_WINRATE)
# MAGIC df_COURCE4_LOCAL_WINRATE.createOrReplaceTempView('local4')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from local4 where playerid4 = 'p2014' order by racedate;

# COMMAND ----------

# MAGIC %python
# MAGIC #5.番組表データから、コース5の１着率、２着率、３着率の実績テーブルを作成する。
# MAGIC df_COURCE5_LOCAL_WINRATE = sql(
# MAGIC """
# MAGIC select * from (
# MAGIC select 
# MAGIC   RACEDATE as RACEDATE_LATEST,
# MAGIC   --LEADで日付をずらし前日までの集計でトレーニングができるようにする
# MAGIC   LEAD(RACEDATE) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACEDATE,
# MAGIC   PLAYERID as PLAYERID5,
# MAGIC   PLACE,
# MAGIC   CAST(RACE_COUNT as double) as COURCE5_LOCAL_RACE_COUNT5,
# MAGIC   ROUND(CAST(( WIN1_COUNT / RACE_COUNT) as double),3) COURCE5_LOCAL_WIN1_RATE5,
# MAGIC   ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT)/ RACE_COUNT) as double),3) COURCE5_LOCAL_WIN12_RATE5,
# MAGIC   ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT+WIN3_COUNT)/ RACE_COUNT) as double),3) COURCE5_LOCAL_WIN123_RATE5
# MAGIC   from
# MAGIC   (
# MAGIC select
# MAGIC   RACEDATE,
# MAGIC   PLAYERID,
# MAGIC   PLACE,
# MAGIC   SUM(RACE_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as RACE_COUNT,
# MAGIC   SUM(WIN1_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN1_COUNT,
# MAGIC   SUM(WIN2_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN2_COUNT,
# MAGIC   SUM(WIN3_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN3_COUNT
# MAGIC from(
# MAGIC     select
# MAGIC       RACEDATE,
# MAGIC       PLAYERID5 as PLAYERID ,
# MAGIC       PLACE,
# MAGIC       COUNT(1) as RACE_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 1, 1) = 5 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN1_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 3, 1) = 5 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN2_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 5, 1) = 5 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN3_COUNT
# MAGIC     from
# MAGIC       TRAINING_SILVER
# MAGIC       --training_base1
# MAGIC     GROUP BY
# MAGIC       RACEDATE,
# MAGIC       PLAYERID,
# MAGIC       PLACE
# MAGIC   )
# MAGIC GROUP BY
# MAGIC   RACEDATE,
# MAGIC   PLAYERID,
# MAGIC   PLACE,
# MAGIC   RACE_COUNT,
# MAGIC   WIN1_COUNT,
# MAGIC   WIN2_COUNT,
# MAGIC   WIN3_COUNT
# MAGIC   )
# MAGIC ) as T1 WHERE RACEDATE IS NOT NULL
# MAGIC   """)
# MAGIC
# MAGIC display(df_COURCE5_LOCAL_WINRATE)
# MAGIC df_COURCE5_LOCAL_WINRATE.createOrReplaceTempView('local5')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from local5 where playerid5 = 'p2014' order by racedate;

# COMMAND ----------

# MAGIC %python
# MAGIC #6.番組表データから、コース6の１着率、２着率、３着率の実績テーブルを作成する。
# MAGIC df_COURCE6_LOCAL_WINRATE = sql(
# MAGIC """
# MAGIC select * from (
# MAGIC select 
# MAGIC   RACEDATE as RACEDATE_LATEST,
# MAGIC   --LEADで日付をずらし前日までの集計でトレーニングができるようにする
# MAGIC   LEAD(RACEDATE) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACEDATE,
# MAGIC   PLAYERID as PLAYERID6,
# MAGIC   PLACE,
# MAGIC   CAST(RACE_COUNT as double) as COURCE6_LOCAL_RACE_COUNT6,
# MAGIC   ROUND(CAST(( WIN1_COUNT / RACE_COUNT) as double),3) COURCE6_LOCAL_WIN1_RATE6,
# MAGIC   ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT)/ RACE_COUNT) as double),3) COURCE6_LOCAL_WIN12_RATE6,
# MAGIC   ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT+WIN3_COUNT)/ RACE_COUNT) as double),3) COURCE6_LOCAL_WIN123_RATE6
# MAGIC   from
# MAGIC   (
# MAGIC select
# MAGIC   RACEDATE,
# MAGIC   PLAYERID,
# MAGIC   PLACE,
# MAGIC   SUM(RACE_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as RACE_COUNT,
# MAGIC   SUM(WIN1_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN1_COUNT,
# MAGIC   SUM(WIN2_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN2_COUNT,
# MAGIC   SUM(WIN3_COUNT) OVER (PARTITION BY PLAYERID,PLACE ORDER BY RACEDATE ) as WIN3_COUNT
# MAGIC from(
# MAGIC     select
# MAGIC       RACEDATE,
# MAGIC       PLAYERID6 as PLAYERID ,
# MAGIC       PLACE,
# MAGIC       COUNT(1) as RACE_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 1, 1) = 6 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN1_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 3, 1) = 6 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN2_COUNT,
# MAGIC       SUM(
# MAGIC         CASE
# MAGIC           WHEN SUBSTR(RENTAN3, 5, 1) = 6 THEN 1
# MAGIC           ELSE 0
# MAGIC         END
# MAGIC       ) as WIN3_COUNT
# MAGIC     from
# MAGIC       TRAINING_SILVER
# MAGIC       --training_base1
# MAGIC     GROUP BY
# MAGIC       RACEDATE,
# MAGIC       PLAYERID,
# MAGIC       PLACE
# MAGIC   )
# MAGIC GROUP BY
# MAGIC   RACEDATE,
# MAGIC   PLAYERID,
# MAGIC   PLACE,
# MAGIC   RACE_COUNT,
# MAGIC   WIN1_COUNT,
# MAGIC   WIN2_COUNT,
# MAGIC   WIN3_COUNT
# MAGIC   )
# MAGIC ) as T1 WHERE RACEDATE IS NOT NULL
# MAGIC   """)
# MAGIC
# MAGIC display(df_COURCE6_LOCAL_WINRATE)
# MAGIC df_COURCE6_LOCAL_WINRATE.createOrReplaceTempView('local6')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from local6 where playerid6 = 'p2014' order by racedate;

# COMMAND ----------

# MAGIC %md ### フィーチャーストアライブラリを使用して新しい特徴量テーブルを作成 

# COMMAND ----------

# MAGIC %python
# MAGIC from databricks import feature_store
# MAGIC from pyspark.sql.functions import *
# MAGIC from pyspark.sql.types import FloatType, IntegerType, StringType
# MAGIC from pytz import timezone

# COMMAND ----------

# MAGIC %md 次に、create_table APIを使用してFeature Storeクライアントのインスタンスを作成します。</br>
# MAGIC https://docs.databricks.com/applications/machine-learning/feature-store/feature-tables.html

# COMMAND ----------

# MAGIC %python
# MAGIC fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md
# MAGIC create_table API (v0.3.6 以降) または create_feature_table API (v0.3.5 以降) を用いてスキーマおよび一意な ID キーを定義します。</br>
# MAGIC オプション引数 df (0.3.6 以上) または features_df (0.3.5 以下) を渡すと、そのデータを Feature Store にも書き出します。

# COMMAND ----------

# MAGIC %python
# MAGIC # 最初に既存のフィーチャーストアテーブルを削除しておきます。
# MAGIC try: 
# MAGIC   fs.drop_table("FS_COURCE1_LOCAL_WINRATE") 
# MAGIC except:
# MAGIC   print('table does not exists')
# MAGIC
# MAGIC try: 
# MAGIC   fs.drop_table("FS_COURCE2_LOCAL_WINRATE")
# MAGIC except:
# MAGIC   print('table does not exists')
# MAGIC
# MAGIC try:
# MAGIC   fs.drop_table("FS_COURCE3_LOCAL_WINRATE")
# MAGIC except:
# MAGIC   print('table does not exists')
# MAGIC
# MAGIC try:
# MAGIC   fs.drop_table("FS_COURCE4_LOCAL_WINRATE")
# MAGIC except:
# MAGIC   print('table does not exists')
# MAGIC
# MAGIC try:
# MAGIC   fs.drop_table("FS_COURCE5_LOCAL_WINRATE")
# MAGIC except:
# MAGIC   print('table does not exists')
# MAGIC
# MAGIC try:
# MAGIC   fs.drop_table("FS_COURCE6_LOCAL_WINRATE")
# MAGIC except:
# MAGIC   print('table does not exists')

# COMMAND ----------

# MAGIC %md データチェック

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from local1 where place is null;

# COMMAND ----------

# MAGIC %md フィーチャーテーブルを作成する

# COMMAND ----------

# MAGIC %python
# MAGIC # このセルは、Feature Storeクライアントv0.3.6で導入されたAPIを使用しています。
# MAGIC
# MAGIC fs.create_table(
# MAGIC     name="FS_COURCE1_LOCAL_WINRATE",
# MAGIC     primary_keys=["RACEDATE","PLACE","PLAYERID1"],#一意キーの指定
# MAGIC     df=df_COURCE1_LOCAL_WINRATE,#このデータフレームをフィーチャーストアに書き出す
# MAGIC     description="この場所での１コースでの勝率",
# MAGIC )

# COMMAND ----------

fs.create_table(
    name="FS_COURCE2_LOCAL_WINRATE",
    primary_keys=["RACEDATE","PLACE","PLAYERID2"],#一意キーの指定
    df=df_COURCE2_LOCAL_WINRATE,#このデータフレームをフィーチャーストアに書き出す
    description="この場所での2コースでの勝率",
)

# COMMAND ----------

fs.create_table(
    name="FS_COURCE3_LOCAL_WINRATE",
    primary_keys=["RACEDATE","PLACE","PLAYERID3"],#一意キーの指定
    df=df_COURCE3_LOCAL_WINRATE,#このデータフレームをフィーチャーストアに書き出す
    description="この場所での3コースでの勝率",
)

# COMMAND ----------

fs.create_table(
    name="FS_COURCE4_LOCAL_WINRATE",
    primary_keys=["RACEDATE","PLACE","PLAYERID4"],#一意キーの指定
    df=df_COURCE4_LOCAL_WINRATE,#このデータフレームをフィーチャーストアに書き出す
    description="この場所での4コースでの勝率",
)

# COMMAND ----------

fs.create_table(
    name="FS_COURCE5_LOCAL_WINRATE",
    primary_keys=["RACEDATE","PLACE","PLAYERID5"],#一意キーの指定
    df=df_COURCE5_LOCAL_WINRATE,#このデータフレームをフィーチャーストアに書き出す
    description="この場所での5コースでの勝率",
)

# COMMAND ----------

fs.create_table(
    name="FS_COURCE6_LOCAL_WINRATE",
    primary_keys=["RACEDATE","PLACE","PLAYERID6"],#一意キーの指定
    df=df_COURCE6_LOCAL_WINRATE,#このデータフレームをフィーチャーストアに書き出す
    description="この場所での6コースでの勝率",
)

# COMMAND ----------

# MAGIC %md ### トレーニング用データの作成

# COMMAND ----------

# MAGIC %python
# MAGIC from pyspark.sql import *
# MAGIC from pyspark.sql.functions import current_timestamp
# MAGIC from pyspark.sql.types import IntegerType
# MAGIC import math
# MAGIC from datetime import timedelta
# MAGIC import mlflow.pyfunc

# COMMAND ----------

# MAGIC %python
# MAGIC # 事前に定義しているシルバーテーブルを使用
# MAGIC training_base2 = sql("select * from TRAINING_SILVER")

# COMMAND ----------

# MAGIC %python
# MAGIC display(training_base2)

# COMMAND ----------

# MAGIC %python
# MAGIC from databricks.feature_store import FeatureLookup
# MAGIC import mlflow
# MAGIC
# MAGIC cource1_features_table = "FS_COURCE1_LOCAL_WINRATE"
# MAGIC cource2_features_table = "FS_COURCE2_LOCAL_WINRATE"
# MAGIC cource3_features_table = "FS_COURCE3_LOCAL_WINRATE"
# MAGIC cource4_features_table = "FS_COURCE4_LOCAL_WINRATE"
# MAGIC cource5_features_table = "FS_COURCE5_LOCAL_WINRATE"
# MAGIC cource6_features_table = "FS_COURCE6_LOCAL_WINRATE"
# MAGIC
# MAGIC #https://docs.databricks.com/dev-tools/api/python/latest/feature-store/databricks.feature_store.entities.feature_lookup.html
# MAGIC # Feature Store APIを使用して特徴量を取得
# MAGIC cource1_features = [
# MAGIC    FeatureLookup( 
# MAGIC      table_name = cource1_features_table,
# MAGIC      # フィーチャーストアのカラムを指定
# MAGIC      feature_names = ["COURCE1_LOCAL_RACE_COUNT1","COURCE1_LOCAL_WIN1_RATE1","COURCE1_LOCAL_WIN12_RATE1","COURCE1_LOCAL_WIN123_RATE1"],
# MAGIC      # ローデータのカラムを指定（フィーチャストアのプライマリキーと一致させる）
# MAGIC      lookup_key = ["RACEDATE","PLACE","PLAYERID1"],
# MAGIC    ),
# MAGIC ]
# MAGIC
# MAGIC cource2_features = [
# MAGIC    FeatureLookup( 
# MAGIC      table_name = cource2_features_table,
# MAGIC      # フィーチャーストアのカラムを指定
# MAGIC      feature_names = ["COURCE2_LOCAL_RACE_COUNT2","COURCE2_LOCAL_WIN1_RATE2","COURCE2_LOCAL_WIN12_RATE2","COURCE2_LOCAL_WIN123_RATE2"],
# MAGIC      # ローデータのカラムを指定（フィーチャストアのプライマリキーと一致させる）
# MAGIC      lookup_key = ["RACEDATE","PLACE","PLAYERID2"],
# MAGIC    ),
# MAGIC ]
# MAGIC
# MAGIC cource3_features = [
# MAGIC    FeatureLookup( 
# MAGIC      table_name = cource3_features_table,
# MAGIC      # フィーチャーストアのカラムを指定
# MAGIC      feature_names = ["COURCE3_LOCAL_RACE_COUNT3","COURCE3_LOCAL_WIN1_RATE3","COURCE3_LOCAL_WIN12_RATE3","COURCE3_LOCAL_WIN123_RATE3"],
# MAGIC      # ローデータのカラムを指定（フィーチャストアのプライマリキーと一致させる）
# MAGIC      lookup_key = ["RACEDATE","PLACE","PLAYERID3"],
# MAGIC    ),
# MAGIC ]
# MAGIC
# MAGIC cource4_features = [
# MAGIC    FeatureLookup( 
# MAGIC      table_name = cource4_features_table,
# MAGIC      # フィーチャーストアのカラムを指定
# MAGIC      feature_names = ["COURCE4_LOCAL_RACE_COUNT4","COURCE4_LOCAL_WIN1_RATE4","COURCE4_LOCAL_WIN12_RATE4","COURCE4_LOCAL_WIN123_RATE4"],
# MAGIC      # ローデータのカラムを指定（フィーチャストアのプライマリキーと一致させる）
# MAGIC      lookup_key = ["RACEDATE","PLACE","PLAYERID4"],
# MAGIC    ),
# MAGIC ]
# MAGIC
# MAGIC cource5_features = [
# MAGIC    FeatureLookup( 
# MAGIC      table_name = cource5_features_table,
# MAGIC      # フィーチャーストアのカラムを指定
# MAGIC      feature_names = ["COURCE5_LOCAL_RACE_COUNT5","COURCE5_LOCAL_WIN1_RATE5","COURCE5_LOCAL_WIN12_RATE5","COURCE5_LOCAL_WIN123_RATE5"],
# MAGIC      # ローデータのカラムを指定（フィーチャストアのプライマリキーと一致させる）
# MAGIC      lookup_key = ["RACEDATE","PLACE","PLAYERID5"],
# MAGIC    ),
# MAGIC ]
# MAGIC
# MAGIC cource6_features = [
# MAGIC    FeatureLookup( 
# MAGIC      table_name = cource6_features_table,
# MAGIC      # フィーチャーストアのカラムを指定
# MAGIC      feature_names = ["COURCE6_LOCAL_RACE_COUNT6","COURCE6_LOCAL_WIN1_RATE6","COURCE6_LOCAL_WIN12_RATE6","COURCE6_LOCAL_WIN123_RATE6"],
# MAGIC      # ローデータのカラムを指定（フィーチャストアのプライマリキーと一致させる）
# MAGIC      lookup_key = ["RACEDATE","PLACE","PLAYERID6"],
# MAGIC    ),
# MAGIC ]

# COMMAND ----------

# MAGIC %md ### トレーニングデータセットの作成
# MAGIC
# MAGIC 以下の `fs.create_training_set(...)` を呼び出すと、以下のような手順で処理が行われます:
# MAGIC
# MAGIC 1. モデルの学習に使用する特定の特徴量を Feature Store から選択する `TrainingSet` オブジェクトが作成されます。各特徴は、上記で作成した `FeatureLookup` で指定します。
# MAGIC
# MAGIC 1. 各フィーチャーは `FeatureLookup` の `lookup_key` に従って生の入力データと結合される。
# MAGIC
# MAGIC そして、`TrainingSet` は学習用の DataFrame に変換されます。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 新規に実験を作成

# COMMAND ----------

# MAGIC %python
# MAGIC # Case sensitive name
# MAGIC name = "/Users/mitsuhiro.itagaki@databricks.com/kyotei_feature2"

# COMMAND ----------

# 指定した実験がすでにあるか確認
experiment = mlflow.set_experiment(name)
# Get Experiment Details
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
print("Creation timestamp: {}".format(experiment.creation_time))

# 一旦消して
mlflow.delete_experiment(experiment.experiment_id)

# 再度作り直す
experiment = mlflow.set_experiment(name)
# Get Experiment Details
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
print("Creation timestamp: {}".format(experiment.creation_time))

# COMMAND ----------

# MAGIC %python
# MAGIC # End any existing runs (in the case this notebook is being run for a second time)
# MAGIC mlflow.end_run()
# MAGIC
# MAGIC # Start an mlflow run, which is needed for the feature store to log the model
# MAGIC mlflow.start_run() 
# MAGIC
# MAGIC ##################################################################
# MAGIC # 入力された生データに両特徴テーブルから対応する特徴を合成した学習セットを"作成する。
# MAGIC ##################################################################
# MAGIC training2_set = fs.create_training_set(
# MAGIC   training_base2,
# MAGIC   feature_lookups = cource1_features + cource2_features + cource3_features + cource4_features + cource5_features + cource6_features,
# MAGIC   label = "RENFUKU3",
# MAGIC   # 以下をカラムを除外してトレーニングを行わないようにします。
# MAGIC   exclude_columns = [""]
# MAGIC )
# MAGIC
# MAGIC # モデルを学習するために、TrainingSetをdataframeにロードし、sklearnに渡します。
# MAGIC training2_df = training2_set.load_df()

# COMMAND ----------

# MAGIC %python
# MAGIC #null値を０で埋める
# MAGIC training2_df = training2_df.fillna(0)

# COMMAND ----------

# MAGIC %python
# MAGIC # トレーニング用のデータフレームを表示し、生の入力データと特徴量テーブルの両方を含んでいることに注目しましょう
# MAGIC display(training2_df)

# COMMAND ----------

# MAGIC %python
# MAGIC training2_df.createOrReplaceTempView("training_tmp2")

# COMMAND ----------

# DBTITLE 1,データチェック用
# MAGIC %sql
# MAGIC select * from training_tmp2 where 
# MAGIC PLAYERID5 = 'p2014'

# COMMAND ----------

# DBTITLE 1,トレーニング用の特徴量を追加したテーブル2
# MAGIC %sql
# MAGIC create or replace table training_feature2 as
# MAGIC select 
# MAGIC distinct 
# MAGIC  RACEDATE -- 特徴量にはしない
# MAGIC ,PLACE -- 特徴量にはしない
# MAGIC ,RACE -- 特徴量にはしない
# MAGIC
# MAGIC ,PLAYERID1
# MAGIC ,PLAYERID2
# MAGIC ,PLAYERID3
# MAGIC ,PLAYERID4
# MAGIC ,PLAYERID5
# MAGIC ,PLAYERID6
# MAGIC
# MAGIC ,CLASS1
# MAGIC ,CLASS2
# MAGIC ,CLASS3
# MAGIC ,CLASS4
# MAGIC ,CLASS5
# MAGIC ,CLASS6
# MAGIC
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
# MAGIC
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
# MAGIC
# MAGIC ,WIN1RATE1
# MAGIC ,WIN1RATE2
# MAGIC ,WIN1RATE3
# MAGIC ,WIN1RATE4
# MAGIC ,WIN1RATE5
# MAGIC ,WIN1RATE6
# MAGIC
# MAGIC ,WIN2RATE1
# MAGIC ,WIN2RATE2
# MAGIC ,WIN2RATE3
# MAGIC ,WIN2RATE4
# MAGIC ,WIN2RATE5
# MAGIC ,WIN2RATE6
# MAGIC
# MAGIC ,LOCALWIN1RATE1
# MAGIC ,LOCALWIN1RATE2
# MAGIC ,LOCALWIN1RATE3
# MAGIC ,LOCALWIN1RATE4
# MAGIC ,LOCALWIN1RATE5
# MAGIC ,LOCALWIN1RATE6
# MAGIC
# MAGIC ,LOCALWIN2RATE1
# MAGIC ,LOCALWIN2RATE2
# MAGIC ,LOCALWIN2RATE3
# MAGIC ,LOCALWIN2RATE4
# MAGIC ,LOCALWIN2RATE5
# MAGIC ,LOCALWIN2RATE6
# MAGIC
# MAGIC ,MOTORWIN2RATE1
# MAGIC ,MOTORWIN2RATE2
# MAGIC ,MOTORWIN2RATE3
# MAGIC ,MOTORWIN2RATE4
# MAGIC ,MOTORWIN2RATE5
# MAGIC ,MOTORWIN2RATE6
# MAGIC
# MAGIC ,MOTORWIN3RATE1
# MAGIC ,MOTORWIN3RATE2
# MAGIC ,MOTORWIN3RATE3
# MAGIC ,MOTORWIN3RATE4
# MAGIC ,MOTORWIN3RATE5
# MAGIC ,MOTORWIN3RATE6
# MAGIC
# MAGIC ,BOATWIN2RATE1
# MAGIC ,BOATWIN2RATE2
# MAGIC ,BOATWIN2RATE3
# MAGIC ,BOATWIN2RATE4
# MAGIC ,BOATWIN2RATE5
# MAGIC ,BOATWIN2RATE6
# MAGIC
# MAGIC ,BOATWIN3RATE1
# MAGIC ,BOATWIN3RATE2
# MAGIC ,BOATWIN3RATE3
# MAGIC ,BOATWIN3RATE4
# MAGIC ,BOATWIN3RATE5
# MAGIC ,BOATWIN3RATE6
# MAGIC
# MAGIC ,ST_AVG1
# MAGIC ,ST_AVG2
# MAGIC ,ST_AVG3
# MAGIC ,ST_AVG4
# MAGIC ,ST_AVG5
# MAGIC ,ST_AVG6
# MAGIC
# MAGIC -- ,COURCE1_RACE_COUNT1
# MAGIC -- ,COURCE1_WIN1_RATE1
# MAGIC -- ,COURCE1_WIN12_RATE1
# MAGIC -- ,COURCE1_WIN123_RATE1
# MAGIC
# MAGIC -- ,COURCE2_RACE_COUNT2
# MAGIC -- ,COURCE2_WIN1_RATE2
# MAGIC -- ,COURCE2_WIN12_RATE2
# MAGIC -- ,COURCE2_WIN123_RATE2
# MAGIC
# MAGIC -- ,COURCE3_RACE_COUNT3
# MAGIC -- ,COURCE3_WIN1_RATE3
# MAGIC -- ,COURCE3_WIN12_RATE3
# MAGIC -- ,COURCE3_WIN123_RATE3
# MAGIC
# MAGIC -- ,COURCE4_RACE_COUNT4
# MAGIC -- ,COURCE4_WIN1_RATE4
# MAGIC -- ,COURCE4_WIN12_RATE4
# MAGIC -- ,COURCE4_WIN123_RATE4
# MAGIC
# MAGIC -- ,COURCE5_RACE_COUNT5
# MAGIC -- ,COURCE5_WIN1_RATE5
# MAGIC -- ,COURCE5_WIN12_RATE5
# MAGIC -- ,COURCE5_WIN123_RATE5
# MAGIC
# MAGIC -- ,COURCE6_RACE_COUNT6
# MAGIC -- ,COURCE6_WIN1_RATE6
# MAGIC -- ,COURCE6_WIN12_RATE6
# MAGIC -- ,COURCE6_WIN123_RATE6
# MAGIC
# MAGIC ,COURCE1_LOCAL_RACE_COUNT1
# MAGIC ,COURCE1_LOCAL_WIN1_RATE1
# MAGIC ,COURCE1_LOCAL_WIN12_RATE1
# MAGIC ,COURCE1_LOCAL_WIN123_RATE1
# MAGIC
# MAGIC ,COURCE2_LOCAL_RACE_COUNT2
# MAGIC ,COURCE2_LOCAL_WIN1_RATE2
# MAGIC ,COURCE2_LOCAL_WIN12_RATE2
# MAGIC ,COURCE2_LOCAL_WIN123_RATE2
# MAGIC
# MAGIC ,COURCE3_LOCAL_RACE_COUNT3
# MAGIC ,COURCE3_LOCAL_WIN1_RATE3
# MAGIC ,COURCE3_LOCAL_WIN12_RATE3
# MAGIC ,COURCE3_LOCAL_WIN123_RATE3
# MAGIC
# MAGIC ,COURCE4_LOCAL_RACE_COUNT4
# MAGIC ,COURCE4_LOCAL_WIN1_RATE4
# MAGIC ,COURCE4_LOCAL_WIN12_RATE4
# MAGIC ,COURCE4_LOCAL_WIN123_RATE4
# MAGIC
# MAGIC ,COURCE5_LOCAL_RACE_COUNT5
# MAGIC ,COURCE5_LOCAL_WIN1_RATE5
# MAGIC ,COURCE5_LOCAL_WIN12_RATE5
# MAGIC ,COURCE5_LOCAL_WIN123_RATE5
# MAGIC
# MAGIC ,COURCE6_LOCAL_RACE_COUNT6
# MAGIC ,COURCE6_LOCAL_WIN1_RATE6
# MAGIC ,COURCE6_LOCAL_WIN12_RATE6
# MAGIC ,COURCE6_LOCAL_WIN123_RATE6
# MAGIC
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
# MAGIC ,RENTAN3
# MAGIC ,RENTAN3K
# MAGIC ,RENFUKU3K
# MAGIC ,RENFUKU3
# MAGIC from training_tmp2
# MAGIC ;

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE1_LOCAL_RACE_COUNT1 COMMENT "１号艇レーサーの開催地レース場での１号艇でのレース回数";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE1_LOCAL_WIN1_RATE1 COMMENT "１号艇レーサーの開催地レース場での１号艇での１着勝率";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE1_LOCAL_WIN12_RATE1 COMMENT "１号艇レーサーの開催地レース場での１号艇での２着以内勝率";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE1_LOCAL_WIN123_RATE1 COMMENT "１号艇レーサーの開催地レース場での１号艇での３着以内勝率";
# MAGIC
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE2_LOCAL_RACE_COUNT2 COMMENT "２号艇レーサーの開催地レース場での２号艇でのレース回数";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE2_LOCAL_WIN1_RATE2 COMMENT "２号艇レーサーの開催地レース場での２号艇での１着勝率";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE2_LOCAL_WIN12_RATE2 COMMENT "２号艇レーサーの開催地レース場での２号艇での２着以内勝率";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE2_LOCAL_WIN123_RATE2 COMMENT "２号艇レーサーの開催地レース場での２号艇での３着以内勝率";
# MAGIC
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE3_LOCAL_RACE_COUNT3 COMMENT "３号艇レーサーの開催地レース場での３号艇でのレース回数";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE3_LOCAL_WIN1_RATE3 COMMENT "３号艇レーサーの開催地レース場での３号艇での１着勝率";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE3_LOCAL_WIN12_RATE3 COMMENT "３号艇レーサーの開催地レース場での３号艇での２着以内勝率";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE3_LOCAL_WIN123_RATE3 COMMENT "３号艇レーサーの開催地レース場での３号艇での３着以内勝率";
# MAGIC
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE4_LOCAL_RACE_COUNT4 COMMENT "４号艇レーサーの開催地レース場での4号艇でのレース回数";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE4_LOCAL_WIN1_RATE4 COMMENT "４号艇レーサーの開催地レース場での4号艇での１着勝率";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE4_LOCAL_WIN12_RATE4 COMMENT "４号艇レーサーの開催地レース場での4号艇での２着以内勝率";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE4_LOCAL_WIN123_RATE4 COMMENT "６号艇レーサーの開催地レース場での4号艇での３着以内勝率";
# MAGIC
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE5_LOCAL_RACE_COUNT5 COMMENT "５号艇レーサーの開催地レース場での5号艇でのレース回数";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE5_LOCAL_WIN1_RATE5 COMMENT "５号艇レーサーの開催地レース場での5号艇での１着勝率";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE5_LOCAL_WIN12_RATE5 COMMENT "５号艇レーサーの開催地レース場での5号艇での２着以内勝率";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE5_LOCAL_WIN123_RATE5 COMMENT "６号艇レーサーの開催地レース場での5号艇での３着以内勝率";
# MAGIC
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE6_LOCAL_RACE_COUNT6 COMMENT "６号艇レーサーの開催地レース場での5号艇でのレース回数";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE6_LOCAL_WIN1_RATE6 COMMENT "６号艇レーサーの開催地レース場での5号艇での１着勝率";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE6_LOCAL_WIN12_RATE6 COMMENT "６号艇レーサーの開催地レース場での5号艇での２着以内勝率";
# MAGIC ALTER TABLE training_feature2 ALTER COLUMN COURCE6_LOCAL_WIN123_RATE6 COMMENT "６号艇レーサーの開催地レース場での5号艇での３着以内勝率";
