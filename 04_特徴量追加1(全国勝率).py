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
#1.番組表データから、コース1の１着率、２着率、３着率の実績テーブルを作成する。
df_COURCE1_WINRATE = sql(
"""
select * from (
select 
  RACEDATE as RACEDATE_LATEST,
  --LEADで日付をずらし前日までの集計でトレーニングができるようにする
  LEAD(RACEDATE) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACEDATE,
  PLAYERID as PLAYERID1,
  CAST(RACE_COUNT as double) as COURCE1_RACE_COUNT1,
  ROUND(CAST(( WIN1_COUNT / RACE_COUNT) as double),3) COURCE1_WIN1_RATE1,
  ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT)/ RACE_COUNT) as double),3) COURCE1_WIN12_RATE1,
  ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT+WIN3_COUNT)/ RACE_COUNT) as double),3) COURCE1_WIN123_RATE1
  from
  (
select
  RACEDATE,
  PLAYERID,
  SUM(RACE_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACE_COUNT,
  SUM(WIN1_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN1_COUNT,
  SUM(WIN2_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN2_COUNT,
  SUM(WIN3_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN3_COUNT
from(
    select
      RACEDATE,
      PLAYERID1 as PLAYERID ,
      COUNT(1) as RACE_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 1, 1) = 1 THEN 1
          ELSE 0
        END
      ) as WIN1_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 3, 1) = 1 THEN 1
          ELSE 0
        END
      ) as WIN2_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 5, 1) = 1 THEN 1
          ELSE 0
        END
      ) as WIN3_COUNT
    from
      TRAINING_SILVER
    GROUP BY
      RACEDATE,
      PLAYERID
  )
GROUP BY
  RACEDATE,
  PLAYERID,
  RACE_COUNT,
  WIN1_COUNT,
  WIN2_COUNT,
  WIN3_COUNT
  )
  ) as T1 WHERE RACEDATE IS NOT NULL
  """)

display(df_COURCE1_WINRATE)

# COMMAND ----------

#2.番組表データから、コース2の１着率、２着率、３着率の実績テーブルを作成する。
df_COURCE2_WINRATE = sql(
"""
select * from (
select 
  RACEDATE as RACEDATE_LATEST,
  --LEADで日付をずらし前日までの集計でトレーニングができるようにする
  LEAD(RACEDATE) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACEDATE,
  PLAYERID as PLAYERID2,
  CAST(RACE_COUNT as double) as COURCE2_RACE_COUNT2,
  ROUND(CAST(( WIN1_COUNT / RACE_COUNT) as double),3) COURCE2_WIN1_RATE2,
  ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT)/ RACE_COUNT) as double),3) COURCE2_WIN12_RATE2,
  ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT+WIN3_COUNT)/ RACE_COUNT) as double),3) COURCE2_WIN123_RATE2
  from
  (
select
  RACEDATE,
  PLAYERID,
  SUM(RACE_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACE_COUNT,
  SUM(WIN1_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN1_COUNT,
  SUM(WIN2_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN2_COUNT,
  SUM(WIN3_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN3_COUNT
from(
    select
      RACEDATE,
      PLAYERID2 as PLAYERID ,
      COUNT(1) as RACE_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 1, 1) = 2 THEN 1
          ELSE 0
        END
      ) as WIN1_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 3, 1) = 2 THEN 1
          ELSE 0
        END
      ) as WIN2_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 5, 1) = 2 THEN 1
          ELSE 0
        END
      ) as WIN3_COUNT
    from
      TRAINING_SILVER
    GROUP BY
      RACEDATE,
      PLAYERID
  )
GROUP BY
  RACEDATE,
  PLAYERID,
  RACE_COUNT,
  WIN1_COUNT,
  WIN2_COUNT,
  WIN3_COUNT
  )
  ) as T1 WHERE RACEDATE IS NOT NULL
  """)

display(df_COURCE2_WINRATE)

# COMMAND ----------

#3.番組表データから、コース3の１着率、２着率、３着率の実績テーブルを作成する。
df_COURCE3_WINRATE = sql(
"""
select * from (
select 
  RACEDATE as RACEDATE_LATEST,
  --LEADで日付をずらし前日までの集計でトレーニングができるようにする
  LEAD(RACEDATE) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACEDATE,
  PLAYERID as PLAYERID3,
  CAST(RACE_COUNT as double) as COURCE3_RACE_COUNT3,
  ROUND(CAST(( WIN1_COUNT / RACE_COUNT) as double),3) COURCE3_WIN1_RATE3,
  ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT)/ RACE_COUNT) as double),3) COURCE3_WIN12_RATE3,
  ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT+WIN3_COUNT)/ RACE_COUNT) as double),3) COURCE3_WIN123_RATE3
  from
  (
select
  RACEDATE,
  PLAYERID,
  SUM(RACE_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACE_COUNT,
  SUM(WIN1_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN1_COUNT,
  SUM(WIN2_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN2_COUNT,
  SUM(WIN3_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN3_COUNT
from(
    select
      RACEDATE,
      PLAYERID3 as PLAYERID ,
      COUNT(1) as RACE_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 1, 1) = 3 THEN 1
          ELSE 0
        END
      ) as WIN1_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 3, 1) = 3 THEN 1
          ELSE 0
        END
      ) as WIN2_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 5, 1) = 3 THEN 1
          ELSE 0
        END
      ) as WIN3_COUNT
    from
      TRAINING_SILVER
    GROUP BY
      RACEDATE,
      PLAYERID
  )
GROUP BY
  RACEDATE,
  PLAYERID,
  RACE_COUNT,
  WIN1_COUNT,
  WIN2_COUNT,
  WIN3_COUNT
  )
  ) as T1 WHERE RACEDATE IS NOT NULL
  """)

display(df_COURCE3_WINRATE)

# COMMAND ----------

#4.番組表データから、コース4の１着率、２着率、３着率の実績テーブルを作成する。
df_COURCE4_WINRATE = sql(
"""
select * from (
select 
  RACEDATE as RACEDATE_LATEST,
  --LEADで日付をずらし前日までの集計でトレーニングができるようにする
  LEAD(RACEDATE) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACEDATE,
  PLAYERID as PLAYERID4,
  CAST(RACE_COUNT as double) as COURCE4_RACE_COUNT4,
  ROUND(CAST(( WIN1_COUNT / RACE_COUNT) as double),3) COURCE4_WIN1_RATE4,
  ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT)/ RACE_COUNT) as double),3) COURCE4_WIN12_RATE4,
  ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT+WIN3_COUNT)/ RACE_COUNT) as double),3) COURCE4_WIN123_RATE4
  from
  (
select
  RACEDATE,
  PLAYERID,
  SUM(RACE_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACE_COUNT,
  SUM(WIN1_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN1_COUNT,
  SUM(WIN2_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN2_COUNT,
  SUM(WIN3_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN3_COUNT
from(
    select
      RACEDATE,
      PLAYERID4 as PLAYERID ,
      COUNT(1) as RACE_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 1, 1) = 4 THEN 1
          ELSE 0
        END
      ) as WIN1_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 3, 1) = 4 THEN 1
          ELSE 0
        END
      ) as WIN2_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 5, 1) = 4 THEN 1
          ELSE 0
        END
      ) as WIN3_COUNT
    from
      TRAINING_SILVER
    GROUP BY
      RACEDATE,
      PLAYERID
  )
GROUP BY
  RACEDATE,
  PLAYERID,
  RACE_COUNT,
  WIN1_COUNT,
  WIN2_COUNT,
  WIN3_COUNT
  )
  ) as T1 WHERE RACEDATE IS NOT NULL
  """)

display(df_COURCE4_WINRATE)

# COMMAND ----------

#5.番組表データから、コース5の１着率、２着率、３着率の実績テーブルを作成する。
df_COURCE5_WINRATE = sql(
"""
select * from (
select 
  RACEDATE as RACEDATE_LATEST,
  --LEADで日付をずらし前日までの集計でトレーニングができるようにする
  LEAD(RACEDATE) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACEDATE,
  PLAYERID as PLAYERID5,
  CAST(RACE_COUNT as double) as COURCE5_RACE_COUNT5,
  ROUND(CAST(( WIN1_COUNT / RACE_COUNT) as double),3) COURCE5_WIN1_RATE5,
  ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT)/ RACE_COUNT) as double),3) COURCE5_WIN12_RATE5,
  ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT+WIN3_COUNT)/ RACE_COUNT) as double),3) COURCE5_WIN123_RATE5
  from
  (
select
  RACEDATE,
  PLAYERID,
  SUM(RACE_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACE_COUNT,
  SUM(WIN1_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN1_COUNT,
  SUM(WIN2_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN2_COUNT,
  SUM(WIN3_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN3_COUNT
from(
    select
      RACEDATE,
      PLAYERID5 as PLAYERID ,
      COUNT(1) as RACE_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 1, 1) = 5 THEN 1
          ELSE 0
        END
      ) as WIN1_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 3, 1) = 5 THEN 1
          ELSE 0
        END
      ) as WIN2_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 5, 1) = 5 THEN 1
          ELSE 0
        END
      ) as WIN3_COUNT
    from
      TRAINING_SILVER
    GROUP BY
      RACEDATE,
      PLAYERID
  )
GROUP BY
  RACEDATE,
  PLAYERID,
  RACE_COUNT,
  WIN1_COUNT,
  WIN2_COUNT,
  WIN3_COUNT
  )
  ) as T1 WHERE RACEDATE IS NOT NULL
  """)

display(df_COURCE5_WINRATE)

# COMMAND ----------

#6.番組表データから、コース6の１着率、２着率、３着率の実績テーブルを作成する。
df_COURCE6_WINRATE = sql(
"""
select * from (
select 
  RACEDATE as RACEDATE_LATEST,
  --LEADで日付をずらし前日までの集計でトレーニングができるようにする
  LEAD(RACEDATE) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACEDATE,
  PLAYERID as PLAYERID6,
  CAST(RACE_COUNT as double) as COURCE6_RACE_COUNT6,
  ROUND(CAST(( WIN1_COUNT / RACE_COUNT) as double),3) COURCE6_WIN1_RATE6,
  ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT)/ RACE_COUNT) as double),3) COURCE6_WIN12_RATE6,
  ROUND(CAST(( (WIN1_COUNT+WIN2_COUNT+WIN3_COUNT)/ RACE_COUNT) as double),3) COURCE6_WIN123_RATE6
  from
  (
select
  RACEDATE,
  PLAYERID,
  SUM(RACE_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as RACE_COUNT,
  SUM(WIN1_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN1_COUNT,
  SUM(WIN2_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN2_COUNT,
  SUM(WIN3_COUNT) OVER (PARTITION BY PLAYERID ORDER BY RACEDATE ) as WIN3_COUNT
from(
    select
      RACEDATE,
      PLAYERID6 as PLAYERID ,
      COUNT(1) as RACE_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 1, 1) = 6 THEN 1
          ELSE 0
        END
      ) as WIN1_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 3, 1) = 6 THEN 1
          ELSE 0
        END
      ) as WIN2_COUNT,
      SUM(
        CASE
          WHEN SUBSTR(RENTAN3, 5, 1) = 6 THEN 1
          ELSE 0
        END
      ) as WIN3_COUNT
    from
      TRAINING_SILVER
    GROUP BY
      RACEDATE,
      PLAYERID
  )
GROUP BY
  RACEDATE,
  PLAYERID,
  RACE_COUNT,
  WIN1_COUNT,
  WIN2_COUNT,
  WIN3_COUNT
  )
  ) as T1 WHERE RACEDATE IS NOT NULL
  """)

display(df_COURCE6_WINRATE)

# COMMAND ----------

# DBTITLE 1,データチェック用
# MAGIC %sql
# MAGIC select 
# MAGIC racedate,place,race,rentan3 
# MAGIC PLAYERID1,
# MAGIC PLAYERID2,
# MAGIC PLAYERID3,
# MAGIC PLAYERID4,
# MAGIC PLAYERID5,
# MAGIC PLAYERID6,
# MAGIC rentan3
# MAGIC from training_silver 
# MAGIC where racedate between 20220901 and 20220905 
# MAGIC and place = '浜名湖'
# MAGIC and 
# MAGIC (
# MAGIC playerid1 = 'p2014' or 
# MAGIC playerid2 = 'p2014' or 
# MAGIC playerid3 = 'p2014' or 
# MAGIC playerid4 = 'p2014' or 
# MAGIC playerid5 = 'p2014' or 
# MAGIC playerid6 = 'p2014' 
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from bangumi where playerid1 = 'p3556' order by racedate;

# COMMAND ----------

# MAGIC %md ### フィーチャーストアライブラリを使用して新しい特徴量テーブルを作成 

# COMMAND ----------

from databricks import feature_store
from pyspark.sql.functions import *
from pyspark.sql.types import FloatType, IntegerType, StringType
from pytz import timezone

# COMMAND ----------

# MAGIC %md 次に、create_table APIを使用してFeature Storeクライアントのインスタンスを作成します。</br>
# MAGIC https://docs.databricks.com/applications/machine-learning/feature-store/feature-tables.html

# COMMAND ----------

fs = feature_store.FeatureStoreClient()

# COMMAND ----------

# MAGIC %md
# MAGIC create_table API (v0.3.6 以降) または create_feature_table API (v0.3.5 以降) を用いてスキーマおよび一意な ID キーを定義します。</br>
# MAGIC オプション引数 df (0.3.6 以上) または features_df (0.3.5 以下) を渡すと、そのデータを Feature Store にも書き出します。

# COMMAND ----------

# MAGIC %python
# MAGIC # 最初に既存のフィーチャーストアテーブルを削除しておきます。
# MAGIC try: 
# MAGIC   fs.drop_table("FS_COURCE1_WINRATE") 
# MAGIC except:
# MAGIC   print('table does not exists')
# MAGIC
# MAGIC try: 
# MAGIC   fs.drop_table("FS_COURCE2_WINRATE")
# MAGIC except:
# MAGIC   print('table does not exists')
# MAGIC
# MAGIC try:
# MAGIC   fs.drop_table("FS_COURCE3_WINRATE")
# MAGIC except:
# MAGIC   print('table does not exists')
# MAGIC
# MAGIC try:
# MAGIC   fs.drop_table("FS_COURCE4_WINRATE")
# MAGIC except:
# MAGIC   print('table does not exists')
# MAGIC
# MAGIC try:
# MAGIC   fs.drop_table("FS_COURCE5_WINRATE")
# MAGIC except:
# MAGIC   print('table does not exists')
# MAGIC
# MAGIC try:
# MAGIC   fs.drop_table("FS_COURCE6_WINRATE")
# MAGIC except:
# MAGIC   print('table does not exists')

# COMMAND ----------

# MAGIC %md フィーチャーテーブルを作成する

# COMMAND ----------

# このセルは、Feature Storeクライアントv0.3.6で導入されたAPIを使用しています。

fs.create_table(
    name="FS_COURCE1_WINRATE",
    primary_keys=["RACEDATE", "PLAYERID1"],#一意キーの指定
    df=df_COURCE1_WINRATE,#このデータフレームをフィーチャーストアに書き出す
    description="全国での１コースでの勝率",
)


# COMMAND ----------

fs.create_table(
    name="FS_COURCE2_WINRATE",
    primary_keys=["RACEDATE", "PLAYERID2"],#一意キーの指定
    df=df_COURCE2_WINRATE,#このデータフレームをフィーチャーストアに書き出す
    description="全国での2コースでの勝率",
)


# COMMAND ----------

fs.create_table(
    name="FS_COURCE3_WINRATE",
    primary_keys=["RACEDATE", "PLAYERID3"],#一意キーの指定
    df=df_COURCE3_WINRATE,#このデータフレームをフィーチャーストアに書き出す
    description="全国での3コースでの勝率",
)


# COMMAND ----------

fs.create_table(
    name="FS_COURCE4_WINRATE",
    primary_keys=["RACEDATE", "PLAYERID4"],#一意キーの指定
    df=df_COURCE4_WINRATE,#このデータフレームをフィーチャーストアに書き出す
    description="全国での4コースでの勝率",
)


# COMMAND ----------

fs.create_table(
    name="FS_COURCE5_WINRATE",
    primary_keys=["RACEDATE", "PLAYERID5"],#一意キーの指定
    df=df_COURCE5_WINRATE,#このデータフレームをフィーチャーストアに書き出す
    description="全国での5コースでの勝率",
)


# COMMAND ----------


fs.create_table(
    name="FS_COURCE6_WINRATE",
    primary_keys=["RACEDATE", "PLAYERID6"],#一意キーの指定
    df=df_COURCE6_WINRATE,#このデータフレームをフィーチャーストアに書き出す
    description="全国での6コースでの勝率",
)

# COMMAND ----------

# MAGIC %md ### トレーニング用データの作成

# COMMAND ----------

from pyspark.sql import *
from pyspark.sql.functions import current_timestamp
from pyspark.sql.types import IntegerType
import math
from datetime import timedelta
import mlflow.pyfunc

# COMMAND ----------

# 事前に定義しているシルバーテーブルを使用
silver_data = sql("select * from TRAINING_SILVER")

# COMMAND ----------

display(silver_data)

# COMMAND ----------

from databricks.feature_store import FeatureLookup
import mlflow

cource1_features_table = "FS_COURCE1_WINRATE"
cource2_features_table = "FS_COURCE2_WINRATE"
cource3_features_table = "FS_COURCE3_WINRATE"
cource4_features_table = "FS_COURCE4_WINRATE"
cource5_features_table = "FS_COURCE5_WINRATE"
cource6_features_table = "FS_COURCE6_WINRATE"

#https://docs.databricks.com/dev-tools/api/python/latest/feature-store/databricks.feature_store.entities.feature_lookup.html
# Feature Store APIを使用して特徴量を取得
cource1_features = [
   FeatureLookup( 
     table_name = cource1_features_table,
     # フィーチャーストアのカラムを指定
     feature_names = ["COURCE1_RACE_COUNT1","COURCE1_WIN1_RATE1","COURCE1_WIN12_RATE1","COURCE1_WIN123_RATE1"],
     # ローデータのカラムを指定（フィーチャストアのプライマリキーと一致させる）
     lookup_key = ["RACEDATE", "PLAYERID1"],
   ),
]

cource2_features = [
   FeatureLookup( 
     table_name = cource2_features_table,
     # フィーチャーストアのカラムを指定
     feature_names = ["COURCE2_RACE_COUNT2","COURCE2_WIN1_RATE2","COURCE2_WIN12_RATE2","COURCE2_WIN123_RATE2"],
     # ローデータのカラムを指定（フィーチャストアのプライマリキーと一致させる）
     lookup_key = ["RACEDATE", "PLAYERID2"],
   ),
]

cource3_features = [
   FeatureLookup( 
     table_name = cource3_features_table,
     # フィーチャーストアのカラムを指定
     feature_names = ["COURCE3_RACE_COUNT3","COURCE3_WIN1_RATE3","COURCE3_WIN12_RATE3","COURCE3_WIN123_RATE3"],
     # ローデータのカラムを指定（フィーチャストアのプライマリキーと一致させる）
     lookup_key = ["RACEDATE", "PLAYERID3"],
   ),
]

cource4_features = [
   FeatureLookup( 
     table_name = cource4_features_table,
     # フィーチャーストアのカラムを指定
     feature_names = ["COURCE4_RACE_COUNT4","COURCE4_WIN1_RATE4","COURCE4_WIN12_RATE4","COURCE4_WIN123_RATE4"],
     # ローデータのカラムを指定（フィーチャストアのプライマリキーと一致させる）
     lookup_key = ["RACEDATE", "PLAYERID4"],
   ),
]

cource5_features = [
   FeatureLookup( 
     table_name = cource5_features_table,
     # フィーチャーストアのカラムを指定
     feature_names = ["COURCE5_RACE_COUNT5","COURCE5_WIN1_RATE5","COURCE5_WIN12_RATE5","COURCE5_WIN123_RATE5"],
     # ローデータのカラムを指定（フィーチャストアのプライマリキーと一致させる）
     lookup_key = ["RACEDATE", "PLAYERID5"],
   ),
]

cource6_features = [
   FeatureLookup( 
     table_name = cource6_features_table,
     # フィーチャーストアのカラムを指定
     feature_names = ["COURCE6_RACE_COUNT6","COURCE6_WIN1_RATE6","COURCE6_WIN12_RATE6","COURCE6_WIN123_RATE6"],
     # ローデータのカラムを指定（フィーチャストアのプライマリキーと一致させる）
     lookup_key = ["RACEDATE", "PLAYERID6"],
   ),
]

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

# Case sensitive name
name = "/Users/mitsuhiro.itagaki@databricks.com/kyotei_feature1"

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

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run() 

##################################################################
# 入力された生データに両特徴テーブルから対応する特徴を合成した学習セットを"作成する。
##################################################################
training_set = fs.create_training_set(
  silver_data,
  feature_lookups = cource1_features + cource2_features + cource3_features + cource4_features + cource5_features + cource6_features,
  label = "RENFUKU3",
  # 以下を除外してトレーニングを行わないようにします。
  exclude_columns = [""]
)

# モデルを学習するために、TrainingSetをdataframeにロードし、sklearnに渡します。
training_df = training_set.load_df()

# COMMAND ----------

#null値を０で埋める
training_df = training_df.fillna(0)

# COMMAND ----------

# トレーニング用のデータフレームを表示し、生の入力データと特徴量テーブルの両方を含んでいることに注目しましょう
display(training_df)

# COMMAND ----------

training_df.createOrReplaceTempView('training_tmp1')

# COMMAND ----------

# DBTITLE 1,トレーニング用の特徴量を追加したテーブル1
# MAGIC %sql
# MAGIC use kyotei_db;
# MAGIC create or replace table training_feature1 as
# MAGIC select 
# MAGIC distinct 
# MAGIC RACEDATE -- 特徴量にはしない
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
# MAGIC ,COURCE1_RACE_COUNT1
# MAGIC ,COURCE1_WIN1_RATE1
# MAGIC ,COURCE1_WIN12_RATE1
# MAGIC ,COURCE1_WIN123_RATE1
# MAGIC
# MAGIC ,COURCE2_RACE_COUNT2
# MAGIC ,COURCE2_WIN1_RATE2
# MAGIC ,COURCE2_WIN12_RATE2
# MAGIC ,COURCE2_WIN123_RATE2
# MAGIC
# MAGIC ,COURCE3_RACE_COUNT3
# MAGIC ,COURCE3_WIN1_RATE3
# MAGIC ,COURCE3_WIN12_RATE3
# MAGIC ,COURCE3_WIN123_RATE3
# MAGIC
# MAGIC ,COURCE4_RACE_COUNT4
# MAGIC ,COURCE4_WIN1_RATE4
# MAGIC ,COURCE4_WIN12_RATE4
# MAGIC ,COURCE4_WIN123_RATE4
# MAGIC
# MAGIC ,COURCE5_RACE_COUNT5
# MAGIC ,COURCE5_WIN1_RATE5
# MAGIC ,COURCE5_WIN12_RATE5
# MAGIC ,COURCE5_WIN123_RATE5
# MAGIC
# MAGIC ,COURCE6_RACE_COUNT6
# MAGIC ,COURCE6_WIN1_RATE6
# MAGIC ,COURCE6_WIN12_RATE6
# MAGIC ,COURCE6_WIN123_RATE6
# MAGIC
# MAGIC ,TAN
# MAGIC ,TANK
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
# MAGIC from training_tmp1
# MAGIC ;
