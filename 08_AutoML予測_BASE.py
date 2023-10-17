# Databricks notebook source
# MAGIC %md
# MAGIC # LightGBM Classifier training
# MAGIC - This is an auto-generated notebook.
# MAGIC - To reproduce these results, attach this notebook to a cluster with runtime version **13.2.x-cpu-ml-scala2.12**, and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/4317267877446177).
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.

# COMMAND ----------

import mlflow
import databricks.automl_runtime

import os
import uuid
import shutil
import pandas as pd

target_col = "kekka"

# COMMAND ----------

# MAGIC %sql
# MAGIC use catalog main;
# MAGIC use kyotei_db;

# COMMAND ----------

#　実験回数
exp_no = sqlContext.sql(\
"         select '00' as race \
union all select '01' \
union all select '02' \
union all select '03' \
union all select '04' \
union all select '05' \
union all select '06' \
union all select '07' \
union all select '08' \
union all select '09' \
union all select '10' \
").rdd.map(lambda row : row[0]).collect()

dbutils.widgets.dropdown("EXPNO", "01", [str(x) for x in exp_no],"実験NO")

# COMMAND ----------

#　累積テーブル削除
isDrop = sqlContext.sql(\
"         select 'YES' as isDrop \
union all select 'NO' \
").rdd.map(lambda row : row[0]).collect()

dbutils.widgets.dropdown("DROPTABLE", "NO", [str(x) for x in isDrop],"累積テーブル削除")

# COMMAND ----------

#　累積テーブル削除
renfuku = sqlContext.sql(\
"         select '123' as renfuku \
union all select '124' \
").rdd.map(lambda row : row[0]).collect()

dbutils.widgets.dropdown("RENFUKU", "123", [str(x) for x in renfuku],"3連複")

# COMMAND ----------

#　学習/予測
mode = sqlContext.sql(\
"         select 'train' as isDrop \
union all select 'predict' \
").rdd.map(lambda row : row[0]).collect()

dbutils.widgets.dropdown("MODE", "predict", [str(x) for x in mode],"学習/予測")

# COMMAND ----------

#変数取得
EXPNO=dbutils.widgets.get("EXPNO")
DROPTABLE=dbutils.widgets.get("DROPTABLE")
RENFUKU=dbutils.widgets.get("RENFUKU")
MODE=dbutils.widgets.get("MODE")

print(EXPNO)
print(DROPTABLE)
print(RENFUKU)
print(MODE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 実験結果を格納するテーブルの設定

# COMMAND ----------

# 予測結果テーブルの削除 (2ショット目は削除しない)
kekka_table = "kekka_" + RENFUKU
if DROPTABLE == "YES":
  sql("drop table if exists " + kekka_table)
  print("drop:" , kekka_table)
else:
  print("keep:" , kekka_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 実験の設定

# COMMAND ----------

# Case sensitive name
name = "/Users/mitsuhiro.itagaki@databricks.com/model_exp" + RENFUKU + "_" + EXPNO
print(name)

# COMMAND ----------

# 実験　名称を指定
experiment = mlflow.set_experiment(name)

if MODE == 'predict':
  # Get Experiment Details
  print("予測実行")
  print("Experiment_id: {}".format(experiment.experiment_id))
  print("Artifact Location: {}".format(experiment.artifact_location))
  print("Tags: {}".format(experiment.tags))
  print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
  print("Creation timestamp: {}".format(experiment.creation_time))

# トレーニングなら別のnotebookで実行することにしたので終了
if MODE == 'train':
  print("学習の実行なら別のnotebookで実行することにしたので終了")
  exit()

# COMMAND ----------

# MAGIC %md ### 実験結果の取得

# COMMAND ----------

# DBTITLE 0,再実行した実験結果の取得
# 実験IDの取得
#experiment_id=mlflow_run.info.experiment_id
experiment_id=experiment.experiment_id
print("experiment_id=", experiment_id)

# APIでも最も精度の高いモデルの情報を取得できます。
query = ""
df = mlflow.search_runs(experiment_ids=experiment_id, filter_string=query, order_by=["metrics.val_precision_score DESC"],  max_results=20000)
sdf = spark.createDataFrame(df)

# 操作しやすいようにTEMPビューとして定義しておく
sdf.createOrReplaceTempView('all_runs_v')

# 表示
out = sql(
"""
select 
run_id,
`tags.mlflow.runName`,
(`metrics.val_true_positives` + `metrics.test_true_positives`) as win,
(`metrics.val_false_positives` + `metrics.test_false_positives` ) as lost,
(`metrics.val_true_positives` + `metrics.test_true_positives`) / 
       (`metrics.val_true_positives` + `metrics.test_true_positives` + `metrics.val_false_positives` + `metrics.test_false_positives` ) as hit_rate,
`metrics.test_precision_score`,
`metrics.val_precision_score`,
`metrics.test_true_positives`,
`metrics.test_false_positives`,
`metrics.val_true_positives`,
`metrics.val_false_positives`
from all_runs_v
where  (`metrics.val_true_positives` + `metrics.test_true_positives`) / 
       (`metrics.val_true_positives` + `metrics.test_true_positives` + `metrics.val_false_positives` + `metrics.test_false_positives` ) >= 0.50
order by 
 (`metrics.val_true_positives` + `metrics.test_true_positives`) / 
       (`metrics.val_true_positives` + `metrics.test_true_positives` + `metrics.val_false_positives` + `metrics.test_false_positives` ) desc
limit 200
""")

# 使用モデルのリストをテーブルに記録
out.write.mode("overwrite").saveAsTable("EXPNO"+EXPNO)

# モデルIDをリストに変換
medel_list = out.toPandas()['run_id'].to_list()

# 結果表示
display(out)

# COMMAND ----------

# MAGIC %md ###  全てのモデルの予測結果を集計する

# COMMAND ----------

# DBTITLE 0, 全てのモデルの予測結果を集計する
# 予測対象データの取得
sql_text = "select * from predict_" + RENFUKU + "f"
print(sql_text)
data_df = sql(sql_text).toPandas()

# 表示
index = 1
for model_id in medel_list:
    model_uri = f"runs:/{ model_id }/model"
    print(index , model_uri)
    index += 1
    
    #　 モデルのダウンロード
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    #  予測実行
    result = model.predict(data_df)
    # 予測結果をデータレームに格納
    result_df = pd.DataFrame(result, columns=["predict"])
    
    # 予測結果を予測データに追加
    dataset = pd.concat([data_df, result_df], axis=1)

    # Sparkデータフレームに変換
    dataset_sdf = spark.createDataFrame(dataset)
    
    # 予測結果をビュー化
    dataset_sdf.createOrReplaceTempView("prediction_tmp")
    
    out = sql(
    """
    select 
    racedate,
    place,
    race,
    --wide1,
    --wide2,
    --wide3,
    --wide1k,
    --wide2k,
    --wide3k,
    rentan2,
    renfuku2,
    rentan2k,
    renfuku2k,
    rentan3,
    rentan3k,
    renfuku3,
    renfuku3k,
    kekka
    from prediction_tmp
    where predict = 1
    order by racedate
    """
    )
    
    # 結果フラグをつけたデータでビュー作成
    out.createOrReplaceTempView("kekka")

    # 予測結果テーブルがなければ新規作成
    sql("create table if not exists " + kekka_table + " as select * from kekka limit 0")
    
    # 予測結果を保存
    sql("insert into " + kekka_table + " select * from kekka")


# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from kekka_123;

# COMMAND ----------

# データ確認
print(kekka_table)

out2 = sql(
f"""select 
racedate,
place,
race,
rentan2,
renfuku2,
rentan2k,
renfuku2k,
rentan3,
rentan3k,
renfuku3,
renfuku3k,
kekka,
count(1) as cnt  from { kekka_table } 
group by 
racedate,
place,
race,
rentan2,
renfuku2,
rentan2k,
renfuku2k,
rentan3,
rentan3k,
renfuku3,
renfuku3k,
kekka
"""
)
display(out2)

# COMMAND ----------

# MAGIC %sql
# MAGIC describe kekka_123

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from predict_123f;

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC racedate,
# MAGIC place,
# MAGIC race,
# MAGIC rentan2,
# MAGIC renfuku2,
# MAGIC rentan2k,
# MAGIC renfuku2k,
# MAGIC rentan3,
# MAGIC rentan3k,
# MAGIC renfuku3,
# MAGIC renfuku3k,
# MAGIC kekka,
# MAGIC count(1) as cnt  from kekka_123
# MAGIC group by 
# MAGIC racedate,
# MAGIC place,
# MAGIC race,
# MAGIC rentan2,
# MAGIC renfuku2,
# MAGIC rentan2k,
# MAGIC renfuku2k,
# MAGIC rentan3,
# MAGIC rentan3k,
# MAGIC renfuku3,
# MAGIC renfuku3k,
# MAGIC kekka

# COMMAND ----------

