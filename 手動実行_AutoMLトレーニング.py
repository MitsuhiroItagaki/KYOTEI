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

dbutils.widgets.dropdown("EXPNO", "00", [str(x) for x in exp_no],"実験NO")

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

dbutils.widgets.dropdown("MODE", "train", [str(x) for x in mode],"学習/予測")

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
  print("予測の実行なら別のnotebookで実行することにしたので終了")
  exit()

# トレーニングなら一旦過去の実験があれば消して
if MODE == 'train':
  print("トレーニング実行")
  mlflow.delete_experiment(experiment.experiment_id)

  # 再度作り直す
  experiment == mlflow.set_experiment(name)
  # Get Experiment Details
  print("Experiment_id: {}".format(experiment.experiment_id))
  print("Artifact Location: {}".format(experiment.artifact_location))
  print("Tags: {}".format(experiment.tags))
  print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
  print("Creation timestamp: {}".format(experiment.creation_time))

# COMMAND ----------

# MAGIC %md
# MAGIC ### データロード

# COMMAND ----------

from sklearn.model_selection import train_test_split
import random

# データをロード
sql_text = "select * from training_" + RENFUKU + "f"
print(sql_text)
input_data = sql(sql_text).toPandas()

# トレーニングデータ(80%)とテストデータ(20%)に分割
train, test_tmp = train_test_split(input_data,test_size=0.2, random_state=int(random.random()*100000))
# トレーニングデータにAUTOMLと同じ識別用データを追加
train['_automl_split_col_0000'] = 'train'

# テストデータをさらにテスト用(10%)/検証用(10%)に分割
test,val = train_test_split(test_tmp,test_size=0.5, random_state=int(random.random()*100000))
# テスト用/検証用データにAUTOMLと同じ識別用データを追加
test['_automl_split_col_0000'] = 'test'
val['_automl_split_col_0000'] = 'val'

#　データを連結
df_loaded = pd.concat([train,test,val], axis=0)
display(df_loaded)

# COMMAND ----------

print(train.count)
print(val.count)
print(test.count)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector

supported_cols = [
    "COURCE1_WIN123_RATE1",
    "MOTORWIN2RATE6",
    "WIN2RATE6",
    "COURCE4_LOCAL_WIN123_RATE4",
    "MOTORWIN2RATE3",
    "ST_AVG3",
    "MOTORWIN2RATE2",
    "WIN1RATE6",
    "WIN1RATE3",
    "WIN2RATE3",
    "CLASS4",
    "WIN1RATE5",
    "CLASS2",
    "MOTORWIN2RATE1",
    "COURCE6_LOCAL_WIN123_RATE6",
    "WIN2RATE1",
    "CLASS3",
    "WIN1RATE1",
    "COURCE2_LOCAL_WIN123_RATE2",
    "WIN1RATE2",
    "WIN1RATE4",
    "COURCE5_LOCAL_WIN123_RATE5",
    "ST_AVG2",
    "MOTORWIN2RATE4",
    "WIN2RATE2",
    "ST_AVG4",
    "COURCE2_WIN123_RATE2",
    "COURCE6_WIN123_RATE6",
    "CLASS1",
    "ST_AVG5",
    "CLASS6",
    "WIN2RATE4",
    "COURCE3_WIN123_RATE3",
    "CLASS5",
    "ST_AVG6",
    "COURCE3_LOCAL_WIN123_RATE3",
    "WIN2RATE5",
    "COURCE5_WIN123_RATE5",
    "COURCE1_LOCAL_WIN123_RATE1",
    "COURCE4_WIN123_RATE4",
    "MOTORWIN2RATE5",
    "ST_AVG1",
]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(
    (
        "impute_mean",
        SimpleImputer(),
        [
            "CLASS1",
            "CLASS2",
            "CLASS3",
            "CLASS4",
            "CLASS5",
            "CLASS6",
            "COURCE1_LOCAL_WIN123_RATE1",
            "COURCE1_WIN123_RATE1",
            "COURCE2_LOCAL_WIN123_RATE2",
            "COURCE2_WIN123_RATE2",
            "COURCE3_LOCAL_WIN123_RATE3",
            "COURCE3_WIN123_RATE3",
            "COURCE4_LOCAL_WIN123_RATE4",
            "COURCE4_WIN123_RATE4",
            "COURCE5_LOCAL_WIN123_RATE5",
            "COURCE5_WIN123_RATE5",
            "COURCE6_LOCAL_WIN123_RATE6",
            "COURCE6_WIN123_RATE6",
            "MOTORWIN2RATE1",
            "MOTORWIN2RATE2",
            "MOTORWIN2RATE3",
            "MOTORWIN2RATE4",
            "MOTORWIN2RATE5",
            "MOTORWIN2RATE6",
            "ST_AVG1",
            "ST_AVG2",
            "ST_AVG3",
            "ST_AVG4",
            "ST_AVG5",
            "ST_AVG6",
            "WIN1RATE1",
            "WIN1RATE2",
            "WIN1RATE3",
            "WIN1RATE4",
            "WIN1RATE5",
            "WIN1RATE6",
            "WIN2RATE1",
            "WIN2RATE2",
            "WIN2RATE3",
            "WIN2RATE4",
            "WIN2RATE5",
            "WIN2RATE6",
        ],
    )
)

numerical_pipeline = Pipeline(
    steps=[
        (
            "converter",
            FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce")),
        ),
        ("imputers", ColumnTransformer(num_imputers)),
        ("standardizer", StandardScaler()),
    ]
)

numerical_transformers = [
    (
        "numerical",
        numerical_pipeline,
        [
            "COURCE1_WIN123_RATE1",
            "MOTORWIN2RATE6",
            "WIN2RATE6",
            "COURCE4_LOCAL_WIN123_RATE4",
            "MOTORWIN2RATE3",
            "ST_AVG3",
            "MOTORWIN2RATE2",
            "WIN1RATE6",
            "WIN1RATE3",
            "WIN2RATE3",
            "CLASS4",
            "WIN1RATE5",
            "CLASS2",
            "MOTORWIN2RATE1",
            "COURCE6_LOCAL_WIN123_RATE6",
            "WIN2RATE1",
            "CLASS3",
            "WIN1RATE1",
            "COURCE2_LOCAL_WIN123_RATE2",
            "WIN1RATE2",
            "WIN1RATE4",
            "COURCE5_LOCAL_WIN123_RATE5",
            "ST_AVG2",
            "MOTORWIN2RATE4",
            "WIN2RATE2",
            "ST_AVG4",
            "COURCE2_WIN123_RATE2",
            "COURCE6_WIN123_RATE6",
            "CLASS1",
            "ST_AVG5",
            "CLASS6",
            "WIN2RATE4",
            "COURCE3_WIN123_RATE3",
            "CLASS5",
            "ST_AVG6",
            "COURCE3_LOCAL_WIN123_RATE3",
            "WIN2RATE5",
            "COURCE5_WIN123_RATE5",
            "COURCE1_LOCAL_WIN123_RATE1",
            "COURCE4_WIN123_RATE4",
            "MOTORWIN2RATE5",
            "ST_AVG1",
        ],
    )
]

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = numerical_transformers

preprocessor = ColumnTransformer(
    transformers, remainder="passthrough", sparse_threshold=0
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC The input data is split by AutoML into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)
# MAGIC
# MAGIC `_automl_split_col_0000` contains the information of which set a given row belongs to.
# MAGIC We use this column to split the dataset into the above 3 sets. 
# MAGIC The column should not be used for training so it is dropped after split is done.

# COMMAND ----------

# AutoML completed train - validation - test split internally and used _automl_split_col_0000 to specify the set
split_train_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "train"]
split_val_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "val"]
split_test_df = df_loaded.loc[df_loaded._automl_split_col_0000 == "test"]

# Separate target column from features and drop _automl_split_col_0000
X_train = split_train_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_train = split_train_df[target_col]

X_val = split_val_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_val = split_val_df[target_col]

X_test = split_test_df.drop([target_col, "_automl_split_col_0000"], axis=1)
y_test = split_test_df[target_col]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/4317267877446177)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

import lightgbm
from lightgbm import LGBMClassifier

help(LGBMClassifier)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the objective function
# MAGIC The objective function used to find optimal hyperparameters. By default, this notebook only runs
# MAGIC this function once (`max_evals=1` in the `hyperopt.fmin` invocation) with fixed hyperparameters, but
# MAGIC hyperparameters can be tuned by modifying `space`, defined below. `hyperopt.fmin` will then use this
# MAGIC function's return value to search the space to minimize the loss.

# COMMAND ----------

import mlflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

from hyperopt import hp, tpe, fmin, STATUS_OK, Trials

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
mlflow.sklearn.autolog(disable=True)
pipeline_val = Pipeline(
    [
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
    ]
)
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)


def objective(params):
     #with mlflow.start_run(experiment_id="4317267877446177") as mlflow_run:
     with mlflow.start_run(experiment_id=experiment.experiment_id) as mlflow_run:
        lgbmc_classifier = LGBMClassifier(**params)

        model = Pipeline(
            [
                ("column_selector", col_selector),
                ("preprocessor", preprocessor),
                ("classifier", lgbmc_classifier),
            ]
        )

        # Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(log_input_examples=True, silent=True)

        model.fit(
            X_train,
            y_train,
            classifier__callbacks=[
                lightgbm.early_stopping(5),
                lightgbm.log_evaluation(0),
            ],
            classifier__eval_set=[(X_val_processed, y_val)],
        )

        # Log metrics for the training set
        mlflow_model = Model()
        pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
        pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
        training_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_train.assign(**{str(target_col): y_train}),
            targets=target_col,
            model_type="classifier",
            evaluator_config={
                "log_model_explainability": False,
                "metric_prefix": "training_",
                "pos_label": 1,
            },
        )
        lgbmc_training_metrics = training_eval_result.metrics
        # Log metrics for the validation set
        val_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_val.assign(**{str(target_col): y_val}),
            targets=target_col,
            model_type="classifier",
            evaluator_config={
                "log_model_explainability": False,
                "metric_prefix": "val_",
                "pos_label": 1,
            },
        )
        lgbmc_val_metrics = val_eval_result.metrics
        # Log metrics for the test set
        test_eval_result = mlflow.evaluate(
            model=pyfunc_model,
            data=X_test.assign(**{str(target_col): y_test}),
            targets=target_col,
            model_type="classifier",
            evaluator_config={
                "log_model_explainability": False,
                "metric_prefix": "test_",
                "pos_label": 1,
            },
        )
        lgbmc_test_metrics = test_eval_result.metrics

        loss = -lgbmc_val_metrics["val_precision_score"]

        # Truncate metric key names so they can be displayed together
        lgbmc_val_metrics = {
            k.replace("val_", ""): v for k, v in lgbmc_val_metrics.items()
        }
        lgbmc_test_metrics = {
            k.replace("test_", ""): v for k, v in lgbmc_test_metrics.items()
        }

        return {
            "loss": loss,
            "status": STATUS_OK,
            "val_metrics": lgbmc_val_metrics,
            "test_metrics": lgbmc_test_metrics,
            "model": model,
            "run": mlflow_run,
        }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure the hyperparameter search space
# MAGIC Configure the search space of parameters. Parameters below are all constant expressions but can be
# MAGIC modified to widen the search space. For example, when training a decision tree classifier, to allow
# MAGIC the maximum tree depth to be either 2 or 3, set the key of 'max_depth' to
# MAGIC `hp.choice('max_depth', [2, 3])`. Be sure to also increase `max_evals` in the `fmin` call below.
# MAGIC
# MAGIC See https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html
# MAGIC for more information on hyperparameter tuning as well as
# MAGIC http://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for documentation on supported
# MAGIC search expressions.
# MAGIC
# MAGIC For documentation on parameters used by the model in use, please see:
# MAGIC https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html
# MAGIC
# MAGIC NOTE: The above URL points to a stable version of the documentation corresponding to the last
# MAGIC released version of the package. The documentation may differ slightly for the package version
# MAGIC used by this notebook.

# COMMAND ----------

# MAGIC %md ### AutoMLで作成されたコードはパラメータのサーチスペースを修正するだけで再チューニング可能

# COMMAND ----------

# DBTITLE 0,AutoMLで作成されたコードはパラメータのサーチスペースを修正するだけで再チューニング可能
##################################################
#ハイパーパラメータの探索範囲設定(lightGBM)
#ベストモデルのパラメータを確認してから範囲を設定しました
##################################################
from hyperopt.pyll import scope
space = {
  "colsample_bytree":hp.uniform("colsample_bytree", 0,1),
  "lambda_l1":hp.uniform("lambda_l1",0.0,200.0),
  "lambda_l2":hp.uniform("lambda_l2",0.0,2000.0),
  "learning_rate":hp.uniform("learning_rate",0,1),
  "max_bin":scope.int(hp.quniform("max_bin",10, 2000, 1)),
  "max_depth":scope.int(hp.quniform('max_depth', 2, 40, 1)),
  "min_child_samples":scope.int(hp.quniform('min_child_samples', 10, 2000, 1)),
  "n_estimators":scope.int(hp.quniform('n_estimators', 20, 8000, 1)),
  "num_leaves":scope.int(hp.quniform('num_leaves', 4, 4000, 1)),
  "path_smooth":hp.uniform("path_smooth",2,400),
  "subsample":hp.uniform("subsample",0,1),
  #"random_state":7777777,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run trials
# MAGIC When widening the search space and training multiple models, switch to `SparkTrials` to parallelize
# MAGIC training on Spark:
# MAGIC ```
# MAGIC from hyperopt import SparkTrials
# MAGIC trials = SparkTrials()
# MAGIC ```
# MAGIC
# MAGIC NOTE: While `Trials` starts an MLFlow run for each set of hyperparameters, `SparkTrials` only starts
# MAGIC one top-level run; it will start a subrun for each set of hyperparameters.
# MAGIC
# MAGIC See http://hyperopt.github.io/hyperopt/scaleout/spark/ for more info.

# COMMAND ----------

print(experiment.experiment_id)

# COMMAND ----------

# MAGIC %md ### SparkTrials テストを Spark workerで分散実行することでハイパーパラメータチューニングを高速化します。

# COMMAND ----------

# DBTITLE 0,SparkTrials テストを Spark workerで分散実行することでハイパーパラメータチューニングを高速化します。
#trials = Trials()
from hyperopt import SparkTrials
trials = SparkTrials(parallelism=10)

#parallelism = 1の場合、Hyperoptは繰り返しハイパーパラメーター空間を探索するTree of Parzen Estimatorsのような適合アルゴリズムを完全に活用します。
#テストされたそれぞれの新規のハイパーパラメーターの設定は、前回の結果に基づいて選択されます。
#parallelismを1とmax_evalsの間に設定することで、スケーラビリティ(迅速に結果を得る)と適合性(時に優れた結果を得る)のトレードオフを選択することができます。
#望ましい選択はsqrt(max_evals)のように中間を取るというものです。

fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=300,  # Increase this when widening the hyperparameter search space.
    trials=trials,
)

best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
    pd.DataFrame(
        [best_result["val_metrics"], best_result["test_metrics"]],
        index=["validation", "test"],
    )
)

set_config(display="diagram")
model

# COMMAND ----------

# MAGIC %md ### ベストモデルのハイパーパラメータの取得

# COMMAND ----------

# DBTITLE 0,ベストモデルのハイパーパラメータの取得
# MAGIC %python
# MAGIC # ベストモデルのハイパーパラメータの取得
# MAGIC print(mlflow_run.data.params)

# COMMAND ----------

# MAGIC %md ### 再実行した実験結果の取得

# COMMAND ----------

# DBTITLE 0,再実行した実験結果の取得
# 実験IDの取得
experiment_id=mlflow_run.info.experiment_id
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
       (`metrics.val_true_positives` + `metrics.test_true_positives` + `metrics.val_false_positives` + `metrics.test_false_positives` ) >= 0.5
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
# Pandasデータフレームで表示する場合はレコードが選択されていないとcan not infer schema from empty datasetエラーになるので注意
display(out)

# COMMAND ----------

# MAGIC %md ###  アンサンブルするために全てのモデルの予測結果を集計する

# COMMAND ----------

# DBTITLE 0, 全てのモデルの予測結果を集計する
# 予測対象データの取得
sql_text = "select * from predict_" + RENFUKU + "f"
print(sql_text)
data_df = sql(sql_text).toPandas()

# 1モデルごとに予測して結果をループしながらテーブルに保存
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
    f"""
    select 
    '{EXPNO}' as expno,
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
    print("REC:",out.count())

    # 予測結果テーブルがなければ新規作成
    sql_text = 
    """
    create table if not exists " + kekka_table + " as select * from kekka limit 0
    """
    sql(sql_text)
    
    # 予測結果を保存
    sql("insert into " + kekka_table + " select * from kekka")


# COMMAND ----------

# 結果フラグをつけたデータでビュー作成
out.createOrReplaceTempView("kekka")
print("REC:",out.count())

# 予測結果テーブルがなければ新規作成
sql_text = (
"""
create table if not exists 
"""
+ kekka_table + 
"""
(
expno string,
racedate string,
place string,
race string,
rentan2 string,
renfuku2 string,
rentan2k string,
renfuku2k string,
rentan3 string,
rentan3k string,
renfuku3 string,
renfuku3k string,
kekka int
)
"""
)

print(sql_text)
sql(sql_text)
    
# 予測結果を保存
sql("insert into " + kekka_table + " select * from kekka")


# COMMAND ----------

# データ確認
out2 = sql(
f"""select 
expno,
racedate,
place,
race,
cast(rentan2 as String) as rentan2,
cast(renfuku2 as String) as renfuku2,
cast(rentan2k as String) as  rentan2k,
cast(renfuku2k as String) as renfuku2k,
cast(rentan3 as String) as rentan3,
cast(rentan3k as String) as rentan3k,
cast(renfuku3 as String) as renfuku3,
cast(renfuku3k as String) as renfuku3k,
kekka,
count(1) as cnt ,
max(current_timestamp()) as insert_time
from { kekka_table } group by 1,2,3,4,5,6,7,8,9,10,11,12,13
"""
)
# 一件もヒットしない場合はエラーになる
display(out2)

# COMMAND ----------

