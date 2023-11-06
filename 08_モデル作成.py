# Databricks notebook source
# MAGIC %md
# MAGIC # LightGBM Classifier training
# MAGIC - This is an auto-generated notebook.
# MAGIC - To reproduce these results, attach this notebook to a cluster with runtime version **14.1.x-cpu-ml-scala2.12**, and rerun it.
# MAGIC - Compare trials in the [MLflow experiment](#mlflow/experiments/3031282363653284).
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.

# COMMAND ----------

import mlflow
import databricks.automl_runtime

target_col = "BUY"
time_col = "RACEDATE"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

import mlflow
import os
import uuid
import shutil
import pandas as pd

# Create temp directory to download input data from MLflow
input_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(input_temp_dir)


# Download the artifact and read it into a pandas DataFrame
input_data_path = mlflow.artifacts.download_artifacts(run_id="d8c25de73dfe4ab0af4b341ed4d3c69c", artifact_path="data", dst_path=input_temp_dir)

df_loaded = pd.read_parquet(os.path.join(input_data_path, "training_data"))
# Delete the temp data
shutil.rmtree(input_temp_dir)

# Preview data
df_loaded.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = ["COURCE6_RACE_COUNT6", "L4", "WEIGHT2", "F4", "COURCE4_LOCAL_WIN123_RATE4", "COURCE1_LOCAL_RACE_COUNT1", "COURCE4_LOCAL_WIN1_RATE4", "WIN1RATE6", "COURCE6_LOCAL_WIN123_RATE6", "CLUB5", "WIN1RATE1", "COURCE1_WIN1_RATE1", "COURCE5_LOCAL_WIN123_RATE5", "COURCE5_LOCAL_WIN12_RATE5", "BOATWIN2RATE1", "COURCE3_LOCAL_WIN1_RATE3", "COURCE4_WIN12_RATE4", "AGE4", "LOCALWIN2RATE5", "CLUB4", "COURCE6_LOCAL_WIN12_RATE6", "BOATWIN2RATE4", "WIN2RATE4", "COURCE5_LOCAL_RACE_COUNT5", "COURCE3_LOCAL_WIN123_RATE3", "LOCALWIN2RATE6", "COURCE4_WIN123_RATE4", "MOTORWIN2RATE5", "COURCE3_RACE_COUNT3", "BOATWIN2RATE3", "COURCE1_WIN12_RATE1", "COURCE2_WIN1_RATE2", "F5", "WIN2RATE6", "COURCE6_LOCAL_RACE_COUNT6", "ST_AVG3", "L1", "MOTORWIN2RATE2", "BOATWIN2RATE5", "LOCALWIN1RATE2", "CLASS4", "CLASS2", "MOTORWIN2RATE1", "LOCALWIN1RATE6", "LOCALWIN1RATE3", "COURCE2_LOCAL_WIN123_RATE2", "F3", "WEIGHT5", "ST_AVG2", "LOCALWIN2RATE2", "L5", "WIN2RATE2", "AGE5", "ST_AVG5", "AGE2", "BOATWIN2RATE6", "COURCE3_LOCAL_RACE_COUNT3", "COURCE4_LOCAL_RACE_COUNT4", "COURCE3_LOCAL_WIN12_RATE3", "WIN2RATE5", "COURCE2_WIN12_RATE2", "COURCE5_WIN123_RATE5", "COURCE6_WIN12_RATE6", "ST_AVG1", "COURCE5_LOCAL_WIN1_RATE5", "AGE3", "COURCE6_WIN1_RATE6", "COURCE5_WIN1_RATE5", "COURCE2_RACE_COUNT2", "AGE1", "COURCE5_WIN12_RATE5", "WIN1RATE3", "WIN1RATE5", "COURCE1_RACE_COUNT1", "CLUB2", "WEIGHT6", "WIN2RATE1", "F1", "WIN1RATE2", "ST_AVG4", "BOATWIN2RATE2", "WEIGHT4", "COURCE2_WIN123_RATE2", "CLASS1", "LOCALWIN1RATE4", "CLASS6", "COURCE5_RACE_COUNT5", "L6", "ST_AVG6", "COURCE1_LOCAL_WIN123_RATE1", "COURCE3_WIN12_RATE3", "L3", "LOCALWIN2RATE3", "COURCE1_WIN123_RATE1", "LOCALWIN2RATE4", "MOTORWIN2RATE6", "MOTORWIN2RATE3", "L2", "CLUB3", "COURCE6_LOCAL_WIN1_RATE6", "WIN2RATE3", "LOCALWIN1RATE5", "RACE", "CLASS3", "F2", "WEIGHT1", "LOCALWIN1RATE1", "WIN1RATE4", "COURCE1_LOCAL_WIN12_RATE1", "COURCE2_LOCAL_RACE_COUNT2", "RACEDATE", "MOTORWIN2RATE4", "COURCE4_LOCAL_WIN12_RATE4", "CLUB6", "COURCE6_WIN123_RATE6", "COURCE2_LOCAL_WIN1_RATE2", "PLACE", "COURCE2_LOCAL_WIN12_RATE2", "COURCE3_WIN123_RATE3", "COURCE4_RACE_COUNT4", "WEIGHT3", "COURCE1_LOCAL_WIN1_RATE1", "AGE6", "F6", "CLASS5", "COURCE3_WIN1_RATE3", "COURCE4_WIN1_RATE4", "LOCALWIN2RATE1", "CLUB1"]
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

# MAGIC %md
# MAGIC ### Datetime Preprocessor
# MAGIC For each datetime column, extract relevant information from the date:
# MAGIC - Unix timestamp
# MAGIC - whether the date is a weekend
# MAGIC - whether the date is a holiday
# MAGIC
# MAGIC Additionally, extract extra information from columns with timestamps:
# MAGIC - hour of the day (one-hot encoded)
# MAGIC
# MAGIC For cyclic features, plot the values along a unit circle to encode temporal proximity:
# MAGIC - hour of the day
# MAGIC - hours since the beginning of the week
# MAGIC - hours since the beginning of the month
# MAGIC - hours since the beginning of the year

# COMMAND ----------

from pandas import Timestamp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from databricks.automl_runtime.sklearn import DatetimeImputer
from databricks.automl_runtime.sklearn import OneHotEncoder
from databricks.automl_runtime.sklearn import TimestampTransformer
from sklearn.preprocessing import StandardScaler

imputers = {
  "RACEDATE": DatetimeImputer(),
}

datetime_transformers = []

for col in ["RACEDATE"]:
    ohe_transformer = ColumnTransformer(
        [("ohe", OneHotEncoder(sparse=False, handle_unknown="indicator"), [TimestampTransformer.HOUR_COLUMN_INDEX])],
        remainder="passthrough")
    timestamp_preprocessor = Pipeline([
        (f"impute_{col}", imputers[col]),
        (f"transform_{col}", TimestampTransformer()),
        (f"onehot_encode_{col}", ohe_transformer),
        (f"standardize_{col}", StandardScaler()),
    ])
    datetime_transformers.append((f"timestamp_{col}", timestamp_preprocessor, [col]))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Boolean columns
# MAGIC For each column, impute missing values and then convert into ones and zeros.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


bool_imputers = []

bool_pipeline = Pipeline(steps=[
    ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
    ("imputers", ColumnTransformer(bool_imputers, remainder="passthrough")),
    ("onehot", SklearnOneHotEncoder(handle_unknown="ignore", drop="first")),
])

bool_transformers = [("boolean", bool_pipeline, ["L4", "L5", "L6", "L2", "L1", "L3"])]

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
num_imputers.append(("impute_mean", SimpleImputer(), ["AGE1", "AGE2", "AGE3", "AGE4", "AGE5", "AGE6", "BOATWIN2RATE1", "BOATWIN2RATE2", "BOATWIN2RATE3", "BOATWIN2RATE4", "BOATWIN2RATE5", "BOATWIN2RATE6", "CLASS1", "CLASS2", "CLASS3", "CLASS4", "CLASS5", "CLASS6", "COURCE1_LOCAL_RACE_COUNT1", "COURCE1_LOCAL_WIN123_RATE1", "COURCE1_LOCAL_WIN12_RATE1", "COURCE1_LOCAL_WIN1_RATE1", "COURCE1_RACE_COUNT1", "COURCE1_WIN123_RATE1", "COURCE1_WIN12_RATE1", "COURCE1_WIN1_RATE1", "COURCE2_LOCAL_RACE_COUNT2", "COURCE2_LOCAL_WIN123_RATE2", "COURCE2_LOCAL_WIN12_RATE2", "COURCE2_LOCAL_WIN1_RATE2", "COURCE2_RACE_COUNT2", "COURCE2_WIN123_RATE2", "COURCE2_WIN12_RATE2", "COURCE2_WIN1_RATE2", "COURCE3_LOCAL_RACE_COUNT3", "COURCE3_LOCAL_WIN123_RATE3", "COURCE3_LOCAL_WIN12_RATE3", "COURCE3_LOCAL_WIN1_RATE3", "COURCE3_RACE_COUNT3", "COURCE3_WIN123_RATE3", "COURCE3_WIN12_RATE3", "COURCE3_WIN1_RATE3", "COURCE4_LOCAL_RACE_COUNT4", "COURCE4_LOCAL_WIN123_RATE4", "COURCE4_LOCAL_WIN12_RATE4", "COURCE4_LOCAL_WIN1_RATE4", "COURCE4_RACE_COUNT4", "COURCE4_WIN123_RATE4", "COURCE4_WIN12_RATE4", "COURCE4_WIN1_RATE4", "COURCE5_LOCAL_RACE_COUNT5", "COURCE5_LOCAL_WIN123_RATE5", "COURCE5_LOCAL_WIN12_RATE5", "COURCE5_LOCAL_WIN1_RATE5", "COURCE5_RACE_COUNT5", "COURCE5_WIN123_RATE5", "COURCE5_WIN12_RATE5", "COURCE5_WIN1_RATE5", "COURCE6_LOCAL_RACE_COUNT6", "COURCE6_LOCAL_WIN123_RATE6", "COURCE6_LOCAL_WIN12_RATE6", "COURCE6_LOCAL_WIN1_RATE6", "COURCE6_RACE_COUNT6", "COURCE6_WIN123_RATE6", "COURCE6_WIN12_RATE6", "COURCE6_WIN1_RATE6", "F1", "F2", "F3", "F4", "F5", "F6", "L1", "L2", "L3", "L4", "L5", "L6", "LOCALWIN1RATE1", "LOCALWIN1RATE2", "LOCALWIN1RATE3", "LOCALWIN1RATE4", "LOCALWIN1RATE5", "LOCALWIN1RATE6", "LOCALWIN2RATE1", "LOCALWIN2RATE2", "LOCALWIN2RATE3", "LOCALWIN2RATE4", "LOCALWIN2RATE5", "LOCALWIN2RATE6", "MOTORWIN2RATE1", "MOTORWIN2RATE2", "MOTORWIN2RATE3", "MOTORWIN2RATE4", "MOTORWIN2RATE5", "MOTORWIN2RATE6", "ST_AVG1", "ST_AVG2", "ST_AVG3", "ST_AVG4", "ST_AVG5", "ST_AVG6", "WEIGHT1", "WEIGHT2", "WEIGHT3", "WEIGHT4", "WEIGHT5", "WEIGHT6", "WIN1RATE1", "WIN1RATE2", "WIN1RATE3", "WIN1RATE4", "WIN1RATE5", "WIN1RATE6", "WIN2RATE1", "WIN2RATE2", "WIN2RATE3", "WIN2RATE4", "WIN2RATE5", "WIN2RATE6"]))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors='coerce'))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

numerical_transformers = [("numerical", numerical_pipeline, ["COURCE6_RACE_COUNT6", "L4", "WEIGHT2", "F4", "COURCE4_LOCAL_WIN123_RATE4", "COURCE1_LOCAL_RACE_COUNT1", "COURCE4_LOCAL_WIN1_RATE4", "WIN1RATE6", "COURCE6_LOCAL_WIN123_RATE6", "WIN1RATE1", "COURCE1_WIN1_RATE1", "COURCE5_LOCAL_WIN123_RATE5", "COURCE5_LOCAL_WIN12_RATE5", "BOATWIN2RATE1", "COURCE3_LOCAL_WIN1_RATE3", "COURCE4_WIN12_RATE4", "AGE4", "LOCALWIN2RATE5", "COURCE6_LOCAL_WIN12_RATE6", "BOATWIN2RATE4", "WIN2RATE4", "COURCE5_LOCAL_RACE_COUNT5", "COURCE3_LOCAL_WIN123_RATE3", "LOCALWIN2RATE6", "COURCE4_WIN123_RATE4", "MOTORWIN2RATE5", "COURCE3_RACE_COUNT3", "BOATWIN2RATE3", "COURCE1_WIN12_RATE1", "COURCE2_WIN1_RATE2", "F5", "WIN2RATE6", "COURCE6_LOCAL_RACE_COUNT6", "ST_AVG3", "L1", "MOTORWIN2RATE2", "BOATWIN2RATE5", "LOCALWIN1RATE2", "CLASS4", "CLASS2", "MOTORWIN2RATE1", "LOCALWIN1RATE6", "LOCALWIN1RATE3", "COURCE2_LOCAL_WIN123_RATE2", "F3", "WEIGHT5", "ST_AVG2", "LOCALWIN2RATE2", "L5", "WIN2RATE2", "AGE5", "ST_AVG5", "AGE2", "BOATWIN2RATE6", "COURCE3_LOCAL_RACE_COUNT3", "COURCE4_LOCAL_RACE_COUNT4", "COURCE3_LOCAL_WIN12_RATE3", "WIN2RATE5", "COURCE2_WIN12_RATE2", "COURCE5_WIN123_RATE5", "COURCE6_WIN12_RATE6", "ST_AVG1", "COURCE5_LOCAL_WIN1_RATE5", "AGE3", "COURCE6_WIN1_RATE6", "COURCE5_WIN1_RATE5", "COURCE2_RACE_COUNT2", "AGE1", "COURCE5_WIN12_RATE5", "WIN1RATE3", "WIN1RATE5", "COURCE1_RACE_COUNT1", "WEIGHT6", "WIN2RATE1", "F1", "WIN1RATE2", "ST_AVG4", "BOATWIN2RATE2", "WEIGHT4", "COURCE2_WIN123_RATE2", "CLASS1", "LOCALWIN1RATE4", "CLASS6", "COURCE5_RACE_COUNT5", "L6", "ST_AVG6", "COURCE1_LOCAL_WIN123_RATE1", "COURCE3_WIN12_RATE3", "L3", "LOCALWIN2RATE3", "COURCE1_WIN123_RATE1", "LOCALWIN2RATE4", "MOTORWIN2RATE6", "MOTORWIN2RATE3", "L2", "COURCE6_LOCAL_WIN1_RATE6", "WIN2RATE3", "LOCALWIN1RATE5", "CLASS3", "F2", "WEIGHT1", "LOCALWIN1RATE1", "WIN1RATE4", "COURCE1_LOCAL_WIN12_RATE1", "COURCE2_LOCAL_RACE_COUNT2", "MOTORWIN2RATE4", "COURCE4_LOCAL_WIN12_RATE4", "COURCE6_WIN123_RATE6", "COURCE2_LOCAL_WIN1_RATE2", "COURCE2_LOCAL_WIN12_RATE2", "COURCE3_WIN123_RATE3", "COURCE4_RACE_COUNT4", "WEIGHT3", "COURCE1_LOCAL_WIN1_RATE1", "AGE6", "F6", "CLASS5", "COURCE3_WIN1_RATE3", "COURCE4_WIN1_RATE4", "LOCALWIN2RATE1"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### Low-cardinality categoricals
# MAGIC Convert each low-cardinality categorical column into multiple binary columns through one-hot encoding.
# MAGIC For each input categorical column (string or numeric), the number of output columns is equal to the number of unique values in the input column.

# COMMAND ----------

from databricks.automl_runtime.sklearn import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

one_hot_imputers = []

one_hot_pipeline = Pipeline(steps=[
    ("imputers", ColumnTransformer(one_hot_imputers, remainder="passthrough")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="indicator")),
])

categorical_one_hot_transformers = [("onehot", one_hot_pipeline, ["CLUB1", "CLUB2", "CLUB3", "CLUB4", "CLUB5", "CLUB6", "F1", "F2", "F3", "PLACE", "RACE"])]

# COMMAND ----------

from sklearn.compose import ColumnTransformer

transformers = datetime_transformers + bool_transformers + numerical_transformers + categorical_one_hot_transformers

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

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
# MAGIC
# MAGIC Given that `RACEDATE` is provided as the `time_col`, the data is split based on time order,
# MAGIC where the most recent data is split to the test data.

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

# AutoML balanced the data internally and use _automl_sample_weight_0000 to calibrate the probability distribution
sample_weight = X_train.loc[:, "_automl_sample_weight_0000"].to_numpy()
X_train = X_train.drop(["_automl_sample_weight_0000"], axis=1)
X_val = X_val.drop(["_automl_sample_weight_0000"], axis=1)
X_test = X_test.drop(["_automl_sample_weight_0000"], axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/3031282363653284)
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
pipeline_val = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)

def objective(params):
  with mlflow.start_run(experiment_id="3031282363653284") as mlflow_run:
    lgbmc_classifier = LGBMClassifier(**params)

    best_model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", preprocessor),
        ("classifier", lgbmc_classifier),
    ])

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        silent=True)

    best_model.fit(X_train, y_train, classifier__callbacks=[lightgbm.early_stopping(5), lightgbm.log_evaluation(0)], classifier__eval_set=[(X_val_processed,y_val)], classifier__sample_weight=sample_weight)

    
    # Log metrics for the training set
    mlflow_model = Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=best_model)
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_train.assign(**{str(target_col):y_train}),
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_" , "sample_weight": sample_weight }
    )
    lgbmc_training_metrics = training_eval_result.metrics
    # Log metrics for the validation set
    val_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_val.assign(**{str(target_col):y_val}),
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "val_"  }
    )
    lgbmc_val_metrics = val_eval_result.metrics
    # Log metrics for the test set
    test_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test.assign(**{str(target_col):y_test}),
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "test_"  }
    )
    lgbmc_test_metrics = test_eval_result.metrics

    loss = -lgbmc_val_metrics["val_roc_auc"]

    # Truncate metric key names so they can be displayed together
    lgbmc_val_metrics = {k.replace("val_", ""): v for k, v in lgbmc_val_metrics.items()}
    lgbmc_test_metrics = {k.replace("test_", ""): v for k, v in lgbmc_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": lgbmc_val_metrics,
      "test_metrics": lgbmc_test_metrics,
      "model": best_model,
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

space = {
  "colsample_bytree": 0.6449481099714283,
  "lambda_l1": 1.0924303685882055,
  "lambda_l2": 235.78025288486074,
  "learning_rate": 0.07220516352335749,
  "max_bin": 56,
  "max_depth": 5,
  "min_child_samples": 63,
  "n_estimators": 439,
  "num_leaves": 5,
  "path_smooth": 20.895687603375798,
  "subsample": 0.6469705972548407,
  "random_state": 515957044,
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

trials = Trials()
fmin(objective,
     space=space,
     algo=tpe.suggest,
     max_evals=1,  # Increase this when widening the hyperparameter search space.
     trials=trials)

best_result = trials.best_trial["result"]
best_model = best_result["model"]
mlflow_run = best_result["run"]

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=["validation", "test"]))

set_config(display="diagram")
best_model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Patch pandas version in logged model
# MAGIC
# MAGIC Ensures that model serving uses the same version of pandas that was used to train the model.

# COMMAND ----------

import mlflow
import os
import shutil
import tempfile
import yaml

run_id = mlflow_run.info.run_id

# Set up a local dir for downloading the artifacts.
tmp_dir = tempfile.mkdtemp()

client = mlflow.tracking.MlflowClient()

# Fix conda.yaml
conda_file_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model/conda.yaml", dst_path=tmp_dir)
with open(conda_file_path) as f:
  conda_libs = yaml.load(f, Loader=yaml.FullLoader)
pandas_lib_exists = any([lib.startswith("pandas==") for lib in conda_libs["dependencies"][-1]["pip"]])
if not pandas_lib_exists:
  print("Adding pandas dependency to conda.yaml")
  conda_libs["dependencies"][-1]["pip"].append(f"pandas=={pd.__version__}")

  with open(f"{tmp_dir}/conda.yaml", "w") as f:
    f.write(yaml.dump(conda_libs))
  client.log_artifact(run_id=run_id, local_path=conda_file_path, artifact_path="model")

# Fix requirements.txt
venv_file_path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model/requirements.txt", dst_path=tmp_dir)
with open(venv_file_path) as f:
  venv_libs = f.readlines()
venv_libs = [lib.strip() for lib in venv_libs]
pandas_lib_exists = any([lib.startswith("pandas==") for lib in venv_libs])
if not pandas_lib_exists:
  print("Adding pandas dependency to requirements.txt")
  venv_libs.append(f"pandas=={pd.__version__}")

  with open(f"{tmp_dir}/requirements.txt", "w") as f:
    f.write("\n".join(venv_libs))
  client.log_artifact(run_id=run_id, local_path=venv_file_path, artifact_path="model")

shutil.rmtree(tmp_dir)

# COMMAND ----------

################################################
# pyfuncで再度登録してpredict_probaを登録することにする。
################################################
import functools

class model(mlflow.pyfunc.PythonModel):
  def __init__(self, best_model):
    self.best_model = best_model
  def load_context(self, context):
    import numpy as np
    import pandas as pd
    return
  # 入力はpandasデータフレームかシリーズとなります       
  def predict(self, context, model_input):
    import pandas as pd

    # 全ての確率を取得
    proba = self.best_model.predict_proba(model_input)
    
    # ラベルを取得
    pred = self.best_model.predict(model_input)
  
    # 多値分類の時にどのラベルに対応するのか判定はどうする？
    
    # 結果をデータフレームに変換
    df1 = pd.DataFrame(data=proba,columns=['prob1','prob2','prob3','prob4','prob5','prob6'])
    df2 = pd.DataFrame(data=pred,columns=['predict'])
   
    # データフレームを連結
    pdf = pd.concat([df1, df2 ], axis=1)
    
    #返却
    return pdf
  
###################
# ベストモデルを登録
###################
with mlflow.start_run() as model_run:  
  #MLflowトラッキングサーバーに記録
  mlflow.pyfunc.log_model("model", python_model=model(best_model))
  
# model_uri for the generated model
print("model_uri for the generated model:",f"runs:/{ model_run.info.run_id }/model")

# COMMAND ----------

import mlflow
#モデルを登録のIDでモデルをロード
print(f"runs:/{ model_run.info.run_id }/model")
logged_model = f"runs:/{ model_run.info.run_id }/model"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

# Python関数としてコールするので予測対象データはPandasデータフレームに変換して渡します。
input_table_name = "main.kyotei_db.predict"

# load table as a Spark DataFrame
table = spark.table(input_table_name)

# データロード
df1 = table.toPandas()

# COMMAND ----------

# 確率を含む予測結果の取得
proba = loaded_model.predict(df1)
# display(proba)

# データフレームを連結
pdf = pd.concat([df1, proba ], axis=1)

# Sparkデータフレームに変換
sdf = spark.createDataFrame(pdf)
sql("drop table if exists main.kyotei_db.predict_tan")
sdf.write.saveAsTable("main.kyotei_db.predict_tan")

# COMMAND ----------

# MAGIC %md ## 予測結果の分析

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   RACEDATE,
# MAGIC   PLACE,
# MAGIC   RACE,
# MAGIC   tan,
# MAGIC   tank,
# MAGIC   rentan3,
# MAGIC   predict,
# MAGIC   round(prob1,3) as p1,
# MAGIC   round(prob2,3) as p2,
# MAGIC   round(prob3,3) as p3,
# MAGIC   round(prob4,3) as p4,
# MAGIC   round(prob5,3) as p5,
# MAGIC   round(prob6,3) as p6
# MAGIC from
# MAGIC   main.kyotei_db.predict_tan

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC count(1),
# MAGIC sum(win),
# MAGIC round(sum(win) / count(1) , 3) as `勝率`,
# MAGIC sum(wink) as `払戻金額`,
# MAGIC count(1) * 100 as `コスト`,
# MAGIC sum(wink) - (count(1) * 100) as `利益金額`,
# MAGIC round(sum(wink) / (count(1) * 100),3) as `回収率`
# MAGIC from(
# MAGIC select
# MAGIC   RACEDATE,
# MAGIC   PLACE,
# MAGIC   RACE,
# MAGIC   tan,
# MAGIC   tank,
# MAGIC   rentan3,
# MAGIC   predict,
# MAGIC   round(prob1,3) as p1,
# MAGIC   round(prob2,3) as p2,
# MAGIC   round(prob3,3) as p3,
# MAGIC   round(prob4,3) as p4,
# MAGIC   round(prob5,3) as p5,
# MAGIC   round(prob6,3) as p6,
# MAGIC   case when predict = tan then 1 else 0 end as win,
# MAGIC   case when predict = tan then tank else 0 end as wink
# MAGIC from
# MAGIC   main.kyotei_db.predict_tan
# MAGIC   where predict=2 and round(prob2,3) between 0.4 and 0.41 and tan is not null
# MAGIC ) 

# COMMAND ----------

# MAGIC %md ### 2連単の予測UDF

# COMMAND ----------

# SparkSQLでの関数として実行できるようにする
from pyspark.sql.types import ArrayType, FloatType, StringType

def rentan2(p1,p2,p3,p4,p5,p6):
  import heapq
  #最大値・最小値から順にn個の要素を取得
  prob_list = [p1,p2,p3,p4,p5,p6]
  sorted_list = heapq.nlargest(3, prob_list)

  # 一着
  b1 = ""
  b2 = ""
  b3 = ""

  #1着の予測結果を出力
  if sorted_list[0] == p1:
    b1 = '1'
  if sorted_list[0] == p2:
    b1 = '2'
  if sorted_list[0] == p3:
    b1 = '3'
  if sorted_list[0] == p4:
    b1 = '4'
  if sorted_list[0] == p5:
    b1 = '5'
  if sorted_list[0] == p6:
    b1 = '6'

  #2着の予測結果を出力
  if sorted_list[1] == p1:
    b2 = '1'
  if sorted_list[1] == p2:
    b2 = '2'
  if sorted_list[1] == p3:
    b2 = '3'
  if sorted_list[1] == p4:
    b2 = '4'
  if sorted_list[1] == p5:
    b2 = '5'
  if sorted_list[1] == p6:
    b2 = '6'

  rentan2 = b1 + '>' + b2 
  return rentan2

#Spark-UDFとして登録　(戻り値の型を指定すること)
udf_rentan2 = udf(rentan2, StringType())

#Spark-UDFをSQL関数として登録
spark.udf.register('sql_rentan2', udf_rentan2 ) 

# COMMAND ----------

# MAGIC %md ### 2連複の予測UDF

# COMMAND ----------

# SparkSQLでの関数として実行できるようにする
from pyspark.sql.types import ArrayType, FloatType, StringType

def renfuku2(p1,p2,p3,p4,p5,p6):
  import heapq
  #最大値・最小値から順にn個の要素を取得
  prob_list = [p1,p2,p3,p4,p5,p6]
  sorted_list = heapq.nlargest(3, prob_list)

  # 一着
  b1 = ""
  b2 = ""
  b3 = ""

  #1着の予測結果を出力
  if sorted_list[0] == p1:
    b1 = '1'
  if sorted_list[0] == p2:
    b1 = '2'
  if sorted_list[0] == p3:
    b1 = '3'
  if sorted_list[0] == p4:
    b1 = '4'
  if sorted_list[0] == p5:
    b1 = '5'
  if sorted_list[0] == p6:
    b1 = '6'

  #2着の予測結果を出力
  if sorted_list[1] == p1:
    b2 = '1'
  if sorted_list[1] == p2:
    b2 = '2'
  if sorted_list[1] == p3:
    b2 = '3'
  if sorted_list[1] == p4:
    b2 = '4'
  if sorted_list[1] == p5:
    b2 = '5'
  if sorted_list[1] == p6:
    b2 = '6'

  # 並び替え
  rentan2 = [b1,b2]
  sorted_list = (heapq.nsmallest(2, rentan2))
  renfuku2 = sorted_list[0] + '=' + sorted_list[1] 
  return renfuku2

#Spark-UDFとして登録　(戻り値の型を指定すること)
udf_renfuku2 = udf(renfuku2, StringType())

#Spark-UDFをSQL関数として登録
spark.udf.register('sql_renfuku2', udf_renfuku2 ) 

# COMMAND ----------

# MAGIC %md ### 3連単の予測UDF

# COMMAND ----------

# SparkSQLでの関数として実行できるようにする
from pyspark.sql.types import ArrayType, FloatType, StringType

def rentan3(p1,p2,p3,p4,p5,p6):
  import heapq
  #最大値・最小値から順にn個の要素を取得
  prob_list = [p1,p2,p3,p4,p5,p6]
  sorted_list = heapq.nlargest(3, prob_list)

  # 一着
  b1 = ""
  b2 = ""
  b3 = ""

  #1着の予測結果を出力
  if sorted_list[0] == p1:
    b1 = '1'
  if sorted_list[0] == p2:
    b1 = '2'
  if sorted_list[0] == p3:
    b1 = '3'
  if sorted_list[0] == p4:
    b1 = '4'
  if sorted_list[0] == p5:
    b1 = '5'
  if sorted_list[0] == p6:
    b1 = '6'

  #2着の予測結果を出力
  if sorted_list[1] == p1:
    b2 = '1'
  if sorted_list[1] == p2:
    b2 = '2'
  if sorted_list[1] == p3:
    b2 = '3'
  if sorted_list[1] == p4:
    b2 = '4'
  if sorted_list[1] == p5:
    b2 = '5'
  if sorted_list[1] == p6:
    b2 = '6'

  #3着の予測結果を出力
  if sorted_list[2] == p1:
    b3 = '1'
  if sorted_list[2] == p2:
    b3 = '2'
  if sorted_list[2] == p3:
    b3 = '3'
  if sorted_list[2] == p4:
    b3 = '4'
  if sorted_list[2] == p5:
    b3 = '5'
  if sorted_list[2] == p6:
    b3 = '6'

  rentan3 = b1 + '>' + b2 + '>' + b3
  return rentan3

#Spark-UDFとして登録　(戻り値の型を指定すること)
udf_rentan3 = udf(rentan3, StringType())

#Spark-UDFをSQL関数として登録
spark.udf.register('sql_rentan3', udf_rentan3 ) 

# COMMAND ----------

# MAGIC %md ### 3連複の予測UDF

# COMMAND ----------

# SparkSQLでの関数として実行できるようにする
from pyspark.sql.types import ArrayType, FloatType, StringType

def renfuku3(p1,p2,p3,p4,p5,p6):
  import heapq
  #最大値・最小値から順にn個の要素を取得
  prob_list = [p1,p2,p3,p4,p5,p6]
  sorted_list = heapq.nlargest(3, prob_list)

  # 一着
  b1 = ""
  b2 = ""
  b3 = ""

  #1着の予測結果を出力
  if sorted_list[0] == p1:
    b1 = '1'
  if sorted_list[0] == p2:
    b1 = '2'
  if sorted_list[0] == p3:
    b1 = '3'
  if sorted_list[0] == p4:
    b1 = '4'
  if sorted_list[0] == p5:
    b1 = '5'
  if sorted_list[0] == p6:
    b1 = '6'

  #2着の予測結果を出力
  if sorted_list[1] == p1:
    b2 = '1'
  if sorted_list[1] == p2:
    b2 = '2'
  if sorted_list[1] == p3:
    b2 = '3'
  if sorted_list[1] == p4:
    b2 = '4'
  if sorted_list[1] == p5:
    b2 = '5'
  if sorted_list[1] == p6:
    b2 = '6'

  #3着の予測結果を出力
  if sorted_list[2] == p1:
    b3 = '1'
  if sorted_list[2] == p2:
    b3 = '2'
  if sorted_list[2] == p3:
    b3 = '3'
  if sorted_list[2] == p4:
    b3 = '4'
  if sorted_list[2] == p5:
    b3 = '5'
  if sorted_list[2] == p6:
    b3 = '6'

  # 並び替え
  rentan3 = [b1,b2,b3]
  sorted_list = (heapq.nsmallest(3, rentan3))
  renfuku3 = sorted_list[0] + '=' + sorted_list[1] + '=' + sorted_list[2]
  return renfuku3

#Spark-UDFとして登録　(戻り値の型を指定すること)
udf_renfuku3 = udf(renfuku3, StringType())

#Spark-UDFをSQL関数として登録
spark.udf.register('sql_renfuku3', udf_renfuku3 ) 

# COMMAND ----------

# MAGIC %md ### 上位３着の予測

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   RACEDATE,
# MAGIC   PLACE,
# MAGIC   RACE,
# MAGIC   --rentan3,
# MAGIC   renfuku3,
# MAGIC   --sql_rentan3(prob1,prob2,prob3,prob4,prob5,prob6) as p_rentan3,
# MAGIC   sql_renfuku3(prob1,prob2,prob3,prob4,prob5,prob6) as p_renfuku3,
# MAGIC   round(prob1,3) as p1,
# MAGIC   round(prob2,3) as p2,
# MAGIC   round(prob3,3) as p3,
# MAGIC   round(prob4,3) as p4,
# MAGIC   round(prob5,3) as p5,
# MAGIC   round(prob6,3) as p6
# MAGIC from
# MAGIC   main.kyotei_db.predict_tan

# COMMAND ----------

# MAGIC %md ### 上位2着の予測

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   RACEDATE,
# MAGIC   PLACE,
# MAGIC   RACE,
# MAGIC   --rentan2,
# MAGIC   renfuku2,
# MAGIC   --sql_rentan2(prob1,prob2,prob3,prob4,prob5,prob6) as p_rentan2,
# MAGIC   sql_renfuku2(prob1,prob2,prob3,prob4,prob5,prob6) as p_renfuku2,
# MAGIC   round(prob1,3) as p1,
# MAGIC   round(prob2,3) as p2,
# MAGIC   round(prob3,3) as p3,
# MAGIC   round(prob4,3) as p4,
# MAGIC   round(prob5,3) as p5,
# MAGIC   round(prob6,3) as p6
# MAGIC from
# MAGIC   main.kyotei_db.predict_tan

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).
# MAGIC
# MAGIC > **NOTE:** SHAP run may take a long time with the datetime columns in the dataset.

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = False

# COMMAND ----------

if shap_enabled:
    mlflow.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, X_train.shape[0]), random_state=515957044)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(100, X_val.shape[0]), random_state=515957044)

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="logit")
    shap_values = explainer.shap_values(example, l1_reg=False, nsamples=500)
    summary_plot(shap_values, example, class_names=model.classes_,max_display=50)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) is a collaborative hub where teams can share ML models, work together from experimentation to online testing and production, integrate with approval and governance workflows, and monitor ML deployments and their performance. The snippets below show how to add the model trained in this notebook to the model registry and to retrieve it later for inference.
# MAGIC
# MAGIC > **NOTE:** The `model_uri` for the model already trained in this notebook can be found in the cell below
# MAGIC
# MAGIC ### Register to Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```
# MAGIC
# MAGIC ### Load from Model Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC model_version = registered_model_version.version
# MAGIC
# MAGIC model_uri=f"models:/{model_name}/{model_version}"
# MAGIC model = mlflow.pyfunc.load_model(model_uri=model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```
# MAGIC
# MAGIC ### Load model without registering
# MAGIC ```
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC
# MAGIC model = mlflow.pyfunc.load_model(model_uri=model_uri)
# MAGIC model.predict(input_X)
# MAGIC ```

# COMMAND ----------

# model_uri for the generated model
print(f"runs:/{ mlflow_run.info.run_id }/model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix for validation data
# MAGIC
# MAGIC We show the confusion matrix of the model on the validation data.
# MAGIC
# MAGIC For the plots evaluated on the training and the test data, check the artifacts on the MLflow run page.

# COMMAND ----------

# Click the link to see the MLflow run page
displayHTML(f"<a href=#mlflow/experiments/3031282363653284/runs/{ mlflow_run.info.run_id }/artifactPath/model> Link to model run page </a>")

# COMMAND ----------

import uuid
from IPython.display import Image

# Create temp directory to download MLflow model artifact
eval_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(eval_temp_dir, exist_ok=True)

# Download the artifact
eval_path = mlflow.artifacts.download_artifacts(run_id=mlflow_run.info.run_id, dst_path=eval_temp_dir)

# COMMAND ----------

eval_confusion_matrix_path = os.path.join(eval_path, "val_confusion_matrix.png")
display(Image(filename=eval_confusion_matrix_path))
