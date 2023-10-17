# Databricks notebook source
# MAGIC %sql
# MAGIC use catalog main;
# MAGIC use kyotei_db;

# COMMAND ----------

#　累積テーブル削除
renfuku = sqlContext.sql(\
"         select '123' as renfuku \
union all select '124' \
").rdd.map(lambda row : row[0]).collect()

dbutils.widgets.dropdown("RENFUKU", "123", [str(x) for x in renfuku],"3連複")
RENFUKU=dbutils.widgets.get("RENFUKU")

# COMMAND ----------

# 予測結果テーブルの名称取得
kekka_table = "kekka_" + RENFUKU
print(kekka_table)

# COMMAND ----------

# MAGIC %md ###  全てのモデルの予測結果を集計する

# COMMAND ----------

# MAGIC %python
# MAGIC out1 = sql(
# MAGIC   f"""select 
# MAGIC   racedate,
# MAGIC   place,
# MAGIC   race,
# MAGIC   rentan2,
# MAGIC   renfuku2,
# MAGIC   rentan2k,
# MAGIC   renfuku2k,
# MAGIC   rentan3,
# MAGIC   rentan3k,
# MAGIC   renfuku3,
# MAGIC   renfuku3k,
# MAGIC   kekka,
# MAGIC   count(1) as cnt  from kekka_{ RENFUKU } 
# MAGIC   group by 
# MAGIC racedate,
# MAGIC   place,
# MAGIC   race,
# MAGIC   rentan2,
# MAGIC   renfuku2,
# MAGIC   rentan2k,
# MAGIC   renfuku2k,
# MAGIC   rentan3,
# MAGIC   rentan3k,
# MAGIC   renfuku3,
# MAGIC   renfuku3k,
# MAGIC   kekka
# MAGIC   """
# MAGIC )
# MAGIC display(out1)

# COMMAND ----------

# MAGIC %md ### 全モデル数は以下のテーブルから取得する

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) as cnt from expno01
# MAGIC union all 
# MAGIC select count(*) as cnt from expno02
# MAGIC union all 
# MAGIC select count(*) as cnt from expno03
# MAGIC union all 
# MAGIC select count(*) as cnt from expno04
# MAGIC union all 
# MAGIC select count(*) as cnt from expno05

# COMMAND ----------

# MAGIC %python
# MAGIC # 使用したモデル数を取得
# MAGIC model_total = sql(
# MAGIC """
# MAGIC select sum(cnt) from 
# MAGIC ( 
# MAGIC select count(*) as cnt from expno01
# MAGIC union all 
# MAGIC select count(*) as cnt from expno02
# MAGIC union all 
# MAGIC select count(*) as cnt from expno03
# MAGIC union all 
# MAGIC select count(*) as cnt from expno04
# MAGIC union all 
# MAGIC select count(*) as cnt from expno05
# MAGIC )
# MAGIC """
# MAGIC ).toPandas().to_string(index=False, header=False)
# MAGIC
# MAGIC print(model_total)

# COMMAND ----------

# MAGIC %md ### 各モデルのアンサンブルを取る

# COMMAND ----------

# MAGIC %python
# MAGIC out2 = sql(
# MAGIC   f"""select 
# MAGIC   expno,
# MAGIC   racedate,
# MAGIC   place,
# MAGIC   race,
# MAGIC   rentan2,
# MAGIC   renfuku2,
# MAGIC   rentan2k,
# MAGIC   renfuku2k,
# MAGIC   rentan3,
# MAGIC   rentan3k,
# MAGIC   renfuku3,
# MAGIC   renfuku3k,
# MAGIC   kekka,
# MAGIC   count(1) as cnt  from kekka_{ RENFUKU } 
# MAGIC   group by 
# MAGIC   expno,
# MAGIC   racedate,
# MAGIC   place,
# MAGIC   race,
# MAGIC   rentan2,
# MAGIC   renfuku2,
# MAGIC   rentan2k,
# MAGIC   renfuku2k,
# MAGIC   rentan3,
# MAGIC   rentan3k,
# MAGIC   renfuku3,
# MAGIC   renfuku3k,
# MAGIC   kekka
# MAGIC   """
# MAGIC )
# MAGIC display(out2)

# COMMAND ----------

sql(
f"""
create or replace table kekka_{ RENFUKU }_sum1 as 
select 
  expno,
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
  count(1) as cnt 
  from kekka_{ RENFUKU } 
  group by 
  expno,
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

out2 = sql(f"select * from kekka_{ RENFUKU }_sum1")
display(out2)

# COMMAND ----------

sql(
f"""
create or replace table kekka_{ RENFUKU }_sum2 as 
select 
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
count(1) as exp_vote,
sum(cnt) as model_num,
max( {model_total} ) as model_tolal_num
--round(percent_rank() over( order by cnt) , 3 ) as percent_rank
from kekka_{ RENFUKU }_sum1
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

out2 = sql(f"select * from kekka_{ RENFUKU }_sum2")
display(out2)

# COMMAND ----------

if RENFUKU == "123":
  print("過去データの実績")
  display(
    sql(
        f"""
            select 
            '1>2>3 or 1>3>2' as target,
            count(1) as race_cnt ,
            sum(case when rentan3 in ('1>2>3','1>3>2') then 1 else 0 end ) as win_cnt ,
            round(sum(case when rentan3 in ('1>2>3','1>3>2') then 1 else 0 end )/ cast( count(1) as double),2) as win_rate,
            count(1)*200 as cost,
            sum(case when rentan3 in ('1>2>3','1>3>2') then rentan3k else 0 end ) as win_rentan3k ,
            sum(case when rentan3 in ('1>2>3','1>3>2') then rentan3k else 0 end ) - (count(1)*200) as profit,
            ( sum(case when rentan3 in ('1>2>3','1>3>2') then rentan3k else 0 end ) ) / (count(1)*200) as profit_drate,
            round(sum(case when rentan3 in ('1>2>3','1>3>2') then rentan3k else 0 end ) / sum(case when rentan3 in ('1>2>3','1>3>2') then 1 else 0 end ),1)  as avg_wink
            from ( select distinct * from kekka_{RENFUKU}_sum2 where rentan3 is not null and rentan3 not like '%不成立%' )
            """
    )
  )
  
  display(
    sql(
        f"""
            select 
            '1=2=3' as target,
            count(1) as race_cnt ,
            sum(case when renfuku3 in ('1=2=3') then 1 else 0 end ) as win_cnt ,
            round(sum(case when renfuku3 in ('1=2=3') then 1 else 0 end )/ cast( count(1) as double),2) as win_rate,
            count(1)*100 as cost,
            sum(case when renfuku3 in ('1=2=3') then renfuku3k else 0 end ) as win_renfuku3k, 
            sum(case when renfuku3 in ('1=2=3') then renfuku3k else 0 end ) - (count(1)*100) as profit,
            ( sum(case when renfuku3 in ('1=2=3') then renfuku3k else 0 end )) / (count(1)*100) as profit_drate,
            round(sum(case when renfuku3 in ('1=2=3') then renfuku3k else 0 end ) / sum(case when renfuku3 in ('1=2=3') then 1 else 0 end ),1)  as avg_wink
            from ( select distinct * from kekka_{RENFUKU}_sum2 where rentan3 is not null and rentan3 not like '%不成立%')
            """
    )
  )

  display(
    sql(
        f"""
            select 
            '1>2 or 1>3' as target,
            count(1) as race_cnt ,
            sum(case when rentan2 in ('1>2','1>3') then 1 else 0 end ) as win_cnt ,
            round(sum(case when rentan2 in ('1>2','1>3') then 1 else 0 end )/ cast( count(1) as double),2) as win_rate,
            count(1)*200 as cost,
            sum(case when rentan2 in ('1>2','1>3') then rentan2k else 0 end ) as win_rentan2k ,
            sum(case when rentan2 in ('1>2','1>3') then rentan2k else 0 end ) - (count(1)*200) as profit,
            ( sum(case when rentan2 in ('1>2','1>3') then rentan2k else 0 end ) ) / (count(1)*200) as profit_drate,
            round(sum(case when rentan2 in ('1>2','1>3') then rentan2k else 0 end ) / sum(case when rentan2 in ('1>2','1>3') then 1 else 0 end ),1)  as avg_wink
            from ( select distinct * from kekka_{RENFUKU}_sum2 where rentan2 is not null and rentan2 not like '%不成立%' )
            """
    )
  )

  display(
    sql(
        f"""
            select 
            '1=2 or 1=3' as target,
            count(1) as race_cnt ,
            sum(case when renfuku2 in ('1=2','1=3') then 1 else 0 end ) as win_cnt ,
            round(sum(case when renfuku2 in ('1=2','1=3') then 1 else 0 end )/ cast( count(1) as double),2) as win_rate,
            count(1)*200 as cost,
            sum(case when renfuku2 in ('1=2','1=3') then renfuku2k else 0 end ) as win_renfuku2k ,
            sum(case when renfuku2 in ('1=2','1=3') then renfuku2k else 0 end ) - (count(1)*200) as profit,
            ( sum(case when renfuku2 in ('1=2','1=3') then renfuku2k else 0 end ) ) / (count(1)*200) as profit_drate,
            round(sum(case when renfuku2 in ('1=2','1=3') then renfuku2k else 0 end ) / sum(case when renfuku2 in ('1=2','1=3') then 1 else 0 end ),1)  as avg_wink
            from ( select distinct * from kekka_{RENFUKU}_sum2 where renfuku2 is not null and renfuku2 not like '%不成立%' )
            """
    )
  )

  # 予測結果
  print("予測結果")
  out4 = sql(f"select racedate,place,race, '1=2=3' as predict from kekka_{RENFUKU}_sum where rentan3 is null")
  display(out4)

# COMMAND ----------

if RENFUKU == "124":
  display(
    sql(
        f"""
            select 
            '1>2>4 or 1>4>2' as target,
            count(1) as race_cnt ,
            sum(case when rentan3 in ('1>2>4','1>4>2') then 1 else 0 end ) as win_cnt ,
            round(sum(case when rentan3 in ('1>2>4','1>4>2') then 1 else 0 end )/ cast( count(1) as double),2) as win_rate,
            count(1)*200 as cost,
            sum(case when rentan3 in ('1>2>4','1>4>2') then rentan3k else 0 end ) as win_rentan3k ,
            sum(case when rentan3 in ('1>2>4','1>4>2') then rentan3k else 0 end ) - (count(1)*200) as profit,
            ( sum(case when rentan3 in ('1>2>4','1>4>2') then rentan3k else 0 end ) ) / (count(1)*200) as profit_drate,
            round(sum(case when rentan3 in ('1>2>4','1>4>2') then rentan3k else 0 end ) / sum(case when rentan3 in ('1>2>4','1>4>2') then 1 else 0 end ),1)  as avg_wink
            from ( select distinct * from kekka_{RENFUKU}_sum2 where rentan3 is not null and rentan3 not like '%不成立%' )
            """
    )
  )
  
  display(
    sql(
        f"""
            select 
            '1=2=4' as target,
            count(1) as race_cnt ,
            sum(case when renfuku3 in ('1=2=4') then 1 else 0 end ) as win_cnt ,
            round(sum(case when renfuku3 in ('1=2=4') then 1 else 0 end )/ cast( count(1) as double),2) as win_rate,
            count(1)*100 as cost,
            sum(case when renfuku3 in ('1=2=4') then renfuku3k else 0 end ) as win_renfuku3k, 
            sum(case when renfuku3 in ('1=2=4') then renfuku3k else 0 end ) - (count(1)*100) as profit,
            ( sum(case when renfuku3 in ('1=2=4') then renfuku3k else 0 end )) / (count(1)*100) as profit_drate,
            round(sum(case when renfuku3 in ('1=2=4') then renfuku3k else 0 end ) / sum(case when renfuku3 in ('1=2=4') then 1 else 0 end ),1)  as avg_wink
            from ( select distinct * from kekka_{RENFUKU}_sum2 where rentan3 is not null and rentan3 not like '%不成立%')
            """
    )
  )

  out4 = sql(f"select racedate,place,race, '1=2=3' as predict from kekka_{RENFUKU}_all where rentan3 is null")
  display(out4)