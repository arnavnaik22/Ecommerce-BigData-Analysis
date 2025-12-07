"""
E-Commerce Big Data Analytics Pipeline (Final Fixed Version)
Authors: Arnav Devdatt Naik, Gargi Pant
Date: October 12, 2025
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# Initialize Spark
spark = SparkSession.builder.appName("ECommerce_Analysis").getOrCreate()
sc = spark.sparkContext
start_time = time.time()

print("\n" + "="*100)
print("E-COMMERCE BIG DATA ANALYSIS PIPELINE - FINAL FIXED VERSION")
print("="*100 + "\n")

# ===================================================================
# SECTION 1: DATA LOADING & CLEANING
# ===================================================================
df = spark.read.csv("hdfs:///ecommerce/ecommerce_sample.csv", header=True, inferSchema=True)
initial_count = df.count()
print(f"Initial records: {initial_count:,}")

df = df.filter(col("user_id").isNotNull() & col("product_id").isNotNull() & col("event_type").isNotNull())
df = df.fillna({"brand": "unknown", "category_code": "unknown", "price": 0.0})

# ===================================================================
# SECTION 2: FEATURE ENGINEERING
# ===================================================================
df = df.withColumn("event_hour", hour("event_time")) \
       .withColumn("event_date", to_date("event_time")) \
       .withColumn("day_of_week", dayofweek("event_time")) \
       .withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0)) \
       .withColumn("time_segment", 
                   when(col("event_hour").between(6, 11), "morning")
                   .when(col("event_hour").between(12, 17), "afternoon")
                   .when(col("event_hour").between(18, 22), "evening")
                   .otherwise("night")) \
       .withColumn("price_category",
                   when(col("price") < 50, "budget")
                   .when(col("price") < 200, "mid-range")
                   .otherwise("premium"))

print(f"Records processed: {df.count():,}\n")

# ===================================================================
# SECTION 3: BEHAVIORAL ANALYSIS
# ===================================================================
print("SECTION 3: BEHAVIORAL ANALYSIS")
print("-" * 100)
event_counts = df.groupBy("event_type").count().collect()
metrics_dict = {r.event_type: r['count'] for r in event_counts}
view_cnt = metrics_dict.get('view', 0)
cart_cnt = metrics_dict.get('cart', 0)
purch_cnt = metrics_dict.get('purchase', 0)

view_to_cart = (cart_cnt / (view_cnt if view_cnt > 0 else 1)) * 100
view_to_purchase = (purch_cnt / (view_cnt if view_cnt > 0 else 1)) * 100
cart_abandonment = ((cart_cnt - purch_cnt) / (cart_cnt if cart_cnt > 0 else 1)) * 100

print(f"Views: {view_cnt:,}")
print(f"Add to Cart: {cart_cnt:,} ({view_to_cart:.2f}%)")
print(f"Purchases: {purch_cnt:,} ({view_to_purchase:.2f}%)")
print(f"Cart Abandonment: {cart_abandonment:.2f}%\n")

time_conversion = (
    df.groupBy("time_segment", "event_type")
      .count()
      .groupBy("time_segment")
      .pivot("event_type")
      .sum("count")
      .fillna(0)
)
time_conversion = time_conversion.withColumn(
    "conversion_rate", col("purchase") / (col("view") + 1) * 100
)

best_time = time_conversion.orderBy(col("conversion_rate").desc()).first()
print(f"Best conversion time: {best_time['time_segment'].upper()} ({best_time['conversion_rate']:.2f}%)\n")

top_brands = (
    df.filter(col("event_type") == "purchase")
      .groupBy("brand")
      .agg(sum("price").alias("revenue"),
           count("*").alias("num_sales"),
           avg("price").alias("avg_sale"))
      .orderBy(col("revenue").desc())
      .limit(10)
)
top_brand = top_brands.first()
print(f"Top Brand: {top_brand.brand} - ${top_brand.revenue:,.2f}\n")

# ===================================================================
# SECTION 4: CUSTOMER SEGMENTATION (RFM)
# ===================================================================
print("SECTION 4: CUSTOMER SEGMENTATION")
print("-" * 100)
max_date = df.agg(max("event_date")).first()[0]
rfm_data = (
    df.filter(col("event_type") == "purchase")
      .groupBy("user_id")
      .agg(
          datediff(lit(max_date), max("event_date")).alias("recency"),
          count("*").alias("frequency"),
          sum("price").alias("monetary")
      )
)
rfm_data = rfm_data.withColumn(
    "segment",
    when((col("recency") <= 7) & (col("frequency") >= 3) & (col("monetary") >= 500), "VIP")
    .when((col("recency") <= 15) & (col("frequency") >= 2), "Loyal")
    .when(col("recency") <= 30, "Active")
    .otherwise("At-Risk")
)
segment_stats = (
    rfm_data.groupBy("segment")
    .agg(
        count("*").alias("count"),
        avg("monetary").alias("avg_spend"),
        avg("frequency").alias("avg_purchases"),
        avg("recency").alias("avg_recency")
    )
    .orderBy(col("avg_spend").desc())
)
for row in segment_stats.collect():
    print(f"{row.segment}: {row['count']:,} users, "
          f"Avg spend ${row.avg_spend:,.2f}, "
          f"Purchases {row.avg_purchases:.1f}, Recency {row.avg_recency:.0f} days")
print()

# ===================================================================
# SECTION 5: RDD OPERATIONS
# ===================================================================
print("SECTION 5: RDD OPERATIONS")
print("-" * 100)
rdd = df.rdd.repartition(4)
purchase_rdd = rdd.filter(lambda x: x.event_type == 'purchase')
brand_rev = purchase_rdd.map(lambda x: (x.brand, x.price)).reduceByKey(lambda a, b: a + b)
print(f"Total purchases: {purchase_rdd.count():,}")
print(f"Unique products: {rdd.map(lambda x: x.product_id).distinct().count():,}")
print(f"Event counts: {rdd.map(lambda x: x.event_type).countByValue()}\n")

# ===================================================================
# SECTION 6: MACHINE LEARNING (Fixed Label Definition)
# ===================================================================
print("SECTION 6: MACHINE LEARNING (Purchase Prediction)")
print("-" * 100)

# Aggregate user features
user_features = df.groupBy("user_id").agg(
    count("*").alias("total_events"),
    countDistinct("product_id").alias("unique_products"),
    sum(when(col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
    sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("cart_count"),
    sum(when(col("is_weekend") == 1, 1).otherwise(0)).alias("weekend_activity"),
    avg("price").alias("avg_price"),
    max("price").alias("max_price")
).fillna(0.0)

# Label creation FIX: mark users who actually made a purchase
purchase_flag = (
    df.filter(col("event_type") == "purchase")
      .select("user_id")
      .distinct()
      .withColumn("purchased", lit(1))
)
user_features = user_features.join(purchase_flag, "user_id", "left").fillna({"purchased": 0})

# Derived features
user_features = (
    user_features
    .withColumn("engagement_score", col("view_count") + col("cart_count") * 2)
    .withColumn("cart_rate", col("cart_count") / (col("view_count") + 1))
)

feature_cols = [
    "total_events", "unique_products", "view_count", "cart_count",
    "weekend_activity", "avg_price", "max_price",
    "engagement_score", "cart_rate"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
ml_data = assembler.transform(user_features).select("features", "purchased")

train, test = ml_data.randomSplit([0.8, 0.2], seed=42)
print(f"Train size: {train.count():,}, Test size: {test.count():,}\n")

rf = RandomForestClassifier(labelCol="purchased", featuresCol="features", seed=42)
param_grid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [20, 30])
    .addGrid(rf.maxDepth, [6, 8])
    .build()
)
evaluator = BinaryClassificationEvaluator(labelCol="purchased", metricName="areaUnderROC")
cv = CrossValidator(estimator=rf, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3, seed=42)

print("Training with cross-validation...")
cv_model = cv.fit(train)
best_model = cv_model.bestModel
pred = cv_model.transform(test)
auc = evaluator.evaluate(pred)

# Compute confusion matrix
pdf = pred.select("purchased", "prediction").toPandas()
y_true, y_pred = pdf["purchased"].values, pdf["prediction"].values
tn = np.sum((y_true == 0) & (y_pred == 0))
fp = np.sum((y_true == 0) & (y_pred == 1))
fn = np.sum((y_true == 1) & (y_pred == 0))
tp = np.sum((y_true == 1) & (y_pred == 1))
acc = (tp + tn) / (tp + tn + fp + fn)
prec = tp / (tp + fp) if (tp + fp) else 0
rec = tp / (tp + fn) if (tp + fn) else 0
f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) else 0

print(f"AUC-ROC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-Score: {f1:.4f}\n")

# ===================================================================
# SECTION 7: SUMMARY
# ===================================================================
print("="*100)
print("EXECUTION SUMMARY")
print("="*100)
print(f"Total time: {(time.time()-start_time)/60:.2f} minutes")
print(f"Records processed: {df.count():,}")
print(f"Models trained: 12 (cross-validation)")
print("="*100 + "\n")

spark.stop()
