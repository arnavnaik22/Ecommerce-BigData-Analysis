"""
E-Commerce Big Data Analytics Pipeline
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
print("E-COMMERCE BIG DATA ANALYSIS PIPELINE")
print("="*100 + "\n")

# SECTION 1: DATA LOADING
print("SECTION 1: DATA LOADING")
print("-" * 100)
df = spark.read.csv("hdfs:///ecommerce/ecommerce_sample.csv", header=True, inferSchema=True)
initial_count = df.count()
print(f"Initial records: {initial_count:,}")

df = df.filter(col("user_id").isNotNull() & col("product_id").isNotNull() & col("event_type").isNotNull())
df = df.fillna({"brand": "unknown", "category_code": "unknown", "price": 0.0})

# Feature engineering
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

total_records = df.count()
print(f"Records processed: {total_records:,}\n")

# SECTION 2: BEHAVIORAL ANALYSIS
print("SECTION 2: BEHAVIORAL ANALYSIS")
print("-" * 100)
event_counts = df.groupBy("event_type").count().collect()
metrics_dict = {row.event_type: row['count'] for row in event_counts}
view_cnt = metrics_dict.get('view', 0)
cart_cnt = metrics_dict.get('cart', 0)
purch_cnt = metrics_dict.get('purchase', 0)
view_to_cart = (cart_cnt / (view_cnt if view_cnt > 0 else 1)) * 100
view_to_purchase = (purch_cnt / (view_cnt if view_cnt > 0 else 1)) * 100
cart_abandonment = ((cart_cnt - purch_cnt) / (cart_cnt if cart_cnt > 0 else 1)) * 100

print(f"Views: {view_cnt:,}")
print(f"Add to Cart: {cart_cnt:,} ({view_to_cart:.2f}%)")
print(f"Purchases: {purch_cnt:,} ({view_to_purchase:.2f}%)")
print(f"Conversion Rate: {view_to_purchase:.2f}%")
print(f"Cart Abandonment: {cart_abandonment:.2f}%\n")

# Time analysis
time_conversion = df.groupBy("time_segment", "event_type").count() \
                     .groupBy("time_segment").pivot("event_type").sum("count").fillna(0)
time_conversion = time_conversion.withColumn("conversion_rate", 
                                              col("purchase") / (col("view") + 1) * 100)
all_times = ["morning", "afternoon", "evening", "night"]
time_data_dict = {row['time_segment']: row for row in time_conversion.collect()}
best_time = time_conversion.orderBy(col("conversion_rate").desc()).first()
print(f"Best time: {best_time['time_segment'].upper()} ({best_time['conversion_rate']:.2f}%)\n")

# Top brands
top_brands = df.filter(col("event_type") == "purchase") \
               .groupBy("brand").agg(sum("price").alias("revenue"), 
                                     count("*").alias("num_sales"),
                                     avg("price").alias("avg_sale")) \
               .orderBy(col("revenue").desc()).limit(10)
top_brand = top_brands.first()
print(f"Top Brand: {top_brand.brand} - ${top_brand.revenue:,.2f}\n")

# SECTION 3: CUSTOMER SEGMENTATION
print("SECTION 3: CUSTOMER SEGMENTATION")
print("-" * 100)
max_date_in_data = df.agg({"event_date": "max"}).collect()[0][0]
rfm_data = df.filter(col("event_type") == "purchase") \
             .groupBy("user_id").agg(
                 datediff(lit(max_date_in_data), max("event_date")).alias("recency"),
                 count("*").alias("frequency"),
                 sum("price").alias("monetary"))
rfm_data = rfm_data.withColumn("segment",
    when((col("recency") <= 7) & (col("frequency") >= 3) & (col("monetary") >= 500), "VIP")
    .when((col("recency") <= 15) & (col("frequency") >= 2), "Loyal")
    .when(col("recency") <= 30, "Active")
    .otherwise("At-Risk"))

segment_stats = rfm_data.groupBy("segment").agg(
    count("*").alias("count"),
    avg("monetary").alias("avg_spend"),
    avg("frequency").alias("avg_purchases"),
    avg("recency").alias("avg_recency")).orderBy(col("avg_spend").desc())

print("Customer Segments:")
for row in segment_stats.collect():
    print(f"{row.segment}: {row['count']:,} customers, Avg spend: ${row.avg_spend:,.2f}, "
          f"Avg purchases: {row.avg_purchases:.1f}, Avg recency: {row.avg_recency:.0f} days")
print()

# SECTION 4: RDD OPERATIONS
print("SECTION 4: RDD OPERATIONS")
print("-" * 100)
rdd = df.rdd.repartition(4)
purchases_rdd = rdd.filter(lambda x: x.event_type == 'purchase')
brand_price = purchases_rdd.map(lambda x: (x.brand, x.price))
brand_revenue = brand_price.reduceByKey(lambda a, b: a + b)
top_brands_rdd = brand_revenue.sortBy(lambda x: x[1], ascending=False)
unique_products = rdd.map(lambda x: x.product_id).distinct()
event_distribution = rdd.map(lambda x: x.event_type).countByValue()

print(f"Purchase events: {purchases_rdd.count():,}")
print(f"Unique products: {unique_products.count():,}")
print(f"Total events: {rdd.count():,}")
print(f"Event breakdown: Views={event_distribution.get('view',0):,}, "
      f"Cart={event_distribution.get('cart',0):,}, Purchase={event_distribution.get('purchase',0):,}\n")

# SECTION 5: MACHINE LEARNING
print("SECTION 5: MACHINE LEARNING")
print("-" * 100)
user_features = df.groupBy("user_id").agg(
    count("*").alias("total_events"),
    countDistinct("product_id").alias("unique_products"),
    sum(when(col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
    sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("cart_count"),
    sum(when(col("is_weekend") == 1, 1).otherwise(0)).alias("weekend_activity"),
    avg("price").alias("avg_price"),
    max("price").alias("max_price"),
    # FIXED LABEL: actual purchase events
    sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count")
).fillna(0.0)

user_features = user_features.withColumn("purchased", when(col("purchase_count") > 0, 1).otherwise(0))
user_features = user_features.withColumn("engagement_score", col("view_count") + col("cart_count") * 2) \
                             .withColumn("cart_rate", col("cart_count") / (col("view_count") + 1))

feature_cols = ["total_events", "unique_products", "view_count", "cart_count", 
                "weekend_activity", "avg_price", "max_price", "engagement_score", "cart_rate"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
ml_data = assembler.transform(user_features).select("features", "purchased")
train, test = ml_data.randomSplit([0.8, 0.2], seed=42)

print(f"Training set: {train.count():,}")
print(f"Test set: {test.count():,}\n")

rf_model = RandomForestClassifier(labelCol="purchased", featuresCol="features", seed=42)
param_grid = ParamGridBuilder().addGrid(rf_model.numTrees, [20, 30]).addGrid(rf_model.maxDepth, [6, 8]).build()
evaluator = BinaryClassificationEvaluator(labelCol="purchased", metricName="areaUnderROC")
cv = CrossValidator(estimator=rf_model, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3, seed=42)

print("Training with cross-validation...")
cv_model = cv.fit(train)
best_model = cv_model.bestModel
predictions = cv_model.transform(test)
auc_score = evaluator.evaluate(predictions)

pred_pandas = predictions.select("purchased", "prediction").toPandas()
y_true = pred_pandas["purchased"].values
y_pred = pred_pandas["prediction"].values
tn = np.sum((y_true == 0) & (y_pred == 0))
fp = np.sum((y_true == 0) & (y_pred == 1))
fn = np.sum((y_true == 1) & (y_pred == 0))
tp = np.sum((y_true == 1) & (y_pred == 1))
total = tp + tn + fp + fn
accuracy = (tp + tn) / total if total > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nAUC-ROC: {auc_score:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}\n")

importances = sorted(zip(feature_cols, best_model.featureImportances.toArray()), key=lambda x: x[1], reverse=True)
print("Feature Importance:")
for i, (feat, imp) in enumerate(importances[:5], 1):
    print(f"  {i}. {feat}: {imp:.1%}")
print()

# SECTION 6: VISUALIZATION
print("SECTION 6: VISUALIZATION")
print("-" * 100)
try:
    brands_data = top_brands.toPandas()
    segments_data = segment_stats.toPandas()

    time_rows = []
    for t in all_times:
        if t in time_data_dict:
            time_rows.append({'time_segment': t, 'conversion_rate': time_data_dict[t]['conversion_rate']})
        else:
            time_rows.append({'time_segment': t, 'conversion_rate': 0})
    time_data = pd.DataFrame(time_rows)

    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    fig.suptitle('E-COMMERCE BIG DATA ANALYSIS DASHBOARD', fontsize=16, fontweight='bold')

    # Plot 1: Conversion Funnel
    ax1 = fig.add_subplot(gs[0, 0])
    funnel = [view_cnt, cart_cnt, purch_cnt]
    labels = ['Views', 'Cart', 'Purchases']
    bars1 = ax1.bar(labels, funnel, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', linewidth=1.5)
    ax1.set_title('Conversion Funnel', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Count')
    ax1.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars1, funnel):
        ax1.text(bar.get_x() + bar.get_width()/2, v, f'{v:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: Top Brands
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(brands_data['brand'][::-1][:8], brands_data['revenue'][::-1][:8], color=plt.cm.viridis(np.linspace(0.2, 0.8, 8)))
    ax2.set_title('Top 8 Brands by Revenue', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Revenue ($)')
    ax2.grid(axis='x', alpha=0.3)

    # Plot 3: Customer Segments
    ax3 = fig.add_subplot(gs[0, 2])
    colors = {'VIP': '#2ca02c', 'Loyal': '#1f77b4', 'Active': '#ff7f0e', 'At-Risk': '#d62728'}
    wedges, _, autotexts = ax3.pie(segments_data['count'], labels=segments_data['segment'],
                                   autopct='%1.1f%%', colors=[colors.get(x, '#aaa') for x in segments_data['segment']],
                                   startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax3.set_title('Customer Segments (RFM)', fontweight='bold', fontsize=12)

    # Remove evening if it has zero data
    time_data = time_data[time_data["time_segment"] != "evening"]
    # Plot 4: Time Conversion
    ax4 = fig.add_subplot(gs[1, 0])
    bars4 = ax4.bar(time_data['time_segment'], time_data['conversion_rate'], color='#9467bd', edgecolor='black', linewidth=1.5)
    ax4.set_title('Conversion Rate by Time', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Conversion Rate (%)')
    ax4.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars4, time_data['conversion_rate']):
        if v > 0:
            ax4.text(bar.get_x()+bar.get_width()/2, v, f'{v:.2f}%', ha='center', va='bottom', fontsize=8)

    # Plot 5: Event Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    event_types = ['view', 'cart', 'purchase']
    counts = [event_distribution.get(e, 0) for e in event_types]
    ax5.pie(counts, labels=[f'{e}\n({c:,})' for e,c in zip(event_types, counts)], autopct='%1.1f%%',
            colors=['#1f77b4', '#ff7f0e', '#2ca02c'], startangle=90)
    ax5.set_title('Event Distribution', fontweight='bold', fontsize=12)

    # Plot 6: Price Categories
    ax6 = fig.add_subplot(gs[1, 2])
    price_data = df.filter(col("event_type")=="purchase").groupBy("price_category").agg(sum("price").alias("revenue")).toPandas()
    bars6 = ax6.bar(price_data['price_category'], price_data['revenue'], color=['#2ca02c','#ff7f0e','#1f77b4'])
    ax6.set_title('Revenue by Price Category', fontweight='bold', fontsize=12)
    ax6.grid(axis='y', alpha=0.3)
    for bar,v in zip(bars6, price_data['revenue']):
        ax6.text(bar.get_x()+bar.get_width()/2, v, f'${v/1e6:.1f}M', ha='center', va='bottom', fontsize=8)

    # Plot 7: Feature Importance
    ax7 = fig.add_subplot(gs[2, 0:2])
    top_feats = importances[:6]
    names = [f[0] for f in top_feats]
    values = [f[1] for f in top_feats]
    bars7 = ax7.barh(names, values, color=plt.cm.RdYlGn(np.linspace(0.3,0.9,6)))
    ax7.set_title('ML Feature Importance (Top 6)', fontweight='bold', fontsize=12)
    ax7.set_xlabel('Importance')
    for bar,v in zip(bars7, values):
        ax7.text(v, bar.get_y()+bar.get_height()/2, f'{v:.1%}', va='center', fontsize=9)

    # Plot 8: Metrics Summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    summary = f"""MODEL PERFORMANCE

Cross-Validation: 3-fold
Models Trained: 12

Metrics:
  AUC-ROC:    {auc_score:.4f}
  Accuracy:   {accuracy:.4f}
  Precision:  {precision:.4f}
  Recall:     {recall:.4f}
  F1-Score:   {f1:.4f}

Dataset:
  Records:    {total_records:,}
  Time:       {(time.time()-start_time)/60:.1f} min
"""
    ax8.text(0.1, 0.9, summary, transform=ax8.transAxes, fontsize=10, verticalalignment='top',
             family='monospace', bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black', alpha=0.8))

    plt.savefig('ecommerce_analysis_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Dashboard saved: ecommerce_analysis_dashboard.png (300 DPI)\n")
except Exception as e:
    print(f"Visualization error: {e}")

# EXECUTION SUMMARY
execution_time = time.time() - start_time
print("="*100)
print("EXECUTION SUMMARY")
print("="*100)
print(f"Total time: {execution_time:.1f}s ({execution_time/60:.2f} minutes)")
print(f"Records: {total_records:,}")
print(f"Models trained: 12 (cross-validation)")
print(f"Dashboard: ecommerce_analysis_dashboard.png (8 panels, 300 DPI)")
print("="*100 + "\n")

spark.stop()
