"""
Unified E-Commerce Analysis: DataFrame + RDD Operations + Machine Learning
Complete Big Data Analytics Pipeline
Authors: Arnav Devdatt Naik, Gargi Pant
Date: October 12, 2025
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark import StorageLevel
import matplotlib.pyplot as plt
import subprocess
import time

# ============================================================================
# INITIALIZATION
# ============================================================================
spark = SparkSession.builder \
    .appName("Unified ECommerce Analysis") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

sc = spark.sparkContext

def print_section(title, char="="):
    print(f"\n{char*100}\n{title}\n{char*100}")

def print_result(label, value):
    print(f"  > {label}: {value}")

start_time = time.time()
print_section("UNIFIED E-COMMERCE BIG DATA ANALYSIS PIPELINE")
print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Spark Version: {spark.version}")

# ============================================================================
# PART 1: DATAFRAME API - DATA LOADING & FEATURE ENGINEERING
# ============================================================================
print_section("PART 1: DATAFRAME API - ENHANCED DATA PROCESSING")

print("\n[PHASE 1.1] Loading and Feature Engineering")
df = spark.read.csv("hdfs:///ecommerce/ecommerce_sample.csv", header=True, inferSchema=True) \
    .filter(col("user_id").isNotNull() & col("product_id").isNotNull()) \
    .fillna({"brand": "unknown", "category_code": "unknown", "price": 0.0}) \
    .withColumn("event_hour", hour("event_time")) \
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
print_result("Records Loaded", f"{total_records:,}")
print_result("Enhanced Features Added", "time_segment, is_weekend, price_category")

# ============================================================================
# PART 2: DATAFRAME API - BEHAVIORAL ANALYTICS
# ============================================================================
print_section("PART 2: DATAFRAME API - BEHAVIORAL METRICS")

print("\n[PHASE 2.1] Conversion Funnel Analysis")
metrics = df.groupBy("event_type").count().collect()
view_cnt = next((m['count'] for m in metrics if m.event_type == 'view'), 0)
cart_cnt = next((m['count'] for m in metrics if m.event_type == 'cart'), 0)
purch_cnt = next((m['count'] for m in metrics if m.event_type == 'purchase'), 0)

print(f"  Views: {view_cnt:,} -> Carts: {cart_cnt:,} ({cart_cnt/view_cnt*100:.2f}%)")
print(f"  -> Purchases: {purch_cnt:,} ({purch_cnt/cart_cnt*100:.2f}%)")
print_result("Overall Conversion", f"{purch_cnt/view_cnt*100:.2f}%")

print("\n[PHASE 2.2] Time-Based Performance")
time_conversion = df.groupBy("time_segment", "event_type").count() \
    .groupBy("time_segment").pivot("event_type").sum("count").fillna(0) \
    .withColumn("conversion_rate", col("purchase") / col("view") * 100)
best_time = time_conversion.orderBy(col("conversion_rate").desc()).first()
print_result("Best Conversion Period", f"{best_time['time_segment'].upper()} ({best_time['conversion_rate']:.2f}%)")

print("\n[PHASE 2.3] Price Sensitivity Analysis")
price_analysis = df.filter(col("event_type") == "purchase") \
    .groupBy("price_category").agg(
        count("*").alias("purchases"),
        sum("price").alias("revenue")
    ).orderBy(col("revenue").desc())

print("  Revenue by Category:")
for row in price_analysis.collect():
    print(f"    {row.price_category:12s}: ${row.revenue:>12,.0f} ({row.purchases:,} sales)")

print("\n[PHASE 2.4] Top Performers")
top_brands_df = df.filter(col("event_type") == "purchase") \
    .groupBy("brand").agg(
        sum("price").alias("revenue"),
        count("*").alias("purchases")
    ).orderBy(col("revenue").desc()).limit(10)

top_brand = top_brands_df.first()
print_result("Leading Brand", f"{top_brand.brand} - ${top_brand.revenue:,.0f} ({top_brand.purchases:,} sales)")

# ============================================================================
# PART 3: DATAFRAME API - CUSTOMER SEGMENTATION (RFM)
# ============================================================================
print_section("PART 3: DATAFRAME API - CUSTOMER SEGMENTATION")

print("\n[PHASE 3.1] RFM Analysis")
window_spec = Window.partitionBy("user_id").orderBy(col("event_time").desc())
rfm_data = df.filter(col("event_type") == "purchase") \
    .withColumn("row_num", row_number().over(window_spec)) \
    .filter(col("row_num") == 1) \
    .groupBy("user_id").agg(
        datediff(current_date(), max("event_date")).alias("recency"),
        count("*").alias("frequency"),
        sum("price").alias("monetary")
    ) \
    .withColumn("customer_segment",
        when((col("recency") <= 30) & (col("frequency") >= 3) & (col("monetary") >= 500), "VIP")
        .when((col("recency") <= 60) & (col("frequency") >= 2), "Loyal")
        .when(col("recency") <= 90, "Active")
        .otherwise("At-Risk"))

segment_summary = rfm_data.groupBy("customer_segment").agg(
    count("*").alias("customers"),
    avg("monetary").alias("avg_spend")
).orderBy(col("avg_spend").desc())

print("  Customer Segments:")
for row in segment_summary.collect():
    print(f"    {row.customer_segment:10s}: {row.customers:>6,} customers (avg ${row.avg_spend:,.2f})")

# ============================================================================
# PART 4: RDD API - CORE TRANSFORMATIONS & ACTIONS
# ============================================================================
print_section("PART 4: RDD API - TRANSFORMATIONS & ACTIONS")

print("\n[PHASE 4.1] Converting to RDD for Low-Level Operations")
rdd = df.rdd.repartition(8).persist(StorageLevel.MEMORY_AND_DISK)
print_result("RDD Partitions", rdd.getNumPartitions())
print_result("Storage Level", "MEMORY_AND_DISK")

print("\n[PHASE 4.2] RDD Transformations (12 Operations)")

# Transform 1-3: Filter by event type
purchases_rdd = rdd.filter(lambda x: x.event_type == 'purchase').cache()
views_rdd = rdd.filter(lambda x: x.event_type == 'view').cache()
carts_rdd = rdd.filter(lambda x: x.event_type == 'cart')
print_result("1-3. Filter", f"Separated {purchases_rdd.count():,} purchases, {views_rdd.count():,} views")

# Transform 4-5: Map + ReduceByKey
brand_revenue_rdd = purchases_rdd \
    .map(lambda x: (x.brand, x.price)) \
    .reduceByKey(lambda a, b: a + b)
print_result("4-5. Map + ReduceByKey", f"{brand_revenue_rdd.count()} brands aggregated")

# Transform 6: SortBy
top_brands_rdd = brand_revenue_rdd.sortBy(lambda x: x[1], ascending=False)
top_3 = top_brands_rdd.take(3)
print_result("6. SortBy", f"Top brand: {top_3[0][0]} (${top_3[0][1]:,.0f})")

# Transform 7: Distinct
unique_products = rdd.map(lambda x: x.product_id).distinct()
unique_users = rdd.map(lambda x: x.user_id).distinct()
print_result("7. Distinct", f"{unique_products.count():,} products, {unique_users.count():,} users")

# Transform 8-9: GroupByKey + MapValues
user_purchase_stats = purchases_rdd \
    .map(lambda x: (x.user_id, x.price)) \
    .groupByKey() \
    .mapValues(lambda prices: (len(list(prices)), sum(prices)))
print_result("8-9. GroupByKey + MapValues", f"{user_purchase_stats.count():,} user purchase profiles")

# Transform 10: Join
user_view_counts = views_rdd.map(lambda x: (x.user_id, 1)).reduceByKey(lambda a, b: a + b)
user_purch_counts = purchases_rdd.map(lambda x: (x.user_id, 1)).reduceByKey(lambda a, b: a + b)
user_behavior_rdd = user_view_counts.join(user_purch_counts)
avg_conversion = user_behavior_rdd.map(lambda x: x[1][1] / x[1][0] * 100).mean()
print_result("10. Join", f"{user_behavior_rdd.count():,} users analyzed (avg {avg_conversion:.2f}% conversion)")

# Transform 11: FlatMap
user_brands = purchases_rdd \
    .map(lambda x: (x.user_id, [x.brand])) \
    .reduceByKey(lambda a, b: a + b) \
    .flatMapValues(lambda brands: set(brands))
print_result("11. FlatMap", f"{user_brands.count():,} user-brand relationships")

# Transform 12: Union
high_value = purchases_rdd.filter(lambda x: x.price > 200)
popular_products = views_rdd.map(lambda x: x.product_id).distinct()
all_products = high_value.map(lambda x: x.product_id).union(popular_products).distinct()
print_result("12. Union", f"{all_products.count():,} unique products (high-value + popular)")

print("\n[PHASE 4.3] RDD Actions (10 Operations)")

# Action 1: Count
print_result("1. Count", f"{rdd.count():,} total records")

# Action 2-3: Take & First
print_result("2. Take", f"Retrieved {len(top_brands_rdd.take(10))} top brands")
print_result("3. First", f"Top brand: {top_brands_rdd.first()}")

# Action 4: Reduce
total_revenue = purchases_rdd.map(lambda x: x.price).reduce(lambda a, b: a + b)
print_result("4. Reduce", f"Total revenue: ${total_revenue:,.2f}")

# Action 5: CountByValue
event_dist = rdd.map(lambda x: x.event_type).countByValue()
print_result("5. CountByValue", f"Event distribution: {dict(event_dist)}")

# Action 6: Aggregate
revenue_stats = brand_revenue_rdd.aggregate(
    (0, 0.0, float('inf'), 0.0),
    lambda acc, x: (acc[0] + 1, acc[1] + x[1], min(acc[2], x[1]), max(acc[3], x[1])),
    lambda a1, a2: (a1[0] + a2[0], a1[1] + a2[1], min(a1[2], a2[2]), max(a1[3], a2[3]))
)
print_result("6. Aggregate", f"Brands={revenue_stats[0]}, Avg=${revenue_stats[1]/revenue_stats[0]:,.2f}")

# Action 7: Fold
total_items = rdd.map(lambda x: 1).fold(0, lambda a, b: a + b)
print_result("7. Fold", f"{total_items:,} items processed")

# Action 8: Foreach with Accumulator
high_value_acc = sc.accumulator(0)
purchases_rdd.foreach(lambda x: high_value_acc.add(1) if x.price > 200 else None)
print_result("8. Foreach", f"{high_value_acc.value} high-value purchases (>$200)")

# Action 9: TakeSample
sample = purchases_rdd.takeSample(False, 3, seed=42)
print_result("9. TakeSample", f"Sampled 3 random purchases")

# Action 10: SaveAsTextFile
output_path = "hdfs:///ecommerce/unified_analysis_results"
subprocess.run(["hdfs", "dfs", "-rm", "-r", output_path], 
               stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
top_brands_rdd.coalesce(1).saveAsTextFile(f"{output_path}/top_brands")
user_behavior_rdd.coalesce(1).saveAsTextFile(f"{output_path}/user_behavior")
print_result("10. SaveAsTextFile", f"Results saved to {output_path}")

# ============================================================================
# PART 5: RDD PERFORMANCE OPTIMIZATION
# ============================================================================
print_section("PART 5: RDD PERFORMANCE OPTIMIZATION")

print("\n[PHASE 5.1] Caching Performance Test")
test_rdd = rdd.map(lambda x: (x.event_type, 1)).reduceByKey(lambda a, b: a + b)

t1 = time.time()
test_rdd.count()
uncached_time = time.time() - t1
print_result("Uncached Operation", f"{uncached_time:.3f}s")

cached_rdd = test_rdd.cache()
cached_rdd.count()

t2 = time.time()
cached_rdd.count()
cached_time = time.time() - t2
print_result("Cached Operation", f"{cached_time:.3f}s")
print_result("Speedup", f"{uncached_time/cached_time:.1f}x faster with caching")

print("\n[PHASE 5.2] Market Share Analysis")
total_brand_rev = brand_revenue_rdd.map(lambda x: x[1]).sum()
brand_share = brand_revenue_rdd.map(lambda x: (x[0], x[1], x[1]/total_brand_rev*100)) \
    .sortBy(lambda x: x[2], ascending=False)

print("  Top 5 Brands by Market Share:")
for i, (brand, rev, share) in enumerate(brand_share.take(5), 1):
    print(f"    {i}. {brand:15s} ${rev:>10,.0f} ({share:5.1f}%)")

# ============================================================================
# PART 6: MACHINE LEARNING - PURCHASE PREDICTION
# ============================================================================
print_section("PART 6: MACHINE LEARNING - PURCHASE PREDICTION")

print("\n[PHASE 6.1] Feature Engineering for ML")
user_features = df.groupBy("user_id").agg(
    count("*").alias("total_events"),
    countDistinct("product_id").alias("unique_products"),
    countDistinct("brand").alias("unique_brands"),
    sum(when(col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
    sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("cart_count"),
    sum(when(col("is_weekend") == 1, 1).otherwise(0)).alias("weekend_activity"),
    avg("price").alias("avg_price_viewed"),
    max("price").alias("max_price_viewed"),
    countDistinct("event_date").alias("active_days")
).fillna(0.0) \
    .withColumn("engagement_score", col("view_count") + col("cart_count") * 2) \
    .withColumn("cart_rate", col("cart_count") / (col("view_count") + 1)) \
    .withColumn("purchased", when(col("cart_count") > 0, 1).otherwise(0))

print_result("Features Created", "11 features (total_events, view_count, cart_count, etc.)")
print_result("Training Samples", f"{user_features.count():,}")

print("\n[PHASE 6.2] Model Training & Comparison")
feature_cols = ["total_events", "unique_products", "unique_brands", "view_count", 
                "cart_count", "weekend_activity", "avg_price_viewed", "max_price_viewed",
                "active_days", "engagement_score", "cart_rate"]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
ml_data = assembler.transform(user_features).select("features", "purchased")
train, test = ml_data.randomSplit([0.8, 0.2], seed=42)

# Random Forest with Cross-Validation
rf = RandomForestClassifier(labelCol="purchased", featuresCol="features", seed=42)
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [30, 50]) \
    .addGrid(rf.maxDepth, [8, 10]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol="purchased", metricName="areaUnderROC")
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

cv_model = cv.fit(train)
rf_predictions = cv_model.transform(test)
rf_auc = evaluator.evaluate(rf_predictions)

# Gradient Boosted Trees
gbt = GBTClassifier(labelCol="purchased", featuresCol="features", maxIter=20, seed=42)
gbt_model = gbt.fit(train)
gbt_predictions = gbt_model.transform(test)
gbt_auc = evaluator.evaluate(gbt_predictions)

print(f"  Random Forest AUC:  {rf_auc:.4f} {'[BEST]' if rf_auc > gbt_auc else ''}")
print(f"  Gradient Boost AUC: {gbt_auc:.4f} {'[BEST]' if gbt_auc >= rf_auc else ''}")

best_model = cv_model.bestModel
importances = sorted(zip(feature_cols, best_model.featureImportances.toArray()), 
                    key=lambda x: x[1], reverse=True)

print("\n  Top 5 Feature Importance:")
for i, (feat, imp) in enumerate(importances[:5], 1):
    print(f"    {i}. {feat:20s}: {imp:6.1%}")

# ============================================================================
# PART 7: COMPREHENSIVE VISUALIZATION
# ============================================================================
print_section("PART 7: UNIFIED VISUALIZATION DASHBOARD")

# Convert to pandas
brands_pd = top_brands_df.toPandas()
price_pd = price_analysis.toPandas()
segments_pd = segment_summary.toPandas()
time_conv_pd = time_conversion.toPandas()

# RDD data
brands_rdd_list = top_brands_rdd.take(10)
rdd_brands = [b[0][:12] for b in brands_rdd_list]
rdd_revenues = [b[1] for b in brands_rdd_list]
market_share_data = brand_share.take(5)

# Create comprehensive dashboard
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

# Row 1: DataFrame Visualizations
# 1.1 Conversion Funnel
ax1 = fig.add_subplot(gs[0, 0])
funnel = [view_cnt, cart_cnt, purch_cnt]
labels = ['Views', 'Carts', 'Purchases']
colors = ['#3498db', '#f39c12', '#2ecc71']
bars = ax1.bar(labels, funnel, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_title('Conversion Funnel (DataFrame)', fontweight='bold', size=11)
ax1.set_ylabel('Count')
for bar, val in zip(bars, funnel):
    ax1.text(bar.get_x() + bar.get_width()/2, val, f'{val:,}', 
            ha='center', va='bottom', fontweight='bold', fontsize=9)

# 1.2 Top Brands (DataFrame)
ax2 = fig.add_subplot(gs[0, 1])
ax2.barh(brands_pd['brand'][:8][::-1], brands_pd['revenue'][:8][::-1], 
         color=plt.cm.viridis(range(8)))
ax2.set_title('Top Brands - DataFrame', fontweight='bold', size=11)
ax2.set_xlabel('Revenue ($)')

# 1.3 Customer Segments
ax3 = fig.add_subplot(gs[0, 2])
colors_seg = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
ax3.pie(segments_pd['customers'], labels=segments_pd['customer_segment'],
        autopct='%1.1f%%', colors=colors_seg, startangle=90, textprops={'fontsize': 9})
ax3.set_title('Customer Segmentation (RFM)', fontweight='bold', size=11)

# 1.4 Time-Based Conversion
ax4 = fig.add_subplot(gs[0, 3])
time_conv_sorted = time_conv_pd.sort_values('conversion_rate', ascending=False)
ax4.bar(time_conv_sorted['time_segment'], time_conv_sorted['conversion_rate'], 
        color='#9b59b6', edgecolor='black')
ax4.set_title('Conversion by Time', fontweight='bold', size=11)
ax4.set_ylabel('Conversion Rate (%)')

# Row 2: RDD Visualizations
# 2.1 Top Brands (RDD)
ax5 = fig.add_subplot(gs[1, 0])
colors_rdd = plt.cm.plasma(range(len(rdd_brands)))
ax5.barh(range(len(rdd_brands)), rdd_revenues, color=colors_rdd)
ax5.set_yticks(range(len(rdd_brands)))
ax5.set_yticklabels(rdd_brands)
ax5.invert_yaxis()
ax5.set_title('Top Brands - RDD', fontweight='bold', size=11)
ax5.set_xlabel('Revenue ($)')

# 2.2 Market Share
ax6 = fig.add_subplot(gs[1, 1])
share_brands = [m[0][:10] for m in market_share_data]
share_pcts = [m[2] for m in market_share_data]
colors_share = plt.cm.Set3(range(len(share_brands)))
ax6.pie(share_pcts, labels=share_brands, autopct='%1.1f%%', colors=colors_share,
        textprops={'fontsize': 9})
ax6.set_title('Market Share (Top 5)', fontweight='bold', size=11)

# 2.3 Event Distribution
ax7 = fig.add_subplot(gs[1, 2])
events = list(event_dist.keys())
counts = [event_dist[e] for e in events]
ax7.pie(counts, labels=events, autopct='%1.1f%%', 
        colors=['#3498db', '#e74c3c', '#2ecc71'], startangle=90,
        textprops={'fontsize': 10})
ax7.set_title('Event Distribution (RDD)', fontweight='bold', size=11)

# 2.4 Price Categories
ax8 = fig.add_subplot(gs[1, 3])
ax8.bar(price_pd['price_category'], price_pd['revenue'], 
        color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
ax8.set_title('Revenue by Price', fontweight='bold', size=11)
ax8.set_ylabel('Revenue ($)')

# Row 3: ML and Combined Insights
# 3.1 Feature Importance
ax9 = fig.add_subplot(gs[2, 0:2])
top_5_feats = importances[:5]
feat_names = [f[0] for f in top_5_feats][::-1]
feat_imps = [f[1] for f in top_5_feats][::-1]
colors_feat = plt.cm.RdYlGn(range(len(feat_names)))
bars9 = ax9.barh(feat_names, feat_imps, color=colors_feat, edgecolor='black')
ax9.set_title('ML Feature Importance (Top 5)', fontweight='bold', size=11)
ax9.set_xlabel('Importance')
for bar, val in zip(bars9, feat_imps):
    ax9.text(val, bar.get_y() + bar.get_height()/2, f'{val:.1%}', 
            va='center', fontweight='bold', fontsize=9)

# 3.2 Performance Metrics Summary
ax10 = fig.add_subplot(gs[2, 2:])
ax10.axis('off')
summary_text = f"""
+-------------------------------------------------------+
|          UNIFIED ANALYSIS SUMMARY                     |
+-------------------------------------------------------+
|  DATAFRAME API                                        |
|    Records Processed: {total_records:,}               |
|    Conversion Rate: {purch_cnt/view_cnt*100:.2f}%     |
|    Best Time: {best_time['time_segment'].upper()}     |
|    Customer Segments: 4 (VIP/Loyal/Active/At-Risk)    |
|                                                       |
|  RDD API                                              |
|    Transformations: 12 operations                     |
|    Actions: 10 operations                             |
|    Cache Speedup: {uncached_time/cached_time:.1f}x    |
|    Market Leader: {market_share_data[0][0]} ({market_share_data[0][2]:.1f}%)  |
|                                                       |
|  MACHINE LEARNING                                     |
|    Best Model: {'Random Forest' if rf_auc > gbt_auc else 'Gradient Boost'}  |
|    AUC Score: {max(rf_auc, gbt_auc):.4f}              |
|    Top Feature: {importances[0][0]} ({importances[0][1]:.1%})  |
|    Features Used: 11                                  |
+-------------------------------------------------------+
"""
ax10.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.3))

plt.suptitle('Unified E-Commerce Big Data Analysis Dashboard', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('unified_ecommerce_analysis.png', dpi=300, bbox_inches='tight')
print_result("Dashboard Saved", "unified_ecommerce_analysis.png (300 DPI)")

# ============================================================================
# PART 8: FINAL SUMMARY & RECOMMENDATIONS
# ============================================================================
print_section("PART 8: ACTIONABLE BUSINESS INSIGHTS")

print("\nKEY RECOMMENDATIONS:")
print(f"\n1. CONVERSION OPTIMIZATION:")
print(f"   - Target {best_time['time_segment']} hours (best {best_time['conversion_rate']:.2f}% conversion)")
print(f"   - Address {(cart_cnt-purch_cnt)/cart_cnt*100:.1f}% cart abandonment rate")

print(f"\n2. CUSTOMER STRATEGY:")
vip_count = segments_pd[segments_pd['customer_segment'] == 'VIP']['customers'].values[0] if 'VIP' in segments_pd['customer_segment'].values else 0
print(f"   - Focus on {vip_count:,} VIP customers for premium campaigns")
print(f"   - Re-engage At-Risk segment with targeted offers")

print(f"\n3. PRODUCT & INVENTORY:")
print(f"   - Prioritize {top_brand.brand} inventory (market leader)")
print(f"   - Optimize {price_pd.iloc[0]['price_category']} category stock")

print(f"\n4. TECHNICAL INSIGHTS:")
print(f"   - RDD caching provides {uncached_time/cached_time:.1f}x performance boost")
print(f"   - ML prediction accuracy: {max(rf_auc, gbt_auc):.1%} - use for targeting")
print(f"   - {importances[0][0]} is strongest purchase predictor")

print_section("ANALYSIS COMPLETE")
execution_time = time.time() - start_time
print(f"\nOUTPUTS GENERATED:")
print_result("HDFS Results", output_path)
print_result("Visualization", "unified_ecommerce_analysis.png")
print_result("Total Execution Time", f"{execution_time:.1f}s ({execution_time/60:.1f} minutes)")

print(f"\nDEMONSTRATED CONCEPTS:")
print(f"  - DataFrame API: Data loading, feature engineering, aggregations")
print(f"  - RDD API: 12 transformations + 10 actions")
print(f"  - Performance: Caching, partitioning, optimization")
print(f"  - Machine Learning: 2 models, cross-validation, feature importance")
print(f"  - Business Analytics: RFM segmentation, conversion analysis")
print(f"  - Visualization: 10-panel comprehensive dashboard")

print(f"\n{'='*100}")
print("End Time:", time.strftime('%Y-%m-%d %H:%M:%S'))
print('='*100 + "\n")

spark.stop()
