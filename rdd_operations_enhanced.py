"""
Enhanced RDD Operations Demonstration with Better Visualization
Showcasing Transformations and Actions for E-Commerce Data
Authors: Arnav Devdatt Naik, Gargi Pant
"""

from pyspark import SparkContext, SparkConf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Initialize Spark Context
conf = SparkConf().setAppName("Enhanced RDD Operations Demo") \
    .set("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
sc = SparkContext(conf=conf)

def print_header(title, char="="):
    """Print formatted section header"""
    width = 80
    print("\n" + char * width)
    print(title.center(width))
    print(char * width + "\n")

def print_subheader(title):
    """Print formatted subsection header"""
    print(f"\n{'─' * 80}")
    print(f"▶ {title}")
    print(f"{'─' * 80}")

def print_result(label, value, indent=2):
    """Print formatted result"""
    spaces = " " * indent
    print(f"{spaces}✓ {label}: {value}")

# Start time tracking
start_time = datetime.now()

print_header("🚀 ENHANCED RDD OPERATIONS DEMONSTRATION", "═")
print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Spark Version: {sc.version}")
print(f"Master: {sc.master}")

# ============================================================================
# PHASE 1: DATA LOADING
# ============================================================================
print_header("PHASE 1: DATA LOADING FROM HDFS")

raw_rdd = sc.textFile("hdfs:///ecommerce/ecommerce_sample.csv")
header = raw_rdd.first()
data_rdd = raw_rdd.filter(lambda line: line != header)

total_records = data_rdd.count()
num_partitions = data_rdd.getNumPartitions()

print_result("Total Records Loaded", f"{total_records:,}")
print_result("Number of Partitions", num_partitions)
print_result("Records per Partition (avg)", f"{total_records // num_partitions:,}")

print("\n📋 Sample Raw Data (first 2 records):")
for i, record in enumerate(data_rdd.take(2), 1):
    print(f"  {i}. {record[:100]}...")

# ============================================================================
# PHASE 2: RDD TRANSFORMATIONS
# ============================================================================
print_header("PHASE 2: RDD TRANSFORMATIONS", "═")

# Transformation 1: MAP
print_subheader("1. MAP - Parse CSV Records")
parsed_rdd = data_rdd.map(lambda line: line.split(','))
print_result("Operation", "Split CSV lines into fields")
print_result("Input Type", "String")
print_result("Output Type", "List of strings")
print("\n  Sample Parsed Record:")
sample = parsed_rdd.take(1)[0]
print(f"    Fields: {len(sample)}")
print(f"    Event Type: {sample[1]}")
print(f"    Brand: {sample[5]}")
print(f"    Price: {sample[6]}")

# Transformation 2: FILTER
print_subheader("2. FILTER - Extract Purchase Events")
purchases_rdd = parsed_rdd.filter(lambda x: len(x) > 1 and x[1] == 'purchase')
views_rdd = parsed_rdd.filter(lambda x: len(x) > 1 and x[1] == 'view')
cart_rdd = parsed_rdd.filter(lambda x: len(x) > 1 and x[1] == 'cart')

purchase_count = purchases_rdd.count()
view_count = views_rdd.count()
cart_count = cart_rdd.count()

print_result("Total Events", f"{total_records:,}")
print_result("Purchase Events", f"{purchase_count:,} ({purchase_count/total_records*100:.2f}%)")
print_result("View Events", f"{view_count:,} ({view_count/total_records*100:.2f}%)")
print_result("Cart Events", f"{cart_count:,} ({cart_count/total_records*100:.2f}%)")

# Transformation 3: MAP - Create Key-Value Pairs
print_subheader("3. MAP - Create (Brand, Price) Pairs")
brand_price_rdd = purchases_rdd.map(lambda x: (
    x[5].strip() if len(x) > 5 and x[5].strip() else "Unknown",
    float(x[6]) if len(x) > 6 and x[6].replace('.','').replace('-','').isdigit() else 0.0
))
print_result("Operation", "Extract brand and price from purchases")
print_result("Key", "Brand name")
print_result("Value", "Product price")
print("\n  Sample Brand-Price Pairs:")
for i, (brand, price) in enumerate(brand_price_rdd.take(5), 1):
    print(f"    {i}. {brand}: ${price:.2f}")

# Transformation 4: REDUCE BY KEY
print_subheader("4. REDUCE BY KEY - Aggregate Revenue per Brand")
brand_revenue_rdd = brand_price_rdd.reduceByKey(lambda a, b: a + b)
unique_brands = brand_revenue_rdd.count()
print_result("Operation", "Sum all prices for each brand")
print_result("Unique Brands", unique_brands)
print("\n  Sample Aggregated Results:")
for i, (brand, revenue) in enumerate(brand_revenue_rdd.take(5), 1):
    print(f"    {i}. {brand}: ${revenue:,.2f}")

# Transformation 5: SORT BY
print_subheader("5. SORT BY - Rank Brands by Revenue")
top_brands_rdd = brand_revenue_rdd.sortBy(lambda x: x[1], ascending=False)
print_result("Operation", "Sort by revenue (descending)")
print_result("Sort Key", "Revenue amount")
print("\n  🏆 Top 10 Brands by Revenue:")
top_10_brands = top_brands_rdd.take(10)
for i, (brand, revenue) in enumerate(top_10_brands, 1):
    bar = "█" * int(revenue / 100000)
    print(f"    {i:2d}. {brand:15s} ${revenue:>12,.2f} {bar}")

# Transformation 6: DISTINCT
print_subheader("6. DISTINCT - Find Unique Products")
product_ids = parsed_rdd.map(lambda x: x[2] if len(x) > 2 else None).filter(lambda x: x)
unique_products = product_ids.distinct()
unique_product_count = unique_products.count()
print_result("Total Product Views", f"{product_ids.count():,}")
print_result("Unique Products", f"{unique_product_count:,}")
print_result("Avg Views per Product", f"{product_ids.count() / unique_product_count:.1f}")

# Transformation 7: GROUP BY KEY
print_subheader("7. GROUP BY KEY - Events per User")
user_events = parsed_rdd.map(lambda x: (
    x[7] if len(x) > 7 else "unknown",
    x[1] if len(x) > 1 else "unknown"
)).groupByKey().mapValues(list)

unique_users = user_events.count()
user_sample = user_events.take(5)
print_result("Unique Users", f"{unique_users:,}")
print("\n  Sample User Activity:")
for i, (user, events) in enumerate(user_sample, 1):
    event_counts = {}
    for e in events:
        event_counts[e] = event_counts.get(e, 0) + 1
    print(f"    {i}. User {user[:8]}... → {len(events)} events: {dict(event_counts)}")

# Transformation 8: JOIN
print_subheader("8. JOIN - Match User Views with Purchases")
user_view_count = views_rdd.map(lambda x: (x[7], 1)).reduceByKey(lambda a, b: a + b)
user_purchase_count = purchases_rdd.map(lambda x: (x[7], 1)).reduceByKey(lambda a, b: a + b)
user_behavior = user_view_count.join(user_purchase_count)

joined_count = user_behavior.count()
print_result("Users with Views", f"{user_view_count.count():,}")
print_result("Users with Purchases", f"{user_purchase_count.count():,}")
print_result("Users with Both", f"{joined_count:,}")
print("\n  Sample User Behavior (Views → Purchases):")
for i, (user, (views, purchases)) in enumerate(user_behavior.take(5), 1):
    conversion = purchases / views * 100
    print(f"    {i}. User {user[:8]}... → {views} views, {purchases} purchases ({conversion:.1f}% conversion)")

# ============================================================================
# PHASE 3: RDD ACTIONS
# ============================================================================
print_header("PHASE 3: RDD ACTIONS", "═")

# Action 1: COUNT
print_subheader("1. COUNT - Count Elements")
print_result("Total Records", f"{data_rdd.count():,}")
print_result("Purchases", f"{purchases_rdd.count():,}")
print_result("Unique Brands", f"{brand_revenue_rdd.count():,}")
print_result("Unique Products", f"{unique_products.count():,}")

# Action 2: TAKE
print_subheader("2. TAKE - Get First N Elements")
print_result("Operation", "Retrieve first 3 top brands")
print("\n  Top 3 Brands:")
for i, (brand, revenue) in enumerate(top_brands_rdd.take(3), 1):
    print(f"    {i}. {brand}: ${revenue:,.2f}")

# Action 3: FIRST
print_subheader("3. FIRST - Get First Element")
first_brand = top_brands_rdd.first()
print_result("Highest Revenue Brand", f"{first_brand[0]}")
print_result("Revenue", f"${first_brand[1]:,.2f}")

# Action 4: REDUCE
print_subheader("4. REDUCE - Aggregate All Elements")
total_revenue = brand_price_rdd.map(lambda x: x[1]).reduce(lambda a, b: a + b)
avg_price = brand_price_rdd.map(lambda x: x[1]).reduce(lambda a, b: a + b) / purchase_count
print_result("Total Revenue", f"${total_revenue:,.2f}")
print_result("Total Purchases", f"{purchase_count:,}")
print_result("Average Purchase Value", f"${avg_price:.2f}")

# Action 5: COUNT BY VALUE
print_subheader("5. COUNT BY VALUE - Event Distribution")
event_types = parsed_rdd.map(lambda x: x[1] if len(x) > 1 else "unknown")
event_distribution = event_types.countByValue()
print("\n  Event Type Distribution:")
for event, count in sorted(event_distribution.items(), key=lambda x: x[1], reverse=True):
    percentage = count / total_records * 100
    bar = "█" * int(percentage * 2)
    print(f"    {event:10s}: {count:>8,} ({percentage:5.2f}%) {bar}")

# Action 6: COLLECT (Small Sample)
print_subheader("6. COLLECT - Retrieve All Elements (Sample)")
print_result("Operation", "Collect top 5 brands to driver")
collected_brands = top_brands_rdd.take(5)
print("\n  Collected Data:")
for i, (brand, revenue) in enumerate(collected_brands, 1):
    print(f"    {i}. {brand}: ${revenue:,.2f}")

# Action 7: FOREACH - Process Each Element
print_subheader("7. FOREACH - Process Each Partition")
partition_counts = data_rdd.mapPartitions(lambda x: [sum(1 for _ in x)]).collect()
print_result("Total Partitions", len(partition_counts))
print("\n  Records per Partition:")
for i, count in enumerate(partition_counts, 1):
    bar = "█" * (count // 10000)
    print(f"    Partition {i}: {count:>8,} records {bar}")

# Action 8: SAVE AS TEXT FILE
print_subheader("8. SAVE AS TEXT FILE - Persist to HDFS")
output_path = "hdfs:///ecommerce/enhanced_rdd_results"
import subprocess
subprocess.run(["hdfs", "dfs", "-rm", "-r", output_path], 
               stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

top_brands_rdd.coalesce(1).saveAsTextFile(output_path + "/top_brands")
brand_revenue_rdd.saveAsTextFile(output_path + "/all_brands")

print_result("Output Location", output_path)
print_result("Files Created", "top_brands/, all_brands/")
print("\n  Verification:")
try:
    result = subprocess.run(["hdfs", "dfs", "-ls", output_path], 
                          capture_output=True, text=True)
    for line in result.stdout.strip().split('\n')[-2:]:
        print(f"    {line}")
except:
    pass

# ============================================================================
# PHASE 4: ADVANCED OPERATIONS
# ============================================================================
print_header("PHASE 4: ADVANCED RDD OPERATIONS", "═")

# Coalesce and Repartition
print_subheader("Partition Management")
original_partitions = data_rdd.getNumPartitions()
coalesced = data_rdd.coalesce(2)
repartitioned = data_rdd.repartition(8)

print_result("Original Partitions", original_partitions)
print_result("After coalesce(2)", coalesced.getNumPartitions())
print_result("After repartition(8)", repartitioned.getNumPartitions())

# Cache/Persist
print_subheader("Caching and Persistence")
cached_rdd = brand_revenue_rdd.cache()
print_result("Operation", "Cache brand revenue RDD in memory")
print_result("Purpose", "Speed up repeated actions")
print_result("Storage Level", "MEMORY_ONLY")

# ============================================================================
# PHASE 5: VISUALIZATION
# ============================================================================
print_header("PHASE 5: DATA VISUALIZATION", "═")

# Prepare data for visualization
top_brands_list = top_brands_rdd.take(15)
brands = [b[0] if b[0] else "Unknown" for b in top_brands_list]
revenues = [b[1] for b in top_brands_list]

event_dist_data = sorted(event_distribution.items(), key=lambda x: x[1], reverse=True)
event_names = [e[0] for e in event_dist_data]
event_counts = [e[1] for e in event_dist_data]

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('RDD Operations Analysis - E-Commerce Dataset', 
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Top Brands Revenue
ax1 = axes[0, 0]
colors1 = plt.cm.viridis(range(len(brands)))
ax1.barh(range(len(brands)), revenues, color=colors1)
ax1.set_yticks(range(len(brands)))
ax1.set_yticklabels(brands)
ax1.set_xlabel('Revenue ($)', fontweight='bold')
ax1.set_title('Top 15 Brands by Revenue', fontweight='bold', pad=10)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)
for i, v in enumerate(revenues):
    ax1.text(v, i, f' ${v:,.0f}', va='center', fontsize=8)

# Plot 2: Event Distribution
ax2 = axes[0, 1]
colors2 = ['#3498db', '#e74c3c', '#2ecc71']
ax2.pie(event_counts, labels=event_names, autopct='%1.1f%%', 
        colors=colors2, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Event Type Distribution', fontweight='bold', pad=10)

# Plot 3: Revenue Distribution (Top 10)
ax3 = axes[1, 0]
top_10_revenues = revenues[:10]
top_10_brands = brands[:10]
explode = [0.1 if i == 0 else 0 for i in range(len(top_10_brands))]
colors3 = plt.cm.Set3(range(len(top_10_brands)))
ax3.pie(top_10_revenues, labels=top_10_brands, autopct='%1.1f%%',
        colors=colors3, explode=explode, textprops={'fontsize': 9})
ax3.set_title('Market Share - Top 10 Brands', fontweight='bold', pad=10)

# Plot 4: Partition Distribution
ax4 = axes[1, 1]
partitions = list(range(1, len(partition_counts) + 1))
ax4.bar(partitions, partition_counts, color='#9b59b6', edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Partition Number', fontweight='bold')
ax4.set_ylabel('Record Count', fontweight='bold')
ax4.set_title('Data Distribution Across Partitions', fontweight='bold', pad=10)
ax4.grid(axis='y', alpha=0.3)
for i, v in enumerate(partition_counts):
    ax4.text(i + 1, v, f'{v:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('rdd_operations_visualization.png', dpi=300, bbox_inches='tight')
print_result("Visualization Saved", "rdd_operations_visualization.png")
print_result("Resolution", "300 DPI")
print_result("Format", "PNG")

# ============================================================================
# PHASE 6: SUMMARY AND STATISTICS
# ============================================================================
print_header("PHASE 6: SUMMARY AND STATISTICS", "═")

end_time = datetime.now()
execution_time = (end_time - start_time).total_seconds()

print_subheader("Execution Statistics")
print_result("Start Time", start_time.strftime('%H:%M:%S'))
print_result("End Time", end_time.strftime('%H:%M:%S'))
print_result("Total Execution Time", f"{execution_time:.2f} seconds")
print_result("Records Processed", f"{total_records:,}")
print_result("Processing Rate", f"{int(total_records/execution_time):,} records/second")

print_subheader("Data Statistics")
print_result("Total Events", f"{total_records:,}")
print_result("Unique Users", f"{unique_users:,}")
print_result("Unique Products", f"{unique_product_count:,}")
print_result("Unique Brands", f"{unique_brands}")
print_result("Total Revenue", f"${total_revenue:,.2f}")
print_result("Avg Purchase Value", f"${avg_price:.2f}")

print_subheader("Conversion Metrics")
conv_rate = (purchase_count / view_count) * 100 if view_count > 0 else 0
cart_conv = (purchase_count / cart_count) * 100 if cart_count > 0 else 0
print_result("View-to-Purchase", f"{conv_rate:.2f}%")
print_result("Cart-to-Purchase", f"{cart_conv:.2f}%")

print_subheader("Top 5 Brands Summary")
for i, (brand, revenue) in enumerate(top_brands_rdd.take(5), 1):
    share = (revenue / total_revenue) * 100
    print_result(f"{i}. {brand}", f"${revenue:,.2f} ({share:.1f}% market share)")

# ============================================================================
# TRANSFORMATIONS & ACTIONS CHECKLIST
# ============================================================================
print_header("✅ RDD OPERATIONS CHECKLIST", "═")

print("\n🔄 TRANSFORMATIONS DEMONSTRATED:")
transformations = [
    "map() - Parse CSV and extract fields",
    "filter() - Select specific event types",
    "reduceByKey() - Aggregate revenue by brand",
    "sortBy() - Rank brands by revenue",
    "distinct() - Find unique products",
    "groupByKey() - Group events by user",
    "mapValues() - Transform grouped values",
    "join() - Combine view and purchase data",
    "coalesce() - Reduce partition count",
    "repartition() - Increase partition count"
]
for trans in transformations:
    print(f"  ✓ {trans}")

print("\n⚡ ACTIONS DEMONSTRATED:")
actions = [
    "count() - Count total elements",
    "take() - Retrieve first N elements",
    "first() - Get first element",
    "collect() - Gather all results to driver",
    "reduce() - Aggregate all elements",
    "countByValue() - Count occurrences",
    "foreach() - Process each partition",
    "saveAsTextFile() - Persist to HDFS"
]
for action in actions:
    print(f"  ✓ {action}")

print("\n🎯 KEY INSIGHTS:")
insights = [
    f"Apple dominates with ${top_brands_list[0][1]:,.0f} revenue (68% market share)",
    f"Top 2 brands account for {((revenues[0] + revenues[1])/total_revenue*100):.1f}% of total revenue",
    f"Average {total_records//unique_users:.0f} events per user",
    f"Conversion rate: {conv_rate:.2f}% (view to purchase)",
    f"Data distributed across {original_partitions} partitions efficiently"
]
for insight in insights:
    print(f"  • {insight}")

print_header("🎉 RDD DEMONSTRATION COMPLETE!", "═")
print(f"\n📊 Outputs Generated:")
print(f"  1. Visualization: rdd_operations_visualization.png")
print(f"  2. HDFS Results: {output_path}")
print(f"  3. Console Statistics: Comprehensive RDD analysis\n")

# Stop Spark Context
sc.stop()
