"""
RDD Operations Demonstration
Showcasing Transformations and Actions for E-Commerce Data
"""

from pyspark import SparkContext, SparkConf

# Initialize Spark Context
conf = SparkConf().setAppName("RDD Operations Demo") \
    .set("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
sc = SparkContext(conf=conf)

print("=" * 80)
print("RDD OPERATIONS DEMONSTRATION")
print("=" * 80)

# Load data from HDFS as RDD
print("\n[1] Loading data from HDFS as RDD...")
raw_rdd = sc.textFile("hdfs:///ecommerce/ecommerce_sample.csv")

# Get header and data
header = raw_rdd.first()
data_rdd = raw_rdd.filter(lambda line: line != header)

print(f"Total records: {data_rdd.count():,}")

# ============================================================================
# TRANSFORMATION OPERATIONS
# ============================================================================
print("\n" + "=" * 80)
print("RDD TRANSFORMATIONS")
print("=" * 80)

# Transformation 1: MAP - Parse CSV lines
print("\n[Transformation 1: MAP] Parsing CSV data...")
parsed_rdd = data_rdd.map(lambda line: line.split(','))
print("Sample parsed record:")
print(parsed_rdd.take(1))

# Transformation 2: FILTER - Get only purchase events
print("\n[Transformation 2: FILTER] Extracting purchase events...")
# event_type is at index 1
purchases_rdd = parsed_rdd.filter(lambda x: len(x) > 1 and x[1] == 'purchase')
print(f"Purchase records: {purchases_rdd.count():,}")

# Transformation 3: MAP - Extract (brand, price) tuples
print("\n[Transformation 3: MAP] Creating (brand, price) pairs...")
# brand is at index 5, price is at index 6
brand_price_rdd = purchases_rdd.map(lambda x: (
    x[5] if len(x) > 5 else "unknown",
    float(x[6]) if len(x) > 6 and x[6].replace('.','').isdigit() else 0.0
))
print("Sample brand-price pairs:")
print(brand_price_rdd.take(3))

# Transformation 4: REDUCE BY KEY - Total revenue per brand
print("\n[Transformation 4: REDUCE BY KEY] Calculating revenue per brand...")
brand_revenue_rdd = brand_price_rdd.reduceByKey(lambda a, b: a + b)
print("Sample brand revenues:")
print(brand_revenue_rdd.take(5))

# Transformation 5: SORT BY - Top brands by revenue
print("\n[Transformation 5: SORT BY] Sorting brands by revenue...")
top_brands_rdd = brand_revenue_rdd.sortBy(lambda x: x[1], ascending=False)
print("\nTop 10 Brands by Revenue:")
for brand, revenue in top_brands_rdd.take(10):
    print(f"  {brand}: ${revenue:,.2f}")

# Transformation 6: MAP - Get product IDs
print("\n[Transformation 6: MAP] Extracting product IDs...")
# product_id is at index 2
product_ids_rdd = parsed_rdd.map(lambda x: x[2] if len(x) > 2 else None) \
    .filter(lambda x: x is not None)

# Transformation 7: DISTINCT - Unique products
print("\n[Transformation 7: DISTINCT] Finding unique products...")
unique_products_rdd = product_ids_rdd.distinct()
print(f"Total unique products: {unique_products_rdd.count():,}")

# Transformation 8: GROUP BY KEY - Events per user
print("\n[Transformation 8: GROUP BY KEY] Grouping events by user...")
# user_id is at index 7, event_type is at index 1
user_events_rdd = parsed_rdd.map(lambda x: (
    x[7] if len(x) > 7 else "unknown",
    x[1] if len(x) > 1 else "unknown"
)).groupByKey()

# Map values to list
user_events_list_rdd = user_events_rdd.mapValues(list)
print("Sample user event groups:")
sample_users = user_events_list_rdd.take(2)
for user, events in sample_users:
    print(f"  User {user}: {len(events)} events")

# ============================================================================
# ACTION OPERATIONS
# ============================================================================
print("\n" + "=" * 80)
print("RDD ACTIONS")
print("=" * 80)

# Action 1: COUNT
print("\n[Action 1: COUNT] Counting records...")
total_count = data_rdd.count()
purchase_count = purchases_rdd.count()
print(f"Total records: {total_count:,}")
print(f"Purchase records: {purchase_count:,}")

# Action 2: TAKE
print("\n[Action 2: TAKE] Taking first 3 records...")
first_three = parsed_rdd.take(3)
for i, record in enumerate(first_three, 1):
    print(f"  Record {i}: {record[:3]}...")

# Action 3: FIRST
print("\n[Action 3: FIRST] Getting first record...")
first_record = parsed_rdd.first()
print(f"First record: {first_record[:3]}...")

# Action 4: COLLECT (on small dataset)
print("\n[Action 4: COLLECT] Collecting top 5 brands...")
top_5_brands = top_brands_rdd.take(5)
collected_data = top_5_brands  # Use take instead of collect for safety
print("Collected data:")
for brand, revenue in collected_data:
    print(f"  {brand}: ${revenue:,.2f}")

# Action 5: REDUCE - Total revenue
print("\n[Action 5: REDUCE] Calculating total revenue...")
total_revenue = brand_price_rdd.map(lambda x: x[1]).reduce(lambda a, b: a + b)
print(f"Total revenue: ${total_revenue:,.2f}")

# Action 6: FOREACH (demonstrate with count)
print("\n[Action 6: FOREACH] Processing each partition...")
partition_counts = data_rdd.mapPartitions(lambda x: [sum(1 for _ in x)]).collect()
print(f"Records per partition: {partition_counts}")
print(f"Total partitions: {len(partition_counts)}")

# Action 7: COUNT BY VALUE - Event type distribution
print("\n[Action 7: COUNT BY VALUE] Event type distribution...")
event_types_rdd = parsed_rdd.map(lambda x: x[1] if len(x) > 1 else "unknown")
event_distribution = event_types_rdd.countByValue()
print("Event distribution:")
for event, count in event_distribution.items():
    print(f"  {event}: {count:,}")

# Action 8: SAVE AS TEXT FILE
print("\n[Action 8: SAVE AS TEXT FILE] Saving results to HDFS...")
output_path = "hdfs:///ecommerce/rdd_results"
try:
    # Delete if exists
    import subprocess
    subprocess.run(["hdfs", "dfs", "-rm", "-r", output_path], 
                   stderr=subprocess.DEVNULL)
except:
    pass

# Save top brands
top_brands_rdd.coalesce(1).saveAsTextFile(output_path + "/top_brands")
print(f"Results saved to {output_path}")

# ============================================================================
# ADVANCED OPERATIONS
# ============================================================================
print("\n" + "=" * 80)
print("ADVANCED RDD OPERATIONS")
print("=" * 80)

# Join Operation: Users with purchases
print("\n[JOIN] Finding users with purchases...")
# Create user purchase RDD
user_purchase_rdd = purchases_rdd.map(lambda x: (
    x[7] if len(x) > 7 else "unknown",
    1
)).reduceByKey(lambda a, b: a + b)

# Create user view RDD
views_rdd = parsed_rdd.filter(lambda x: len(x) > 1 and x[1] == 'view')
user_view_rdd = views_rdd.map(lambda x: (
    x[7] if len(x) > 7 else "unknown",
    1
)).reduceByKey(lambda a, b: a + b)

# Join
user_behavior_rdd = user_view_rdd.join(user_purchase_rdd)
print(f"Users with both views and purchases: {user_behavior_rdd.count():,}")
print("Sample user behavior (user_id, (views, purchases)):")
for user_data in user_behavior_rdd.take(3):
    user_id, (views, purchases) = user_data
    print(f"  User {user_id}: {views} views, {purchases} purchases")

# CartesianProduct (on small sample for demonstration)
print("\n[CARTESIAN] Demonstrating cartesian product (small sample)...")
small_sample = parsed_rdd.take(3)
small_rdd1 = sc.parallelize([x[1] for x in small_sample])  # event types
small_rdd2 = sc.parallelize([x[5] for x in small_sample])  # brands
cartesian = small_rdd1.cartesian(small_rdd2)
print(f"Cartesian product size: {cartesian.count()}")

# Coalesce and Repartition
print("\n[COALESCE/REPARTITION] Partition management...")
print(f"Original partitions: {data_rdd.getNumPartitions()}")
coalesced_rdd = data_rdd.coalesce(2)
print(f"After coalesce(2): {coalesced_rdd.getNumPartitions()}")
repartitioned_rdd = data_rdd.repartition(8)
print(f"After repartition(8): {repartitioned_rdd.getNumPartitions()}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("RDD OPERATIONS SUMMARY")
print("=" * 80)

print("\nTransformations Demonstrated:")
print("  ✓ map() - Transform each element")
print("  ✓ filter() - Select elements")
print("  ✓ reduceByKey() - Aggregate by key")
print("  ✓ sortBy() - Sort elements")
print("  ✓ distinct() - Remove duplicates")
print("  ✓ groupByKey() - Group by key")
print("  ✓ mapValues() - Transform values")
print("  ✓ join() - Join two RDDs")

print("\nActions Demonstrated:")
print("  ✓ count() - Count elements")
print("  ✓ take() - Get first N elements")
print("  ✓ first() - Get first element")
print("  ✓ collect() - Get all elements")
print("  ✓ reduce() - Aggregate all elements")
print("  ✓ foreach() - Process each element")
print("  ✓ countByValue() - Count occurrences")
print("  ✓ saveAsTextFile() - Save to file")

print("\n" + "=" * 80)
print("RDD DEMONSTRATION COMPLETE!")
print("=" * 80)

# Stop Spark Context
sc.stop()
