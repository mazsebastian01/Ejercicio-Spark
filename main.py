from decimal import Decimal
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import (
    StructType, StructField, StringType
)

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("taller-data-engineer")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

customers_schema = StructType([
    StructField("customer_id", StringType(), True),
    StructField("first_name", StringType(), True),
    StructField("last_name", StringType(), True),
    StructField("email", StringType(), True),
    StructField("country", StringType(), True),
    StructField("signup_date", StringType(), True),
    StructField("segment", StringType(), True),
])
products_schema = StructType([
    StructField("product_id", StringType(), True),
    StructField("product_name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("unit_price", StringType(), True),
    StructField("active", StringType(), True),
])
orders_schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("order_ts", StringType(), True),
    StructField("status", StringType(), True),
    StructField("channel", StringType(), True),
    StructField("updated_at", StringType(), True),
])
items_schema = StructType([
    StructField("line_id", StringType(), True),
    StructField("order_id", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("quantity", StringType(), True),
    StructField("discount_pct", StringType(), True),
])

def read_csv(path, schema):
    return (spark.read.option("header", True)
            .option("encoding", "UTF-8")
            .schema(schema)
            .csv(path))

customers_raw = read_csv("datos/customers.csv", customers_schema)
products_raw = read_csv("datos/products.csv", products_schema)
orders_raw = read_csv("datos/orders.csv", orders_schema)
items_raw = read_csv("datos/order_items.csv", items_schema)

orders_inc_raw = spark.read.schema(orders_schema).json(
    "datos/incremental/orders_incremental.jsonl"
)
items_inc_raw = spark.read.schema(items_schema).json(
    "datos/incremental/order_items_incremental.jsonl"
)

def profile(name, df, key):
    print(f"\\n--- PROFILE: {name} ---")
    print("Rows:", df.count())
    df.printSchema()
    null_counts = df.select([
        F.sum(F.when(F.col(c).isNull() | (F.trim(F.col(c)) == ""), 1)
              .otherwise(0)).alias(c)
        for c in df.columns
    ])
    null_counts.show(truncate=False)
    duplicate_keys = (df.groupBy(key).count()
                      .filter(F.col(key).isNotNull() & (F.col("count") > 1)))
    print("Duplicate key groups:", duplicate_keys.count())
    duplicate_keys.show(10, truncate=False)

profile("customers", customers_raw, "customer_id")
profile("products", products_raw, "product_id")
profile("orders", orders_raw, "order_id")
profile("order_items", items_raw, "line_id")

customers_typed = (customers_raw
    .withColumn("customer_id", F.trim("customer_id"))
    .withColumn("signup_date", F.expr("try_cast(signup_date as date)"))
    .withColumn("country", F.upper(F.trim("country")))
)

products_typed = (products_raw
    .withColumn("product_id", F.trim("product_id"))
    .withColumn("unit_price", F.expr("try_cast(unit_price as decimal(12,2))"))
    .withColumn("active_norm", F.lower(F.trim("active")))
    .withColumn("active", F.when(F.col("active_norm").isin("true", "1", "yes"), True)
                           .when(F.col("active_norm").isin("false", "0", "no"), False)
                           .otherwise(None))
    .drop("active_norm")
)

orders_typed = (orders_raw
    .withColumn("order_id", F.trim("order_id"))
    .withColumn("customer_id", F.trim("customer_id"))
    .withColumn("order_ts", F.expr("try_to_timestamp(order_ts)"))
    .withColumn("updated_at", F.expr("try_to_timestamp(updated_at)"))
    .withColumn("status", F.lower(F.trim("status")))
    .withColumn("status", F.when(F.col("status") == "deliverd", "delivered")
                           .otherwise(F.col("status")))
)

items_typed = (items_raw
    .withColumn("line_id", F.trim("line_id"))
    .withColumn("order_id", F.trim("order_id"))
    .withColumn("product_id", F.trim("product_id"))
    .withColumn("quantity", F.expr("try_cast(quantity as int)"))
    .withColumn("discount_pct", F.expr("try_cast(discount_pct as decimal(8,4))"))
)

def split_duplicates(df, key, order_columns):
    w = Window.partitionBy(key).orderBy(*[F.col(c).asc_nulls_last() for c in order_columns])
    ranked = df.withColumn("_duplicate_rank", F.row_number().over(w))
    duplicates = ranked.filter(F.col("_duplicate_rank") > 1).drop("_duplicate_rank")
    unique = (ranked.filter(F.col("_duplicate_rank") == 1)
                    .drop("_duplicate_rank"))
    return unique, duplicates

customers, duplicate_customers = split_duplicates(
    customers_typed, "customer_id",
    ["signup_date", "email", "first_name", "last_name"]
)
products, duplicate_products = split_duplicates(
    products_typed, "product_id",
    ["product_name", "unit_price", "category"]
)
orders_unique, duplicate_orders = split_duplicates(
    orders_typed, "order_id", ["updated_at", "order_ts", "status"]
)
items_unique, duplicate_items = split_duplicates(
    items_typed, "line_id", ["order_id", "product_id", "quantity"]
)

bad_customers = customers.filter(
    F.col("customer_id").isNull() | (F.col("customer_id") == "") |
    F.col("signup_date").isNull()
)
customers_valid = customers.filter(
    F.col("customer_id").isNotNull() & (F.col("customer_id") != "") &
    F.col("signup_date").isNotNull()
)

bad_products = products.filter(
    F.col("product_id").isNull() | (F.col("product_id") == "") |
    F.col("unit_price").isNull() | (F.col("unit_price") <= 0) |
    F.col("category").isNull()
)
products_valid = products.filter(
    F.col("product_id").isNotNull() & (F.col("product_id") != "") &
    F.col("unit_price").isNotNull() & (F.col("unit_price") > 0) &
    F.col("category").isNotNull()
)

bad_orders = orders_unique.filter(
    F.col("order_id").isNull() | (F.col("order_id") == "") |
    F.col("customer_id").isNull() | (F.col("customer_id") == "") |
    F.col("order_ts").isNull() | F.col("updated_at").isNull()
)
orders_valid = orders_unique.filter(
    F.col("order_id").isNotNull() & (F.col("order_id") != "") &
    F.col("customer_id").isNotNull() & (F.col("customer_id") != "") &
    F.col("order_ts").isNotNull() & F.col("updated_at").isNotNull()
)

bad_items = items_unique.filter(
    F.col("line_id").isNull() | (F.col("line_id") == "") |
    F.col("quantity").isNull() | (F.col("quantity") <= 0) |
    F.col("discount_pct").isNull() |
    (F.col("discount_pct") < 0) | (F.col("discount_pct") > 1)
)
items_valid = items_unique.filter(
    F.col("line_id").isNotNull() & (F.col("line_id") != "") &
    F.col("quantity").isNotNull() & (F.col("quantity") > 0) &
    F.col("discount_pct").isNotNull() &
    (F.col("discount_pct") >= 0) & (F.col("discount_pct") <= 1)
)

orders_unknown_customer = orders_valid.join(
    customers_valid.select("customer_id"), "customer_id", "left_anti"
)
orders_with_customer = orders_valid.join(
    customers_valid.select("customer_id"), "customer_id", "inner"
)

items_unknown_order = items_valid.join(
    orders_with_customer.select("order_id"), "order_id", "left_anti"
)
items_with_order = items_valid.join(
    orders_with_customer.select("order_id"), "order_id", "inner"
)
items_unknown_product = items_with_order.join(
    products_valid.select("product_id"), "product_id", "left_anti"
)
items_with_product = items_with_order.join(
    products_valid.select("product_id"), "product_id", "inner"
)

print("\\n--- QUALITY FINDINGS ---")
for label, df in [
    ("Duplicate customers", duplicate_customers),
    ("Duplicate products", duplicate_products),
    ("Duplicate orders", duplicate_orders),
    ("Duplicate order lines", duplicate_items),
    ("Invalid customers", bad_customers),
    ("Invalid products", bad_products),
    ("Invalid orders / malformed timestamps", bad_orders),
    ("Invalid order lines (quantity / discount)", bad_items),
    ("Orders with unknown customer", orders_unknown_customer),
    ("Lines with unknown order", items_unknown_order),
    ("Lines with unknown product", items_unknown_product),
]:
    print(label + ":", df.count())

sales_base = (items_with_product
    .join(orders_with_customer.select(
        "order_id", "customer_id", "order_ts", "status", "channel"
    ), "order_id", "inner")
    .join(customers_valid.select("customer_id", "country"), "customer_id", "inner")
    .join(products_valid.select("product_id", "product_name", "category", "unit_price"),
          "product_id", "inner")
)

detail_sales = (sales_base
    .withColumn(
        "net_amount",
        (F.col("quantity").cast("decimal(12,0)") *
         F.col("unit_price") *
         (F.lit(1).cast("decimal(8,4)") - F.col("discount_pct")))
        .cast("decimal(18,2)")
    )
    .withColumn("order_month", F.date_format("order_ts", "yyyy-MM"))
)

print("\\n--- VALID SALES DETAIL ---")
detail_sales.select(
    "line_id", "order_id", "customer_id", "product_id", "quantity",
    "unit_price", "discount_pct", "net_amount", "order_month",
    "country", "channel", "category"
).show(10, truncate=False)

detail_sales.createOrReplaceTempView("detail_sales")
sales_summary = spark.sql("""
    SELECT
        order_month,
        country,
        channel,
        category,
        CAST(ROUND(SUM(net_amount), 2) AS DECIMAL(18,2)) AS net_sales
    FROM detail_sales
    GROUP BY order_month, country, channel, category
    ORDER BY order_month, country, channel, category
""")

print("\\n--- NET SALES BY MONTH / COUNTRY / CHANNEL / CATEGORY ---")
sales_summary.show(50, truncate=False)

orders_inc = (orders_inc_raw
    .withColumn("order_id", F.trim("order_id"))
    .withColumn("customer_id", F.trim("customer_id"))
    .withColumn("order_ts", F.expr("try_to_timestamp(order_ts)"))
    .withColumn("updated_at", F.expr("try_to_timestamp(updated_at)"))
    .withColumn("status", F.lower(F.trim("status")))
    .withColumn("status", F.when(F.col("status") == "deliverd", "delivered")
                           .otherwise(F.col("status")))
)
items_inc = (items_inc_raw
    .withColumn("line_id", F.trim("line_id"))
    .withColumn("order_id", F.trim("order_id"))
    .withColumn("product_id", F.trim("product_id"))
    .withColumn("quantity", F.expr("try_cast(quantity as int)"))
    .withColumn("discount_pct", F.expr("try_cast(discount_pct as decimal(8,4))"))
)

all_orders_once = orders_typed.unionByName(orders_inc)
all_orders_twice = all_orders_once.unionByName(orders_inc)
order_window = Window.partitionBy("order_id").orderBy(
    F.col("updated_at").desc_nulls_last(),
    F.col("order_ts").desc_nulls_last(),
    F.col("status").desc_nulls_last()
)
def latest_orders(df):
    return (df.withColumn("_rn", F.row_number().over(order_window))
              .filter(F.col("_rn") == 1)
              .drop("_rn"))

orders_current = latest_orders(all_orders_once)
orders_current_twice = latest_orders(all_orders_twice)

all_items_once = items_typed.unionByName(items_inc)
all_items_twice = all_items_once.unionByName(items_inc)
items_current = all_items_once.dropDuplicates(["line_id"])
items_current_twice = all_items_twice.dropDuplicates(["line_id"])

orders_once_count = orders_current.count()
orders_twice_count = orders_current_twice.count()
items_once_count = items_current.count()
items_twice_count = items_current_twice.count()

detail_total = detail_sales.agg(F.sum("net_amount").alias("total")).first()["total"]
summary_total = sales_summary.agg(F.sum("net_sales").alias("total")).first()["total"]
control_1 = (detail_total or Decimal("0.00")) == (summary_total or Decimal("0.00"))

duplicate_current_orders = (orders_current.groupBy("order_id").count()
    .filter(F.col("order_id").isNotNull() & (F.col("count") > 1)).count())
control_2 = (
    duplicate_current_orders == 0 and
    orders_once_count == orders_twice_count and
    items_once_count == items_twice_count
)

print("\\n--- QUALITY CONTROLS ---")
print("Control 1 - detail total:", detail_total)
print("Control 1 - aggregated total:", summary_total)
print("Control 1 - reconciliation:", "PASS" if control_1 else "FAIL")
print("Control 2 - duplicate current order IDs:", duplicate_current_orders)
print("Orders after one incremental load:", orders_once_count)
print("Orders after same batch twice:", orders_twice_count)
print("Order lines after one incremental load:", items_once_count)
print("Order lines after same batch twice:", items_twice_count)
print("Control 2 - incremental idempotence:", "PASS" if control_2 else "FAIL")

detail_sales.write.mode("overwrite").parquet("output/detail_sales")
sales_summary.write.mode("overwrite").parquet("output/sales_summary")
orders_current.write.mode("overwrite").parquet("output/orders_current")
items_current.write.mode("overwrite").parquet("output/order_items_current")

quarantine = {
    "duplicate_customers": duplicate_customers,
    "duplicate_products": duplicate_products,
    "duplicate_orders": duplicate_orders,
    "duplicate_order_items": duplicate_items,
    "invalid_customers": bad_customers,
    "invalid_products": bad_products,
    "invalid_orders": bad_orders,
    "invalid_order_items": bad_items,
    "orders_unknown_customer": orders_unknown_customer,
    "items_unknown_order": items_unknown_order,
    "items_unknown_product": items_unknown_product,
}
for name, df in quarantine.items():
    df.write.mode("overwrite").parquet(f"output/quarantine/{name}")

print("\\nETL FINISHED. Parquet files saved under output/.")
spark.stop()
