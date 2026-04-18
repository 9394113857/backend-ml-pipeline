# =====================================================
# ML RECOMMENDATION PIPELINE (FINAL VERSION)
# =====================================================

import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime

print("🚀 ML PIPELINE STARTED")
print("-" * 60)

# =========================
# ENV CONFIG
# =========================
EVENTS_DB_URL = os.getenv("EVENTS_DB_URL")
PRODUCT_DB_URL = os.getenv("PRODUCT_DB_URL")
RECO_DB_URL = os.getenv("RECO_DB_URL")

TOP_K = 10

if not EVENTS_DB_URL or not PRODUCT_DB_URL or not RECO_DB_URL:
    print("❌ Missing environment variables")
    exit(1)

print("✅ Environment variables loaded")

# =========================
# DB CONNECTIONS
# =========================
events_engine = create_engine(EVENTS_DB_URL)
product_engine = create_engine(PRODUCT_DB_URL)
reco_engine = create_engine(RECO_DB_URL)

print("✅ DB connections ready")
print("-" * 60)

# =========================
# STEP 1: FETCH EVENTS
# =========================
events_df = pd.read_sql("""
SELECT user_id, event_type, object_id
FROM user_events
WHERE user_id IS NOT NULL
AND object_type = 'product'
""", events_engine)

print("📊 Events fetched:", len(events_df))

if events_df.empty:
    print("⚠️ No events found. Exiting safely.")
    exit(0)

# =========================
# STEP 2: FETCH PRODUCTS
# =========================
products_df = pd.read_sql("""
SELECT id, name
FROM products
""", product_engine)

print("📦 Products fetched:", len(products_df))

if products_df.empty:
    print("⚠️ No products found. Exiting safely.")
    exit(0)

print("-" * 60)

# =========================
# STEP 3: CLEAN DATA
# =========================
events_df["object_id"] = pd.to_numeric(events_df["object_id"], errors="coerce")
events_df = events_df.dropna(subset=["object_id"])
events_df["object_id"] = events_df["object_id"].astype(int)

# =========================
# STEP 4: EVENT WEIGHTS
# =========================
EVENT_WEIGHTS = {
    "view_product": 1,
    "add_to_cart": 3,
    "checkout": 5,
    "remove_from_cart": -2,
    "order_cancelled": -3
}

events_df["score"] = events_df["event_type"].map(EVENT_WEIGHTS).fillna(0)

# =========================
# STEP 5: FEATURE BUILDING
# =========================
features_df = (
    events_df
    .groupby(["user_id", "object_id"])["score"]
    .sum()
    .reset_index()
    .rename(columns={"object_id": "product_id"})
)

print("🧠 Feature rows:", len(features_df))

if features_df.empty:
    print("⚠️ No features generated. Exiting safely.")
    exit(0)

# =========================
# STEP 6: RANKING
# =========================
features_df["rank"] = (
    features_df
    .groupby("user_id")["score"]
    .rank(method="first", ascending=False)
)

# 🔥 IMPORTANT FIX → convert rank to int (matches DB)
features_df["rank"] = features_df["rank"].astype(int)

ranked_df = features_df[features_df["rank"] <= TOP_K]

print("📊 Ranked rows:", len(ranked_df))

# =========================
# STEP 7: MERGE PRODUCTS
# =========================
final_df = ranked_df.merge(
    products_df,
    left_on="product_id",
    right_on="id",
    how="inner"
)

print("🔥 FINAL ROWS:", len(final_df))

if final_df.empty:
    print("⚠️ No matching products. Exiting safely.")
    exit(0)

# =========================
# STEP 8: FINAL FORMAT
# =========================
final_df["created_at"] = datetime.utcnow()

final_df = final_df[
    ["user_id", "product_id", "score", "rank", "created_at"]
]

# =========================
# STEP 9: CLEAR OLD DATA
# =========================
try:
    with reco_engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE recommendations"))
    print("🧹 Old recommendations cleared")
except Exception as e:
    print("⚠️ Truncate failed:", str(e))

# =========================
# STEP 10: INSERT DATA
# =========================
try:
    final_df.to_sql(
        "recommendations",
        reco_engine,
        if_exists="append",
        index=False
    )
    print("✅ RECOMMENDATIONS INSERTED SUCCESSFULLY")
except Exception as e:
    print("❌ Insert failed:", str(e))
    exit(1)

print("-" * 60)
print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
