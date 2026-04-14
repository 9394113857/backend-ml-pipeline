import os
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# ==================================================
# PIPELINE START
# ==================================================
PIPELINE_START = datetime.utcnow()
print("🚀 ML PIPELINE STARTED")
print(f"🕒 Start Time (UTC): {PIPELINE_START}")
print("-" * 60)

# =========================
# ENV CONFIG
# =========================
EVENTS_DB_URL = os.getenv("EVENTS_DB_URL")
PRODUCT_DB_URL = os.getenv("PRODUCT_DB_URL")
RECO_DB_URL = os.getenv("RECO_DB_URL")

TOP_K = 10

assert EVENTS_DB_URL, "❌ EVENTS_DB_URL missing"
assert PRODUCT_DB_URL, "❌ PRODUCT_DB_URL missing"
assert RECO_DB_URL, "❌ RECO_DB_URL missing"

print("✅ Environment variables loaded")
print("-" * 60)

# =========================
# DB CONNECTIONS
# =========================
events_engine = create_engine(EVENTS_DB_URL)
product_engine = create_engine(PRODUCT_DB_URL)
reco_engine = create_engine(RECO_DB_URL)

print("✅ DB connections ready")
print("-" * 60)

# =========================
# STEP 1: FETCH DATA
# =========================
events_df = pd.read_sql("""
    SELECT user_id, event_type, object_id
    FROM user_events
    WHERE user_id IS NOT NULL
      AND object_type = 'product'
""", events_engine)

print(f"📊 Events fetched: {len(events_df)}")

products_df = pd.read_sql("""
    SELECT id, name
    FROM products
""", product_engine)

print(f"📦 Products fetched: {len(products_df)}")
print("-" * 60)

# =========================
# STEP 2: CLEAN DATA
# =========================
events_df["object_id"] = pd.to_numeric(events_df["object_id"], errors="coerce")
events_df = events_df.dropna(subset=["object_id"])
events_df["object_id"] = events_df["object_id"].astype(int)

# =========================
# STEP 3: EVENT WEIGHTS
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
# STEP 4: FEATURES
# =========================
features_df = (
    events_df
    .groupby(["user_id", "object_id"])["score"]
    .sum()
    .reset_index()
    .rename(columns={"object_id": "product_id"})
)

print(f"🧠 Feature rows: {len(features_df)}")

# =========================
# STEP 5: RANKING
# =========================
features_df["rank"] = (
    features_df
    .groupby("user_id")["score"]
    .rank(method="first", ascending=False)
)

ranked_df = features_df[features_df["rank"] <= TOP_K]

print(f"📊 Ranked rows: {len(ranked_df)}")

# =========================
# STEP 6: MERGE PRODUCTS
# =========================
final_df = ranked_df.merge(
    products_df,
    left_on="product_id",
    right_on="id",
    how="inner"
)

final_df["created_at"] = datetime.utcnow()

final_df = final_df[[
    "user_id",
    "product_id",
    "score",
    "rank",
    "created_at"
]]

print(f"🔥 FINAL ROWS: {len(final_df)}")

# =========================
# CHECK EMPTY
# =========================
if final_df.empty:
    print("❌ NO RECOMMENDATIONS GENERATED")
    exit(1)

# =========================
# STEP 7: SAVE
# =========================
final_df.to_sql(
    "recommendations",
    reco_engine,
    if_exists="append",
    index=False
)

print("✅ RECOMMENDATIONS INSERTED SUCCESSFULLY")

# =========================
# END
# =========================
PIPELINE_END = datetime.utcnow()
print("-" * 60)
print("🎉 PIPELINE COMPLETED")
print(f"⏱ Duration: {PIPELINE_END - PIPELINE_START}")
