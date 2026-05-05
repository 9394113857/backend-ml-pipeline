# =====================================================
# ML PIPELINE (FINAL - PRODUCTION READY ✅)
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
RECO_DB_URL = os.getenv("RECO_DB_URL")

TOP_K = 10

if not EVENTS_DB_URL or not RECO_DB_URL:
    print("❌ Missing environment variables")
    exit(1)

print("✅ Environment variables loaded")

# =========================
# DB CONNECTIONS
# =========================
events_engine = create_engine(EVENTS_DB_URL)
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

print(f"📊 Events fetched: {len(events_df)}")

if events_df.empty:
    print("⚠️ No events found. Exiting safely.")
    exit(0)

# =========================
# STEP 2: CLEAN DATA
# =========================
events_df["object_id"] = pd.to_numeric(events_df["object_id"], errors="coerce")
events_df = events_df.dropna(subset=["object_id"])
events_df["object_id"] = events_df["object_id"].astype(int)

# =========================
# STEP 3: EVENT WEIGHTS (FIXED ✅)
# =========================
EVENT_WEIGHTS = {
    "view_product": 1,
    "add_to_cart": 3,
    "checkout": 5,            # ✅ FIXED (matches backend)
    "order_cancelled": -3
}

events_df["score"] = events_df["event_type"].map(EVENT_WEIGHTS).fillna(0)

# Remove useless events
events_df = events_df[events_df["score"] != 0]

print(f"🎯 Useful events after filtering: {len(events_df)}")

if events_df.empty:
    print("⚠️ No useful events after scoring. Exiting safely.")
    exit(0)

# =========================
# STEP 4: FEATURE BUILDING
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

if ranked_df.empty:
    print("⚠️ No ranked data. Exiting safely.")
    exit(0)

# =========================
# STEP 6: FINAL DATA
# =========================
final_df = ranked_df.copy()

current_time = datetime.utcnow()

final_df["created_at"] = current_time
final_df["updated_at"] = current_time   # ✅ NEW COLUMN

final_df = final_df[
    ["user_id", "product_id", "score", "rank", "created_at", "updated_at"]
]

print(f"🔥 FINAL ROWS: {len(final_df)}")

# =========================
# STEP 7: UPSERT (FINAL ✅)
# =========================
try:
    with reco_engine.begin() as conn:

        for _, row in final_df.iterrows():
            conn.execute(text("""
                INSERT INTO recommendations 
                (user_id, product_id, score, rank, created_at, updated_at)
                VALUES 
                (:user_id, :product_id, :score, :rank, :created_at, :updated_at)
                ON CONFLICT (user_id, product_id)
                DO UPDATE SET
                    score = EXCLUDED.score,
                    rank = EXCLUDED.rank,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at;
            """), {
                "user_id": int(row["user_id"]),
                "product_id": int(row["product_id"]),
                "score": float(row["score"]),
                "rank": int(row["rank"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            })

    print("✅ RECOMMENDATIONS UPSERTED SUCCESSFULLY")

except Exception as e:
    print("❌ Insert failed:", str(e))
    exit(1)

print("-" * 60)
print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
