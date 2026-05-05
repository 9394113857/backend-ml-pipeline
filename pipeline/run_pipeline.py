# =====================================================
# ML PIPELINE (FINAL - FULLY EXPLAINED VERSION ✅)
# =====================================================

# Import required libraries
import os                          # For reading environment variables
import pandas as pd               # For data processing (DataFrames)
from sqlalchemy import create_engine, text  # For DB connection + SQL execution
from datetime import datetime     # For timestamps

# Initial logs
print("🚀 ML PIPELINE STARTED")
print("-" * 60)

# =========================
# ENV CONFIG
# =========================
# Read database URLs from environment variables
# These should be set in your system / Render / local environment
EVENTS_DB_URL = os.getenv("EVENTS_DB_URL")
RECO_DB_URL = os.getenv("RECO_DB_URL")

# Number of top recommendations per user
TOP_K = 10

# Validate if environment variables exist
if not EVENTS_DB_URL or not RECO_DB_URL:
    print("❌ Missing environment variables")
    exit(1)

print("✅ Environment variables loaded")

# =========================
# DB CONNECTIONS
# =========================
# Create database engines (connections)
# events_engine → reads from user_events table
# reco_engine   → writes into recommendations table
events_engine = create_engine(EVENTS_DB_URL)
reco_engine = create_engine(RECO_DB_URL)

print("✅ DB connections ready")
print("-" * 60)

# =========================
# STEP 1: FETCH EVENTS
# =========================
# Fetch only relevant events from DB
# - user_id must exist
# - only product-related events (important for ML)
events_df = pd.read_sql("""
    SELECT user_id, event_type, object_id
    FROM user_events
    WHERE user_id IS NOT NULL
      AND object_type = 'product'
""", events_engine)

print(f"📊 Events fetched: {len(events_df)}")

# If no events, stop pipeline safely
if events_df.empty:
    print("⚠️ No events found. Exiting safely.")
    exit(0)

# =========================
# STEP 2: CLEAN DATA
# =========================
# Convert object_id → numeric
# (sometimes DB stores as string, so we fix it)
events_df["object_id"] = pd.to_numeric(events_df["object_id"], errors="coerce")

# Drop rows where conversion failed (invalid product_id)
events_df = events_df.dropna(subset=["object_id"])

# Convert to integer (final clean product_id)
events_df["object_id"] = events_df["object_id"].astype(int)

# =========================
# STEP 3: EVENT WEIGHTS
# =========================
# Assign importance (weights) to each event type
# Higher weight = more importance in recommendation
EVENT_WEIGHTS = {
    "view_product": 1,        # low signal
    "add_to_cart": 3,         # medium signal
    "checkout": 5,            # strong signal
    "order_cancelled": -3     # negative signal
}

# Map event_type → score
events_df["score"] = events_df["event_type"].map(EVENT_WEIGHTS).fillna(0)

# Remove events with score = 0 (not useful)
events_df = events_df[events_df["score"] != 0]

print(f"🎯 Useful events after filtering: {len(events_df)}")

# If no useful events, stop
if events_df.empty:
    print("⚠️ No useful events after scoring. Exiting safely.")
    exit(0)

# =========================
# STEP 4: FEATURE BUILDING
# =========================
# Group by user + product
# Sum scores → total interest per product per user
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
# Rank products per user based on score
# Highest score → rank 1
features_df["rank"] = (
    features_df
    .groupby("user_id")["score"]
    .rank(method="first", ascending=False)
)

# Keep only top K recommendations per user
ranked_df = features_df[features_df["rank"] <= TOP_K]

print(f"📊 Ranked rows: {len(ranked_df)}")

# If no ranked data, stop
if ranked_df.empty:
    print("⚠️ No ranked data. Exiting safely.")
    exit(0)

# =========================
# STEP 6: FINAL DATA
# =========================
# Copy final dataset
final_df = ranked_df.copy()

# Generate ONE timestamp for this pipeline run
# (important: all rows get same consistent time)
current_time = datetime.utcnow()

# created_at → when record FIRST inserted
# updated_at → when record LAST updated
final_df["created_at"] = current_time
final_df["updated_at"] = current_time

# Select final columns (matching DB schema exactly)
final_df = final_df[
    ["user_id", "product_id", "score", "rank", "created_at", "updated_at"]
]

print(f"🔥 FINAL ROWS: {len(final_df)}")

# =========================
# STEP 7: UPSERT INTO DB
# =========================
# UPSERT = INSERT + UPDATE
# If row exists → update
# If not → insert
try:
    with reco_engine.begin() as conn:

        # Loop through each row
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
                    updated_at = EXCLUDED.updated_at;

                -- IMPORTANT:
                -- created_at is NOT updated
                -- so original creation time is preserved
            """), {
                "user_id": int(row["user_id"]),          # convert to int
                "product_id": int(row["product_id"]),    # convert to int
                "score": float(row["score"]),            # convert to float
                "rank": int(row["rank"]),                # convert to int
                "created_at": row["created_at"],         # timestamp
                "updated_at": row["updated_at"]          # timestamp
            })

    print("✅ RECOMMENDATIONS UPSERTED SUCCESSFULLY")

# If error occurs
except Exception as e:
    print("❌ Insert failed:", str(e))
    exit(1)

# Final success log
print("-" * 60)
print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
