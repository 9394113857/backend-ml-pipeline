import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import smtplib
from email.message import EmailMessage

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

MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")

MODEL_VERSION = "v1_weighted_events"
TOP_K = 10

assert EVENTS_DB_URL, "❌ EVENTS_DB_URL missing"
assert PRODUCT_DB_URL, "❌ PRODUCT_DB_URL missing"
assert RECO_DB_URL, "❌ RECO_DB_URL missing"

print("✅ Environment variables loaded")
print(f"📌 Model Version: {MODEL_VERSION}")
print("-" * 60)

# =========================
# DB CONNECTIONS
# =========================
events_engine = create_engine(EVENTS_DB_URL)
product_engine = create_engine(PRODUCT_DB_URL)
reco_engine = create_engine(RECO_DB_URL)

print("✅ Database connections established")
print("-" * 60)

# =========================
# STEP 1: FETCH EVENTS
# =========================
print("📥 Fetching user events...")

events_df = pd.read_sql(
    """
    SELECT user_id, event_type, object_id
    FROM user_events
    WHERE user_id IS NOT NULL
      AND object_type = 'product'
    """,
    events_engine
)

print(f"✅ Events fetched: {len(events_df)} rows")

print("📥 Fetching products...")

products_df = pd.read_sql(
    """
    SELECT id, name, category, price
    FROM products
    """,
    product_engine
)

print(f"✅ Products fetched: {len(products_df)} rows")
print("-" * 60)

# =========================
# STEP 2: FEATURE ENGINEERING
# =========================
print("🧠 Feature engineering...")

EVENT_WEIGHTS = {
    "view_product": 1,
    "add_to_cart": 3,
    "checkout": 5,
    "remove_from_cart": -2
}

events_df["score"] = events_df["event_type"].map(EVENT_WEIGHTS).fillna(0)

features_df = (
    events_df
    .groupby(["user_id", "object_id"])["score"]
    .sum()
    .reset_index()
    .rename(columns={"object_id": "product_id"})
)

# Data type safety
features_df["product_id"] = pd.to_numeric(
    features_df["product_id"], errors="coerce"
)
features_df = features_df.dropna(subset=["product_id"])
features_df["product_id"] = features_df["product_id"].astype(int)

print(f"✅ Feature rows: {len(features_df)}")
print("-" * 60)

# =========================
# STEP 3: RANKING
# =========================
print("📊 Ranking products per user...")

features_df["rank"] = (
    features_df
    .groupby("user_id")["score"]
    .rank(method="first", ascending=False)
)

ranked_df = features_df[features_df["rank"] <= TOP_K]

print(f"✅ Ranked rows: {len(ranked_df)}")
print(f"👥 Users covered: {ranked_df['user_id'].nunique()}")
print("-" * 60)

# =========================
# STEP 4: FINAL DATASET
# =========================
print("🔗 Preparing final dataset...")

final_df = ranked_df.merge(
    products_df,
    left_on="product_id",
    right_on="id",
    how="left"
)

final_df["model_version"] = MODEL_VERSION
final_df["created_at"] = datetime.utcnow()

final_df = final_df[[
    "user_id",
    "product_id",
    "score",
    "rank",
    "model_version",
    "created_at"
]]

print(f"✅ Final rows ready: {len(final_df)}")
print("-" * 60)

# =========================
# STEP 5: SAFE SAVE (🔥 IMPORTANT)
# =========================
print("♻ Cleaning old recommendations for this model version...")

with reco_engine.begin() as conn:
    conn.execute(
        text("""
            DELETE FROM recommendations
            WHERE model_version = :version
        """),
        {"version": MODEL_VERSION}
    )

print("💾 Inserting new recommendations...")

final_df.to_sql(
    "recommendations",
    reco_engine,
    if_exists="append",   # ✅ NEVER replace
    index=False
)

print("✅ Recommendations refreshed safely")
print("-" * 60)

# =========================
# STEP 6: EMAIL ALERT
# =========================
print("📧 Sending email alert...")

if MAIL_USERNAME and MAIL_PASSWORD:
    msg = EmailMessage()
    msg["Subject"] = f"✅ ML Pipeline Success | {MODEL_VERSION}"
    msg["From"] = MAIL_USERNAME
    msg["To"] = MAIL_USERNAME

    msg.set_content(f"""
ML Recommendation Pipeline Completed 🚀

Model Version : {MODEL_VERSION}
Users Covered : {final_df['user_id'].nunique()}
Rows Inserted : {len(final_df)}
Run Time      : {datetime.utcnow()}

Status        : SUCCESS ✅
""")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.send_message(msg)

    print("📨 Email sent")
else:
    print("ℹ️ Email skipped (credentials not set)")

# =========================
# PIPELINE END
# =========================
PIPELINE_END = datetime.utcnow()
print("-" * 60)
print("🏁 ML PIPELINE COMPLETED")
print(f"🕒 End Time (UTC): {PIPELINE_END}")
print(f"⏱ Duration: {PIPELINE_END - PIPELINE_START}")
print("✅ ALL STEPS EXECUTED SUCCESSFULLY")
