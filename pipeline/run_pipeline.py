import os
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import smtplib
from email.message import EmailMessage

# ===============================

# Pipeline start

# ===============================

PIPELINE_START = datetime.utcnow()
print("ML PIPELINE STARTED")

# ===============================

# Environment variables

# ===============================

EVENTS_DB_URL = os.getenv("EVENTS_DB_URL")
PRODUCT_DB_URL = os.getenv("PRODUCT_DB_URL")
RECO_DB_URL = os.getenv("RECO_DB_URL")

MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_CC = os.getenv("MAIL_CC")

MODEL_VERSION = "v2_weighted_events"
TOP_K = 5   # number of recommendations per user

# ===============================

# Database connections

# ===============================

events_engine = create_engine(EVENTS_DB_URL)
product_engine = create_engine(PRODUCT_DB_URL)
reco_engine = create_engine(RECO_DB_URL)

# ===============================

# Fetch user events

# ===============================

events_df = pd.read_sql("""
SELECT user_id, event_type, object_id
FROM user_events
WHERE user_id IS NOT NULL
AND object_type = 'product'
""", events_engine)

print("Events rows:", len(events_df))

# ===============================

# Fetch product catalog

# ===============================

products_df = pd.read_sql("""
SELECT id, name, category, price
FROM products
""", product_engine)

print("Products:", len(products_df))

# ===============================

# Event weights

# ===============================

EVENT_WEIGHTS = {
"view_product": 1,
"add_to_cart": 3,
"checkout": 5,
"remove_from_cart": -2
}

# Convert events to score

events_df["score"] = events_df["event_type"].map(EVENT_WEIGHTS).fillna(0)

# ===============================

# Aggregate scores per user-product

# ===============================

features_df = (
events_df
.groupby(["user_id", "object_id"])["score"]
.sum()
.reset_index()
.rename(columns={"object_id": "product_id"})
)

# Clean product id

features_df["product_id"] = pd.to_numeric(features_df["product_id"], errors="coerce")
features_df = features_df.dropna(subset=["product_id"])
features_df["product_id"] = features_df["product_id"].astype(int)

# ===============================

# Rank products per user

# ===============================

features_df["rank"] = (
features_df
.groupby("user_id")["score"]
.rank(method="first", ascending=False)
)

ranked_df = features_df[features_df["rank"] <= TOP_K]

# ===============================

# Remove already purchased items

# ===============================

purchased = events_df[
events_df["event_type"] == "checkout"
][["user_id", "object_id"]]

purchased = purchased.rename(columns={"object_id": "product_id"})

ranked_df = ranked_df.merge(
purchased,
on=["user_id", "product_id"],
how="left",
indicator=True
)

ranked_df = ranked_df[ranked_df["_merge"] == "left_only"]
ranked_df = ranked_df.drop(columns=["_merge"])

# ===============================

# Cold start fallback

# ===============================

if ranked_df.empty:

```
print("Using trending fallback")

trending_df = pd.read_sql("""
SELECT object_id AS product_id, COUNT(*) AS score
FROM user_events
WHERE object_type='product'
GROUP BY object_id
ORDER BY score DESC
LIMIT 5
""", events_engine)

trending_df["user_id"] = 0
trending_df["rank"] = range(1, len(trending_df) + 1)

ranked_df = trending_df
```

# ===============================

# Prepare final dataset

# ===============================

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

print("Final rows:", len(final_df))

# ===============================

# Refresh recommendations table

# ===============================

with reco_engine.begin() as conn:
conn.execute(
text("""
DELETE FROM recommendations
WHERE model_version = :version
"""),
{"version": MODEL_VERSION}
)

# Insert new recommendations

final_df.to_sql(
"recommendations",
reco_engine,
if_exists="append",
index=False
)

print("Recommendations saved")

# ===============================

# Send email notification

# ===============================

if MAIL_USERNAME and MAIL_PASSWORD:

```
msg = EmailMessage()

msg["Subject"] = "ML Pipeline Success"
msg["From"] = MAIL_USERNAME
msg["To"] = MAIL_USERNAME

if MAIL_CC:
    msg["Cc"] = MAIL_CC

msg.set_content(
    f"ML pipeline finished\nUsers: {final_df['user_id'].nunique()}\nRows: {len(final_df)}"
)

recipients = [MAIL_USERNAME]

if MAIL_CC:
    recipients.append(MAIL_CC)

with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(MAIL_USERNAME, MAIL_PASSWORD)
    server.send_message(msg, to_addrs=recipients)

print("Email sent")
```

# ===============================

# Pipeline end

# ===============================

PIPELINE_END = datetime.utcnow()

print("ML PIPELINE COMPLETED")
print("Duration:", PIPELINE_END - PIPELINE_START)
