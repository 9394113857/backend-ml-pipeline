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
print(f"Start Time (UTC): {PIPELINE_START}")
print("-" * 60)

# =========================

# ENV VARIABLES

# =========================

EVENTS_DB_URL = os.getenv("EVENTS_DB_URL")
PRODUCT_DB_URL = os.getenv("PRODUCT_DB_URL")
RECO_DB_URL = os.getenv("RECO_DB_URL")

MAIL_USERNAME = os.getenv("MAIL_USERNAME")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_CC = os.getenv("MAIL_CC")

MODEL_VERSION = "v2_weighted_events_with_fallback"
TOP_K = 5

assert EVENTS_DB_URL
assert PRODUCT_DB_URL
assert RECO_DB_URL

print("Environment variables loaded")
print("Model Version:", MODEL_VERSION)
print("-" * 60)

# =========================

# DB CONNECTIONS

# =========================

events_engine = create_engine(EVENTS_DB_URL)
product_engine = create_engine(PRODUCT_DB_URL)
reco_engine = create_engine(RECO_DB_URL)

print("Database connections established")
print("-" * 60)

# =========================

# STEP 1: FETCH EVENTS

# =========================

print("Fetching user events...")

events_df = pd.read_sql("""
SELECT user_id,event_type,object_id
FROM user_events
WHERE user_id IS NOT NULL
AND object_type='product'
""", events_engine)

print("Events fetched:", len(events_df))

# =========================

# STEP 2: FETCH PRODUCTS

# =========================

print("Fetching product catalog...")

products_df = pd.read_sql("""
SELECT id,name,category,price
FROM products
""", product_engine)

print("Products fetched:", len(products_df))
print("-" * 60)

# =========================

# STEP 3: FEATURE ENGINEERING

# =========================

EVENT_WEIGHTS = {
"view_product":1,
"add_to_cart":3,
"checkout":5,
"remove_from_cart":-2
}

events_df["score"] = events_df["event_type"].map(EVENT_WEIGHTS).fillna(0)

features_df = (
events_df
.groupby(["user_id","object_id"])["score"]
.sum()
.reset_index()
.rename(columns={"object_id":"product_id"})
)

features_df["product_id"] = pd.to_numeric(features_df["product_id"],errors="coerce")
features_df = features_df.dropna(subset=["product_id"])
features_df["product_id"] = features_df["product_id"].astype(int)

print("Feature rows:",len(features_df))
print("-"*60)

# =========================

# STEP 4: REMOVE PURCHASED PRODUCTS

# =========================

purchased = events_df[
events_df["event_type"]=="checkout"
][["user_id","object_id"]]

purchased = purchased.rename(columns={"object_id":"product_id"})

features_df = features_df.merge(
purchased,
on=["user_id","product_id"],
how="left",
indicator=True
)

features_df = features_df[features_df["_merge"]=="left_only"]
features_df = features_df.drop(columns=["_merge"])

print("Purchased items removed")

# =========================

# STEP 5: RANKING

# =========================

features_df["rank"] = (
features_df
.groupby("user_id")["score"]
.rank(method="first",ascending=False)
)

ranked_df = features_df[features_df["rank"]<=TOP_K]

print("Ranked rows:",len(ranked_df))
print("Users covered:",ranked_df["user_id"].nunique())
print("-"*60)

# =========================

# STEP 6: COLD START FALLBACK

# =========================

if ranked_df.empty:

```
print("Cold start detected → using trending products")

trending_df = pd.read_sql("""
SELECT object_id AS product_id,COUNT(*) AS score
FROM user_events
WHERE object_type='product'
GROUP BY object_id
ORDER BY score DESC
LIMIT 5
""",events_engine)

trending_df["user_id"]=0
trending_df["rank"]=range(1,len(trending_df)+1)

ranked_df=trending_df
```

print("Fallback logic checked")
print("-"*60)

# =========================

# STEP 7: FINAL DATASET

# =========================

final_df = ranked_df.merge(
products_df,
left_on="product_id",
right_on="id",
how="left"
)

final_df["model_version"]=MODEL_VERSION
final_df["created_at"]=datetime.utcnow()

final_df=final_df[[
"user_id",
"product_id",
"score",
"rank",
"model_version",
"created_at"
]]

print("Final rows ready:",len(final_df))
print("-"*60)

# =========================

# STEP 8: SAVE RECOMMENDATIONS

# =========================

print("Cleaning old recommendations")

with reco_engine.begin() as conn:
conn.execute(
text("""
DELETE FROM recommendations
WHERE model_version=:version
"""),
{"version":MODEL_VERSION}
)

print("Inserting new recommendations")

final_df.to_sql(
"recommendations",
reco_engine,
if_exists="append",
index=False
)

print("Recommendations saved")
print("-"*60)

# =========================

# STEP 9: EMAIL ALERT

# =========================

if MAIL_USERNAME and MAIL_PASSWORD:

```
msg=EmailMessage()
msg["Subject"]=f"ML Pipeline Success | {MODEL_VERSION}"
msg["From"]=MAIL_USERNAME
msg["To"]=MAIL_USERNAME

if MAIL_CC:
    msg["Cc"]=MAIL_CC

msg.set_content(f"""
```

ML Recommendation Pipeline Completed

Model Version : {MODEL_VERSION}
Users Covered : {final_df['user_id'].nunique()}
Rows Inserted : {len(final_df)}
Run Time      : {datetime.utcnow()}
""")

```
recipients=[MAIL_USERNAME]

if MAIL_CC:
    recipients.append(MAIL_CC)

with smtplib.SMTP_SSL("smtp.gmail.com",465) as server:
    server.login(MAIL_USERNAME,MAIL_PASSWORD)
    server.send_message(msg,to_addrs=recipients)

print("Email sent")
```

else:
print("Email skipped")

# =========================

# PIPELINE END

# =========================

PIPELINE_END=datetime.utcnow()

print("-"*60)
print("ML PIPELINE COMPLETED")
print("Duration:",PIPELINE_END-PIPELINE_START)
