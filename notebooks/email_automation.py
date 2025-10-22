import os
from dotenv import load_dotenv
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import timedelta

# Load environment variables from .env file
load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")

# Define email templates for each segment
email_templates = {
    "Loyal": {
        "subject": "Exclusive Reward for Our Loyal Customers!",
        "body": "Thank you for being a valued customer! Enjoy 20% off your next purchase."
    },
    "Potential": {
        "subject": "Special Offer to Keep You Shopping!",
        "body": "We’ve noticed your interest! Enjoy special recommendations and discounts today."
    },
    "At Risk": {
        "subject": "We Miss You — Come Back for 15% Off!",
        "body": "It’s been a while since your last order. We’d love to see you again with an exclusive discount."
    },
    "Need Attention": {
        "subject": "Welcome to Our Store!",
        "body": "We’re glad to have you! Here’s 10% off your next order to get you started."
    }
}


# 2. FUNCTION TO SEND EMAIL

def send_email(receiver_email, subject, body):
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
            print(f"Email sent to {receiver_email}")
    except Exception as e:
        print(f"Failed to send to {receiver_email}: {e}")


# 3. RFM ANALYSIS AND SEGMENTATION

def perform_rfm_segmentation(df):
    # Data cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C", na=False)]
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", infer_datetime_format=True)
    df = df.dropna(subset=["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Snapshot date for recency
    snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)

    # Compute RFM
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
        "InvoiceNo": "nunique",
        "Revenue": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

    # Score using quantiles
    rfm["R_Score"] = pd.qcut(rfm["Recency"].rank(method="first"), 5, labels=[5,4,3,2,1])
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1,2,3,4,5])

    rfm["RFM_Sum"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1).astype(int)

    # Segment labeling
    def label_customer(row):
        if row["RFM_Sum"] >= 13:
            return "Loyal"
        elif row["RFM_Sum"] >= 9:
            return "Potential"
        elif row["RFM_Sum"] >= 6:
            return "At Risk"
        else:
            return "Need Attention"

    rfm["Segment"] = rfm.apply(label_customer, axis=1)

    # Add dummy email for testing
    rfm["Email"] = "tejas.bhoirekar@gmail.com"  # Replace with your email or real customer emails

    return rfm


# 4. SEND EMAILS BASED ON SEGMENT

def send_segment_emails(df):
    print(f"📊 Total customers: {len(df)}")
    for _, row in df.iterrows():
        segment = row["Segment"]
        receiver = row["Email"]
        if pd.isna(receiver) or receiver.strip() == "":
            continue
        template = email_templates.get(segment)
        if template:
            send_email(receiver, template["subject"], template["body"])


# 5. MAIN EXECUTION

if __name__ == "__main__":
    print("🔍 Loading dataset...")
    df = pd.read_csv("../data/retail_dataset.csv", encoding="ISO-8859-1")
    print("Data loaded successfully!")

    rfm = perform_rfm_segmentation(df)
    print("RFM segmentation completed!")
    print(rfm.head())

    send_segment_emails(rfm)
    print("Email campaign completed!")