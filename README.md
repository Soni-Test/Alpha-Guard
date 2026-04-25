# 🛡️ Alpha-Guard: Automated Market Risk & Sentiment Engine

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon_Cloud-336791?logo=postgresql&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=github-actions&logoColor=white)
![Power BI](https://img.shields.io/badge/Visualization-Power_BI-F2C811?logo=powerbi&logoColor=black)

## 📌 Project Overview
Alpha-Guard is a fully automated, cloud-hosted end-to-end Data Engineering pipeline. It extracts live stock market data and financial news feeds, processes statistical risk metrics and NLP sentiment scores, and loads the transformed data into a remote PostgreSQL database for live Business Intelligence reporting.

The architecture is entirely serverless, relying on **GitHub Actions** for CI/CD orchestration to execute daily micro-batch ETL jobs with zero manual intervention.

## 🏗️ Cloud Architecture & Data Flow

1. **Extract:** Pulls historical tick data via `yfinance` and financial news headlines via Yahoo RSS XML feeds.
2. **Transform (Python):** - Calculates 20-day rolling volatility and flags 3-sigma price anomalies.
   - Applies Natural Language Processing (NLP) using `vaderSentiment` to score headline sentiment (-1.0 to 1.0).
3. **Load (SQLAlchemy):** Pushes cleaned, structured data into a **Neon Serverless PostgreSQL** database.
4. **Automate (CI/CD):** Scheduled cron jobs via GitHub Actions run the pipeline every weekday at 10:00 PM UTC.
5. **Visualize:** A scheduled Power BI Semantic Model fetches the fresh cloud data to update the dashboard automatically.

## 🛠️ Tech Stack & Skills Demonstrated
* **Languages:** Python, SQL
* **Libraries:** `pandas`, `SQLAlchemy`, `vaderSentiment`, `nltk`, `yfinance`
* **Cloud & Infrastructure:** GitHub Actions (Runner & Secrets Vault), Neon.tech (Cloud DB)
* **Security:** Decoupled architecture using runtime Environment Variables to secure database credentials.
* **Analytics:** Power BI (Scheduled Refresh)

## 🚀 Key Engineering Features
* **Resilient Automation:** Replaced manual local execution with a containerized Ubuntu GitHub runner.
* **Secure Credential Management:** Database URIs are never hardcoded. Handled dynamically via `os.environ` and GitHub Secrets.
* **Schema Handling:** Implemented protocol standardizations (auto-converting `postgres://` to `postgresql://` for SQLAlchemy compatibility).
* **Fault Tolerance:** Built-in try/except blocks to prevent broken RSS XML nodes from failing the entire pipeline.

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Soni-Test/Alpha-Guard.git](https://github.com/Soni-Test/Alpha-Guard.git)
   cd Alpha-Guard
