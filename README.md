# 📈 InvestWise: Fund Analysis & Clustering Engine

![Python](https://img.shields.io/badge/Python-Data_Science-blue)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-KMeans-orange)
![Netlify](https://img.shields.io/badge/Deployment-Netlify-00C7B7)

> **The "Market Intelligence" layer of the InvestWise system, handling fund segmentation and user interface.**

## 🧠 The Unsupervised Logic
Instead of relying on generic labels like "Large Cap," this module analyzes **1,000+ Mutual Funds** using hard mathematical metrics.

### Feature Engineering
We engineered composite financial features to capture true behavior:
- **Risk-Adjusted Strength:** Combination of Sharpe, Sortino, and Alpha.
- **Momentum:** 1Y and 3Y return trajectories.
- **Volatility Profile:** Standard Deviation relative to Category benchmarks.

### Clustering (K-Means)
Funds are segmented into dynamic clusters:
- **Cluster 0 (Safe Harbors):** Low Volatility, Stable Returns.
- **Cluster 1 (Growth Engines):** High Alpha, High Beta.
- **Cluster 2 (Underperformers):** High Expense Ratio, Negative Alpha.

## 💻 Frontend (Web Interface)
- Built with **HTML/CSS/JS**.
- Hosted on **Netlify**.
- Consumes the **FastAPI Risk Engine** (Repo 4) to display recommendations.

## 📂 Project Structure
```bash
├── analysis/           # Jupyter Notebooks for EDA and Clustering
├── data/              # Scraped Mutual Fund Data (1000+ schemes)
├── web-interface/     # Frontend code deployed to Netlify
└── models/            # Serialized K-Means models
