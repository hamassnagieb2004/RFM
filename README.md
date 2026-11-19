# Online Retail — RFM Segmentation & Market Basket Analysis

Customer and product analysis of a UK-based online retailer (01/12/2010 – 09/12/2011) using Python and interactive Plotly charts.

## What the notebook does
1. **Data cleaning** with documented decisions: non-product lines, descriptions, zero prices, duplicates, returns.
2. **Interactive EDA**: sales by month/hour/day, countries, basket size, Pareto, returns.
3. **Customer RFM**: 4 main segments (Champions, Loyal, At Risk, Lost) built from 11 sub-segments.
4. **Product RFM**: Stars, Dead Stock and more, linked to the customer segments.
5. **Market basket analysis (Apriori)**: which products are bought together, overall and for Champions.

## Data
UCI *Online Retail* dataset: https://archive.ics.uci.edu/dataset/352/online+retail

Download `Online Retail.xlsx` into the project folder. The first cells of the notebook show how to convert it once to `online_retail_raw.csv` (much faster to load).

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pip install mlxtend           # needed for Apriori, if not already in requirements.txt
jupyter notebook RFM_apriori.ipynb
```

## Notes
- Plotly charts are interactive only when the notebook is run locally; GitHub shows the static parts.
- Apriori ignores quantity and price, and wholesale invoices can inflate support (see section 10.4).
