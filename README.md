
# Toronto Island Ferry Forecasting Project

## Problem Statement

The City of Toronto needs reliable daily forecasts for **ferry ticket redemptions** and **sales** to support operations and planning. The original approach was based only on yearly seasonality, which is not responsive to recent demand trends or uncertainty.

---

## Project Outcomes

* **Improved forecasting model** for daily redemptions, using machine learning and recent demand patterns.
* **New forecasting model** for daily ticket sales, built with the same advanced techniques.
* **Prediction intervals** to show not just forecasts but also the likely range of outcomes for each day.
* All work done with **free and open source tools** for full reproducibility.

---

## Approach

* **Data Preparation**
  Cleaned and formatted the data, removed missing/zero values, and created features to capture both short-term and weekly patterns (lag, 7-day rolling mean, day of week).

* **Modeling**
  For both redemptions and sales, built and compared:

  * A **base seasonal model** (replicates the company’s original approach)
  * **Linear regression** using new features
  * **Random Forest regression** for improved accuracy
  * **Quantile Random Forest (QRF)** to add prediction intervals

* **Validation**
  Used rolling time-based cross-validation to ensure realistic testing and prevent future data leakage.

* **Evaluation**
  Compared models using Mean Absolute Percentage Error (MAPE), coverage (interval accuracy), and average interval width.

* **Conclusion**
  The Random Forest and Quantile Random Forest models with new features provided the best and most practical forecasts, with much lower error than the seasonal baseline and useful uncertainty estimates for planning.

---

## Files in This Repo

* **RedemptionModel.py**
  Python module for forecasting daily redemptions. Includes data cleaning, feature engineering, model training, prediction intervals, plotting, and evaluation.

* **SalesForecastModel.py**
  Python module for forecasting daily sales, built on the same advanced structure as the redemption model.


---

## How to Run

1. Make sure you have **Python 3.x** and the following packages (all free):
   `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `statsmodels`, `tqdm`

   Install with:

   ```bash
   pip install numpy pandas scikit-learn matplotlib statsmodels tqdm
   ```

2. Place your ticket data CSV (with `Timestamp`, `Redemption Count`, and `Sales Count` columns) in the same folder.

3. To run the redemption forecasting model:

   ```bash
   python RedemptionModel.py
   ```

   To run the sales forecasting model:

   ```bash
   python SalesForecastModel.py
   ```

4. The code will print detailed progress, show prediction plots, and output a summary table with model performance.

---

## Why is this better than the baseline?

* **More accurate**: Uses recent and weekly trends, not just last year’s pattern.
* **Flexible**: Advanced models adapt to sudden changes in demand.
* **Transparent**: Shows both the prediction and the confidence range.
* **Reproducible**: All code and tools are open source and free to use.

---

## Contact

Questions or suggestions? Please [open an issue](https://github.com/) or contact the project owner.




