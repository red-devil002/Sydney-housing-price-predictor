# Sydney Housing Price Prediction and Decision Support System

SIT307 Machine Learning, Task 8.1D (Distinction mini project), Deakin University.

Predicts property sale prices in three contrasting Sydney markets (Chatswood, Castle Hill and Blacktown) and serves the model through a Streamlit web app.

## Repository contents

| File | Description |
|---|---|
| `sydney_sales.csv` | Dataset: 103 sold properties (102 after cleaning) |
| `housing_price_prediction.ipynb` | Full workflow: cleaning, EDA, feature engineering, modelling, error analysis |
| `app.py` | Streamlit app (single property form + CSV batch upload) |
| `housing_model.joblib` | Trained Ridge pipeline, created by the notebook |
| `requirements.txt` | Python dependencies (pinned scikit-learn version) |
| `test_properties.csv` | Sample input for the app's CSV upload |

## Dataset

- **Suburbs:** Chatswood (34), Castle Hill (34), Blacktown (35)
- **Period:** October 2025 to September 2026
- **Target:** `price` (sale price, AUD)
- **Features:** suburb, property type, bedrooms, bathrooms, car spaces, land size (m²)
- **Source:** <!-- Describe how and where the data was collected -->

## Setup

Requires Python 3.12+.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the notebook

```bash
pip install jupyter matplotlib
jupyter notebook housing_price_prediction.ipynb
```

Use **Kernel → Restart & Run All**. The final cell saves `housing_model.joblib`.

## Running the app

```bash
streamlit run app.py
```

Open http://localhost:8501.

- **Single property:** choose suburb and property type, enter bedrooms, bathrooms, car spaces and land size, then click **Estimate price**.
- **Upload a CSV:** upload a file with the columns `suburb, property_type, bedrooms, bathrooms, car_spaces, land_size_m2` (leave land size blank if unknown), then download the predictions.

The model must be loaded with the same scikit-learn version it was saved with (see `requirements.txt`).

## Results

5-fold cross-validation, repeated 5 times, with log(price) as the target:

| Model | CV R² | CV MAE ($) |
|---|---|---|
| Baseline (mean) | −0.09 | 1,281,713 |
| **Ridge (selected)** | **0.925** | **413,671** |
| Random Forest | 0.917 | 425,071 |
| KNN | 0.785 | 617,129 |

Median absolute prediction error by suburb: Blacktown 8.0%, Castle Hill 9.1%, Chatswood 28.9%.

## Limitations

- Small dataset (102 sales); few premium or unusual properties.
- Chatswood sample is almost entirely houses, so Chatswood unit predictions are unvalidated.
- No information on condition, renovation, views, lot type or street position.
- Predictions for Chatswood and properties above $3.5M should be treated as rough guides.

## GenAI acknowledgement

<!-- Describe how GenAI tools were used, e.g. planning, code drafting, debugging, feedback -->

## Author

Swetang Pandit (GitHub: [@red-devil002](https://github.com/red-devil002))