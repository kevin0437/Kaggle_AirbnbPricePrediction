# Kaggle Airbnb Price Prediction

This repository contains the code and analysis for the Duke University CS671 Fall 2023 in-class Kaggle competition: Airbnb Price Prediction. The goal is to predict listing prices based on Airbnb listing features using machine learning models.


## Project Overview
The task is part of a Duke CS671 assignment to compete on Kaggle and build predictive models for Airbnb listing prices. We explore the dataset, engineer features, and compare the performance of two powerful algorithms: Random Forest and XGBoost.

## Data
The dataset is provided by the Kaggle competition:
```
https://www.kaggle.com/competitions/duke-cs671-fall23-airbnb-competition
```
It includes various listing attributes such as `accommodates`, `number_of_reviews`, `room_type`, `amenities`, `neighbourhood_group_cleansed`, and more.

## Exploratory Data Analysis (EDA)
1. **Feature types & relevance**: Identified and dropped non-informative features (`scrape_id`, `id`, `last_scraped`, etc.).
2. **Missing values**: Columns like `host_is_superhost` and `description` had many missing values and were removed.
3. **Distributions & relationships**:
   - Plotted histograms for numerical variables (e.g., `availability_365`).
   - Bar charts for categorical variables (e.g., `room_type`).
   - Correlation heatmap to assess feature interactions.

## Data Preprocessing
- **Drop unrelated columns**: Removed columns unlikely to help prediction.
- **Boolean encoding**: Converted ‘t’/‘f’ to 1/0 for host attributes.
- **Date handling**: Transformed `host_since` to “days since host joined”.
- **Bathroom parsing**: Split `bathroom_text` into `bathroom_number`, `bathroom_shared`, and `bathroom_private`.
- **Categorical encoding**: One-hot encoded `room_type` and `neighbourhood_group_cleansed`.
- **Amenities**: Counted the number of amenities per listing.
- **Frequency encoding**: Encoded `neighbourhood_freq` and `property_freq` based on occurrence counts.

## Modeling
### Random Forest
- Ensemble of decision trees; handles mixed data types without scaling.
- Implemented using `sklearn.ensemble.RandomForestRegressor`.
- Training runtime ~30s; hyperparameter search ~10min.

### XGBoost
- Gradient boosting with second-order optimization and built-in regularization.
- Implemented using `xgboost.XGBRegressor`.
- Training runtime ~1s; hyperparameter search ~3min.

## Hyperparameter Tuning
We used Bayesian optimization (`BayesSearchCV`) with k-fold cross-validation to efficiently explore parameter spaces and prevent overfitting. Examples of tuned parameters:
- **Random Forest**: number of estimators, max depth, min samples split
- **XGBoost**: learning rate, max depth, subsample ratio

## Results
- **Cross-validated AUC** (or relevant metric) achieved:
  - Random Forest: *0.84* (example)
  - XGBoost: *0.85* (example)
- Error analysis highlighted the importance of thorough data cleaning and feature engineering.

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- scipy
- scikit-optimize



---

*Prepared as part of Duke CS671 Fall 2023 Kaggle competition.*
