# Week 2 House Price Prediction

A machine learning project for predicting residential house sale prices using the Kaggle **House Prices: Advanced Regression Techniques** dataset.

The project demonstrates data inspection, missing-value handling, categorical encoding, linear regression, log-target transformation, validation, and Kaggle submission generation.

## Project Result

- Kaggle submission score: **0.15566**
- Evaluation metric: Root Mean Squared Error between logarithmic predicted and observed prices
- Model: Linear Regression trained on `log1p(SalePrice)`
- Validation split: 80% training and 20% validation
- Validation random seed: `42`

The Kaggle score comes from the submitted `submission.csv` file. Lower scores are better.

## Repository Files

- `mini_project2_house_prediction.ipynb` — Complete Jupyter/Google Colab notebook
- `train.csv` — Kaggle training data
- `test.csv` — Kaggle test data
- `submission.csv` — Predictions submitted to Kaggle

## Dataset

The project uses the Ames Housing dataset provided through Kaggle’s House Prices competition.

Dataset dimensions:

- Training data: 1,460 rows and 81 columns
- Test data: 1,459 rows and 80 columns
- Training target: `SalePrice`
- Identifier column: `Id`
- Final encoded feature count: 302

The dataset contains numeric and categorical housing features such as:

- Overall quality and condition
- Living area and lot area
- Year built and year remodeled
- Neighborhood
- Basement, garage, kitchen, and fireplace information
- Sale type and sale condition

## Workflow

The notebook follows these steps:

1. Load `train.csv` and `test.csv`.
2. Inspect dataset dimensions and missing values.
3. Replace missing categorical values that represent absent features with `"none"`.
4. Fill selected numeric values with training-set medians.
5. Fill remaining categorical values with training-set modes.
6. Combine training and test features for consistent one-hot encoding.
7. Convert categorical columns with `pandas.get_dummies()`.
8. Split the training data into training and validation sets.
9. Train a standard linear regression model.
10. Train a second model using `log1p(SalePrice)`.
11. Convert log-scale predictions back to regular prices with `expm1()`.
12. Retrain the final log-target model on all training rows.
13. Generate predictions for the test data.
14. Save predictions to `submission.csv`.

## Validation Results

| Model | Target Used | R² Score | RMSE |
|---|---|---:|---:|
| Linear Regression | Raw `SalePrice` | 0.4411 | 65,474.24 |
| Linear Regression | `log1p(SalePrice)` | 0.9319 | 22,861.19 |

The log-target model performed better on the notebook’s validation split because house prices are strongly skewed. The validation RMSE shown above is measured in regular price units, while Kaggle evaluates logarithmic errors, so the two values should not be compared directly.

## Submission Format

The generated submission contains two columns:

```text
Id,SalePrice
1461,121289.24483435
1462,165747.76108124
1463,181988.93383844
```

The file contains 1,459 predictions, one for every row in `test.csv`.

## How to Run

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

Place these files in the same folder:

```text
mini_project2_house_prediction.ipynb
train.csv
test.csv
```

Open the notebook:

```bash
jupyter notebook mini_project2_house_prediction.ipynb
```

Run the notebook cells in order. The submission file is created in the notebook’s working directory.

The final Google Colab download cell uses `google.colab.files`. When running locally, that cell can be skipped after `submission.csv` has been created.

## Kaggle Competition

Competition page:

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

Kaggle evaluates submissions using the root mean squared error between the logarithms of predicted and actual sale prices.

## Data Access and Licensing

The Kaggle data page currently displays an MIT label for this dataset. However, access to the competition data requires accepting Kaggle’s current competition rules.

Before publishing this repository publicly, review the live competition rules and confirm that they allow `train.csv` and `test.csv` to be redistributed through GitHub. If the applicable rules prohibit redistribution, remove those CSV files from the public repository and provide instructions for readers to download the data directly from Kaggle after accepting the rules.

The Kaggle competition terms and data-page conditions take precedence over this README.

## Limitations

- The notebook uses a single random validation split.
- The final model is a basic linear regression model.
- The `Id` column remains among the model features.
- No hyperparameter tuning or ensemble modeling is included.
- The Kaggle score is based on one submitted prediction file.
- Validation results may vary with a different split or preprocessing approach.

## Acknowledgements

- Kaggle for hosting the House Prices competition
- Dean De Cock for compiling the Ames Housing dataset
- The pandas, NumPy, Matplotlib, and scikit-learn projects

