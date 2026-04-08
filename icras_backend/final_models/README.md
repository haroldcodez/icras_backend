# Final Models - Credit Risk Assessment

Test models for credit risk assessment using traditional and alternative data.

## Overview

This folder contains the final implementation of the credit risk assessment system. It includes:

- **Baseline Model**: Uses 18 traditional credit features
- **Advanced Model**: Uses 20 mixed features (12 traditional + 8 alternative)
- **Result**: +8.2 percentage point improvement at 5% default rate (1,639 additional approvals)

## Quick Start

### Step 1: Train Baseline Model

```bash
python 02_train_baseline_model.py
```

**What it does**:

- Selects top 18 traditional features using Mutual Information
- Trains Gradient Boosting Classifier
- Evaluates on test set
- Saves serialized model

**Output files**:

- `baseline_model.pkl` - Serialized model (can be loaded with joblib)
- `baseline_model_metadata.json` - Model configuration and metrics
- `baseline_feature_selection.csv` - Feature selection results
- `baseline_feature_importance.csv` - Feature importance rankings
- `baseline_predictions.csv` - Test set predictions

### Step 2: Train Advanced Model

```bash
python 03_train_advanced_model.py
```

**What it does**:

- Selects top 20 mixed features using Mutual Information
- Trains Gradient Boosting Classifier
- Evaluates on test set
- Saves serialized model

**Output files**:

- `advanced_model.pkl` - Serialized model (can be loaded with joblib)
- `advanced_model_metadata.json` - Model configuration and metrics
- `advanced_feature_selection.csv` - Feature selection results
- `advanced_selected_features.csv` - Top 20 selected features
- `advanced_feature_importance.csv` - Feature importance rankings
- `advanced_predictions.csv` - Test set predictions

### Step 3: Evaluate and Compare

```bash
python 04_evaluate_models.py
```

**What it does**:

- Loads both models
- Finds optimal thresholds for 5% default rate
- Calculates approval rates and improvements
- Creates comparison visualizations

**Output files**:

- `evaluation_results.json` - Detailed comparison results
- `model_comparison.csv` - Comparison table
- `model_comparison.png` - Visualizations (4 plots)

## Model Details

### Baseline Model

**Features (18 traditional)**:

1. AMT_ANNUITY - Loan payment amount
2. goods_price_credit_ratio - Loan-to-value ratio
3. bureau_credit_types - Credit diversity
4. AMT_CREDIT - Credit amount
5. bureau_days_credit_mean - Credit history age
6. bureau_closed_count - Closed credit accounts
7. annuity_income_ratio - Payment-to-income ratio
8. AMT_GOODS_PRICE - Goods price
9. credit_income_ratio - Debt burden
10. bureau_active_count - Active credit accounts
11. cc_utilization - Credit card utilization
12. cc_limit_mean - Credit card limit
13. AMT_INCOME_TOTAL - Income
14. cc_balance_mean - Credit card balance
15. bureau_overdue_mean - Overdue amount
16. bureau_total_count - Total credit accounts
17. cc_dpd_max - Max days past due
18. bureau_debt_sum - Total debt

### Advanced Model

**Features (20 mixed: 12 traditional + 8 alternative)**:

**Traditional (12)**:

1. AMT_ANNUITY
2. goods_price_credit_ratio
3. bureau_credit_types
4. AMT_CREDIT
5. bureau_days_credit_mean
6. credit_income_ratio
7. AMT_GOODS_PRICE
8. annuity_income_ratio
9. bureau_closed_count
10. bureau_active_count
11. cc_limit_mean
12. cc_utilization

**Alternative (8)**:

1. prev_approved_ratio - Previous approval rate
2. FLAG_MOBIL - Mobile phone availability
3. FLAG_EMP_PHONE - Work phone availability
4. geo_stability - Geographic stability
5. inst_payment_ratio - Installment payment ratio
6. FLAG_OWN_REALTY - Real estate ownership
7. inst_dpd_mean - Days past due (installments)
8. FLAG_WORK_PHONE - Work phone availability

## Model Configuration

Both models use the same configuration for fair comparison:

```python
GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
```

**Training/Test Split**:

- Training: 80,000 samples (80%)
- Test: 20,000 samples (20%)
- Stratified by target variable
- Random state: 42 (for reproducibility)

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
joblib>=1.2.0
matplotlib>=3.6.0
```

Install with:

```bash
pip install pandas numpy scikit-learn joblib matplotlib
```
