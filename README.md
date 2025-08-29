# Binary Classification Banking Dataset - Kaggle Competition

A machine learning project focused on predicting whether bank clients will subscribe to a term deposit based on their demographic and campaign interaction data.

## 🎯 Project Overview

This project implements various machine learning algorithms to solve a binary classification problem in the banking domain. The goal is to predict whether a client will subscribe to a term deposit (`y=1`) or not (`y=0`) based on their personal information and previous campaign interactions.

### 🏆 Key Achievements
- **ROC AUC Score**: 0.9206 (Logistic Regression)
- **Best F1-Score**: 0.6018 with optimized threshold
- **Multiple Models**: Logistic Regression, XGBoost implementations
- **Robust Pipeline**: Complete preprocessing and validation framework

## 📊 Dataset Description

The dataset contains information about bank marketing campaigns with **750,000 training samples** and **250,000 test samples**.

### Features Description

#### Demographic Information
- **age**: Age of the client (numeric)
- **job**: Type of job (categorical: "admin.", "blue-collar", "entrepreneur", etc.)
- **marital**: Marital status (categorical: "married", "single", "divorced")
- **education**: Level of education (categorical: "primary", "secondary", "tertiary", "unknown")

#### Financial Information
- **default**: Has credit in default? (categorical: "yes", "no")
- **balance**: Average yearly balance in euros (numeric)
- **housing**: Has a housing loan? (categorical: "yes", "no")
- **loan**: Has a personal loan? (categorical: "yes", "no")

#### Campaign Information
- **contact**: Type of communication contact (categorical: "unknown", "telephone", "cellular")
- **day**: Last contact day of the month (numeric, 1-31)
- **month**: Last contact month of the year (categorical: "jan", "feb", "mar", etc.)
- **duration**: Last contact duration in seconds (numeric)
- **campaign**: Number of contacts performed during this campaign (numeric)
- **pdays**: Number of days since last contact from previous campaign (numeric; -1 = not previously contacted)
- **previous**: Number of contacts performed before this campaign (numeric)
- **poutcome**: Outcome of previous marketing campaign (categorical: "unknown", "other", "failure", "success")

#### Target Variable
- **y**: Whether the client subscribed to a term deposit (binary: 0/1)

## 🛠️ Technical Implementation

### Data Preprocessing Pipeline

1. **Exploratory Data Analysis**
   - Statistical summaries and distributions
   - Class imbalance analysis (~88% class 0, ~12% class 1)
   - Missing value detection (no missing values found)
   - Outlier detection using box plots

2. **Feature Engineering**
   - **Label Encoding**: Categorical variables converted to numerical labels
   - **Standardization**: All features scaled using StandardScaler (mean=0, std=1)
   - **Train-Validation Split**: 70-30 stratified split for model evaluation

3. **Feature Analysis**
   - Categorical variable frequency analysis
   - Outlier detection in numerical features
   - Feature importance ranking

### Model Development

#### 1. Logistic Regression (Baseline)
```python
# Key Performance Metrics
ROC AUC: 0.9206
Optimal Threshold: 0.20
Validation Accuracy: 0.8826
Precision: 0.5094
Recall: 0.7350
F1-Score: 0.6018
```

#### 2. XGBoost Implementation
- Advanced gradient boosting approach
- Hyperparameter optimization
- Feature importance analysis

### Model Evaluation Strategy

- **ROC AUC**: Primary metric for imbalanced classification
- **Precision-Recall AUC**: Additional metric for minority class focus
- **Threshold Optimization**: Custom threshold selection for optimal F1-score
- **Cross-Validation**: 5-fold stratified cross-validation
- **Multiple Submissions**: Different threshold strategies tested

## 📁 Project Structure

```
kaggle-binary-class-banking/
├── README.md                           # Project documentation
├── binClassBank.ipynb                  # Main analysis notebook
├── xgboost-understainding.ipynb       # XGBoost implementation from one of the competitors to learn from them
├── train.csv                          # Training dataset (750K samples)
├── test.csv                           # Test dataset (250K samples)
├── sample_submission.csv              # Kaggle submission format
├── xgb_bank_binary.json              # XGBoost model configuration
└── venv/                              # Python virtual environment
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.9+
Jupyter Notebook
```

### Installation
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd kaggle-binary-class-banking
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis
1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open main notebook**
   ```bash
   binClassBank.ipynb
   ```

3. **Run cells sequentially** for complete analysis pipeline

## 📈 Results Summary

### Model Performance Comparison

| Model | ROC AUC | Accuracy | Precision | Recall | F1-Score | Threshold |
|-------|---------|----------|-----------|---------|----------|-----------|
| Logistic Regression | **0.9206** | 0.8826 | 0.5094 | 0.7350 | **0.6018** | 0.20 |
| XGBoost | TBD | TBD | TBD | TBD | TBD | TBD |

### Key Insights

1. **Class Imbalance Impact**: Significant class imbalance (88:12 ratio) requires careful threshold optimization
2. **Feature Importance**: Duration, campaign interactions, and demographic factors are key predictors
3. **Threshold Optimization**: Using threshold 0.20 instead of default 0.50 significantly improves F1-score
4. **ROC AUC Excellence**: 0.9206 ROC AUC indicates excellent discriminative ability

## 🔍 Key Findings

### Data Insights
- **No missing values** across all features
- **Significant outliers** in numerical features (balance, duration, campaign)
- **Class imbalance** requires specialized evaluation metrics
- **Categorical variables** show varied distributions and importance

### Model Insights
- **Logistic Regression** provides excellent baseline performance
- **Threshold optimization** crucial for imbalanced classification
- **Feature standardization** essential for algorithm performance
- **Cross-validation** confirms model stability

## 🎯 Future Improvements

1. **Advanced Feature Engineering**
   - Polynomial features
   - Feature interactions
   - Domain-specific transformations

2. **Model Ensemble**
   - Voting classifiers
   - Stacking approaches
   - Blending strategies

3. **Hyperparameter Optimization**
   - Grid search
   - Random search
   - Bayesian optimization

4. **Advanced Algorithms**
   - Random Forest
   - Neural Networks
   - LightGBM

## 📊 Submission Files

- `submission_lr.csv`: Logistic regression predictions (threshold 0.20)
- `submission_xgb.csv`: XGBoost binary predictions
- `submission_xgb_proba.csv`: XGBoost probability predictions

## 🏆 Competition Strategy

1. **Robust Validation**: 70-30 split with stratification
2. **Metric Focus**: ROC AUC as primary metric for ranking
3. **Threshold Tuning**: Optimize for competition-specific requirements
4. **Multiple Submissions**: Test various approaches and thresholds

## 📝 Notes

- **Reproducibility**: All random seeds set to 42
- **Scalability**: Pipeline designed for large datasets
- **Documentation**: Comprehensive code comments and markdown explanations
- **Validation**: Thorough testing and verification of preprocessing steps

## 🤝 Contributing

Feel free to contribute by:
- Improving model performance
- Adding new algorithms
- Enhancing feature engineering
- Optimizing code efficiency

## 📄 License

This project is for educational and competition purposes.

---

**Author**: Sarosh Farhan  
**Project**: Kaggle Binary Classification Banking Dataset  
**Last Updated**: December 2024