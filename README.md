# AI/ML Project Framework

A comprehensive Python-based AI/ML framework with modular architecture for machine learning and deep learning projects.

## Project Structure

```
.
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── features/          # Feature engineering and selection
│   ├── models/            # ML models (classifiers, regressors, neural networks)
│   ├── training/          # Model training and hyperparameter tuning
│   ├── evaluation/        # Model evaluation and visualization
│   └── utils/             # Utilities (logging, config, helpers)
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data files
├── models/               # Saved model files
├── logs/                 # Log files
├── results/              # Results and visualizations
├── config.yaml           # Configuration file
├── requirements.txt      # Python dependencies
├── main.py              # Main entry point
└── example_usage.py     # Usage examples
```

## Features

### Data Processing
- Multiple data format support (CSV, JSON, Excel, Parquet, NumPy)
- Missing value handling
- Outlier removal (IQR and Z-score methods)
- Feature scaling (Standard and MinMax)
- Categorical encoding (Label and One-Hot)

### Feature Engineering
- Polynomial features
- Interaction features
- Binned features
- Log transformations
- Time-based features
- Aggregated features

### Feature Selection
- K-Best selection
- Mutual information
- Tree-based importance
- Recursive Feature Elimination (RFE)

### Models
- **Classification**: Random Forest, Gradient Boosting, Logistic Regression, SVM, KNN, Naive Bayes, XGBoost, LightGBM
- **Regression**: Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, ElasticNet, SVR, KNN, XGBoost, LightGBM
- **Neural Networks**: MLP, CNN, LSTM (TensorFlow/Keras)

### Training
- Cross-validation
- Hyperparameter tuning (Grid Search, Random Search)
- Model persistence

### Evaluation
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Regression metrics (MSE, RMSE, MAE, R2, MAPE)
- Visualization (Confusion Matrix, ROC Curve, Feature Importance, Learning Curves, Residuals)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from src.data import DataLoader, DataPreprocessor
from src.models import MLClassifier
from src.evaluation import ModelEvaluator

# Load and preprocess data
loader = DataLoader()
X_train, X_test, y_train, y_test = loader.split_data(X, y)

preprocessor = DataPreprocessor()
X_train_scaled = preprocessor.scale_features(X_train)
X_test_scaled = preprocessor.scale_features(X_test)

# Train model
classifier = MLClassifier(model_type='random_forest')
classifier.train(X_train_scaled, y_train)

# Evaluate
y_pred = classifier.predict(X_test_scaled)
evaluator = ModelEvaluator(task='classification')
metrics = evaluator.evaluate_classification(y_test, y_pred)
```

### Run Examples

```bash
python example_usage.py
```

### Run Main Application

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize project settings:
- Data preprocessing parameters
- Model hyperparameters
- Training configuration
- Evaluation metrics
- File paths

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- XGBoost
- LightGBM
- Matplotlib
- Seaborn

## License

MIT License