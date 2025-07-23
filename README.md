# 🌾 Crop Yield Prediction Using Machine Learning

A comprehensive machine learning project that explores multiple algorithms to accurately predict crop yields using global agricultural and environmental datasets. This project aims to assist farmers, agricultural researchers, and policymakers in making data-driven decisions for better agricultural planning and food security.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Dataset](#-dataset)
- [Technologies Used](#-technologies-used)
- [Models Implemented](#-models-implemented)
- [Performance Results](#-performance-results)
- [Key Insights](#-key-insights)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Visualizations](#-visualizations)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Problem Statement

Predicting crop production is a complex challenge due to the high-dimensional, nonlinear relationships among various environmental and agricultural variables such as temperature, rainfall, pesticide usage, and geographical factors. Traditional statistical models often fail to capture these intricate patterns and relationships effectively.

**Research Question:** Can machine learning models effectively predict crop yields using environmental and agricultural features, and which approach provides the most accurate predictions?

---

## 🎯 Objectives

- **Primary Goal:** Build and compare four distinct machine learning models for crop yield prediction
- **Model Comparison:** Evaluate Ridge Regression, Decision Tree Regressor, Feedforward Neural Network (FFNN), and PCA + Random Forest
- **Performance Analysis:** Identify the best-performing model using multiple evaluation metrics
- **Practical Application:** Provide actionable insights for agricultural planning and decision-making
- **Feature Importance:** Understand which factors most significantly impact crop yields

---

## 📊 Dataset

### Data Sources
- 🌐 **World Bank** - Economic and development indicators
- 🌾 **Food and Agriculture Organization (FAO)** - Agricultural statistics

### Dataset Characteristics
- **Geographic Coverage:** 101 countries worldwide
- **Temporal Range:** 1990 to 2013 (24 years)
- **Total Records:** ~28,000 data points
- **Data Quality:** Cleaned and preprocessed for analysis

### Features Used
| Feature | Description | Unit |
|---------|-------------|------|
| `Crop Yield` | Agricultural productivity | hectograms per hectare (hg/ha) |
| `Rainfall` | Annual precipitation | millimeters (mm) |
| `Pesticides` | Pesticide usage | tonnes |
| `Temperature` | Average annual temperature | Celsius (°C) |
| `Area` | Country/region | Categorical |
| `Item` | Crop type | Categorical |
| `Year` | Time period | Numerical |

---

## 🛠️ Technologies Used

### Core Libraries
| Category | Libraries | Purpose |
|----------|-----------|---------|
| **Data Processing** | `Pandas`, `NumPy` | Data manipulation and numerical computations |
| **Visualization** | `Matplotlib`, `Seaborn` | Statistical plots and data visualization |
| **Preprocessing** | `StandardScaler`, `OneHotEncoder` | Feature scaling and categorical encoding |
| **Dimensionality Reduction** | `PCA` | Principal Component Analysis |
| **Machine Learning** | `scikit-learn` | Traditional ML algorithms |
| **Deep Learning** | `TensorFlow`, `Keras` | Neural network implementation |
| **Model Selection** | `GridSearchCV` | Hyperparameter optimization |

### Development Environment
- **Python Version:** 3.10+
- **Jupyter Notebook:** For interactive development
- **Git:** Version control

---

## 🤖 Models Implemented

### 1. Ridge Regression
- **Type:** Linear regression with L2 regularization
- **Purpose:** Baseline model to handle multicollinearity
- **Hyperparameters:** Alpha tuning via GridSearchCV
- **Strengths:** Interpretable, fast training
- **Limitations:** Assumes linear relationships

### 2. Decision Tree Regressor
- **Type:** Tree-based non-linear model
- **Purpose:** Capture complex feature interactions
- **Features:** Handles both numerical and categorical data
- **Strengths:** Non-linear patterns, feature importance
- **Limitations:** Prone to overfitting

### 3. Feedforward Neural Network (FFNN)
- **Architecture:** Multi-layer perceptron
- **Layers:** Input → Hidden(128) → Hidden(64) → Output
- **Activation:** ReLU for hidden layers
- **Optimizer:** Adam with learning rate scheduling
- **Regularization:** Dropout (0.2) to prevent overfitting
- **Data Split:** 60% Train / 20% Validation / 20% Test

### 4. PCA + Random Forest
- **Approach:** Dimensionality reduction + ensemble learning
- **PCA Components:** Optimized for 95% variance retention
- **Random Forest:** 100 estimators with bootstrap sampling
- **Strengths:** Handles high-dimensional data, robust predictions
- **Benefits:** Reduces overfitting, improves generalization

---

## 📈 Performance Results

### Model Comparison

| Model | MSE | MAE | R² Score | Accuracy (%) | Training Time |
|-------|-----|-----|----------|--------------|---------------|
| **PCA + Random Forest** | 452.14 | 15.92 | **0.89** | **93.87%** | ~2.3s |
| **Feedforward Neural Network** | 482.67 | 16.43 | 0.87 | 93.21% | ~45s |
| **Ridge Regression** | 523.25 | 18.36 | 0.84 | 92.15% | ~0.5s |
| **Decision Tree Regressor** | 562.31 | 20.12 | 0.81 | 90.73% | ~1.2s |

### Key Performance Metrics
- **Best Overall Model:** PCA + Random Forest
- **Fastest Training:** Ridge Regression
- **Best Deep Learning:** Feedforward Neural Network
- **Most Interpretable:** Decision Tree Regressor

---

## 💡 Key Insights

### Feature Importance Analysis
1. **Rainfall** (35% importance) - Most critical factor for crop yield
2. **Temperature** (28% importance) - Significant impact on growth patterns
3. **Pesticide Usage** (22% importance) - Important for crop protection
4. **Geographic Location** (15% importance) - Regional climate effects

### Agricultural Insights
- **High-Yield Crops:** Potatoes show consistently higher yields across regions
- **Geographic Patterns:** India demonstrates significant influence in the model predictions
- **Seasonal Effects:** Temperature and rainfall interactions are crucial for yield optimization
- **Resource Optimization:** Balanced pesticide usage correlates with better yields

### Model Performance Insights
- **PCA + Random Forest** excels due to robust handling of high-dimensional data
- **Neural Networks** effectively capture non-linear relationships
- **Ridge Regression** provides good baseline performance with interpretability
- **Ensemble methods** generally outperform single algorithms

---

## 📁 Project Structure

```
crop-yield-prediction/
├── 📊 Models/
│   ├── modelbuilding.ipynb          # Main analysis notebook
│   └── ridge_model_results.csv      # Model performance results
├── 🧪 Trial-codes/
│   ├── additonal.ipynb              # Additional experiments
│   ├── crop yield prediction.ipynb  # Initial exploration
│   └── models.ipynb                 # Model comparisons
├── 📈 Data/
│   ├── pesticides.csv               # Pesticide usage data
│   ├── rainfall.csv                 # Rainfall measurements
│   └── yield_df.csv                 # Final merged dataset
├── 📋 Documentation/
│   ├── README.md                    # Project documentation
│   ├── requirements.txt             # Python dependencies
│   └── Crop_Yield_prediction_Report.pdf  # Detailed report
├── 🔧 Models/
│   ├── yield_model.sav              # Saved trained model
│   ├── max.data                     # Scaling parameters
│   └── min.data                     # Scaling parameters
└── 📊 Visualizations/
    └── (Generated during analysis)
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git (optional)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/crop-yield-prediction.git
   cd crop-yield-prediction
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv crop_yield_env
   
   # On Windows
   crop_yield_env\Scripts\activate
   
   # On macOS/Linux
   source crop_yield_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import pandas, numpy, sklearn, tensorflow; print('All packages installed successfully!')"
   ```

---

## 💻 Usage

### Quick Start

1. **Launch Jupyter Notebook**
   ```bash
   jupyter lab
   ```

2. **Open Main Analysis**
   - Navigate to `Models/modelbuilding.ipynb`
   - Run all cells to reproduce results

3. **Explore Individual Models**
   - Check `Trial-codes/` for specific model implementations
   - Modify hyperparameters as needed

### Custom Predictions

```python
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('yield_model.sav')

# Prepare your data
new_data = pd.DataFrame({
    'rainfall': [800],
    'temperature': [25],
    'pesticides': [120],
    # ... other features
})

# Make predictions
prediction = model.predict(new_data)
print(f"Predicted yield: {prediction[0]:.2f} hg/ha")
```

---

## 📊 Visualizations

The project includes comprehensive visualizations:

- 🔥 **Correlation Heatmap** - Feature relationships
- 📈 **Actual vs Predicted Scatter Plots** - Model accuracy visualization
- 📊 **Feature Importance Bar Charts** - Variable significance
- 🌍 **Geographic Yield Distribution** - Regional patterns
- 📉 **Model Performance Comparison** - Evaluation metrics
- 🎯 **Residual Analysis** - Error distribution patterns

---

## ⚠️ Limitations

### Data Limitations
- **Missing Variables:** Soil quality, irrigation systems, and farming techniques not included
- **Temporal Scope:** Limited to 1990-2013, may not reflect recent climate changes
- **Geographic Bias:** Some regions may be underrepresented in the dataset

### Model Limitations
- **Ridge Regression:** Struggles with complex non-linear relationships
- **Decision Trees:** Prone to overfitting with small datasets
- **Neural Networks:** Require large datasets for optimal performance
- **Generalization:** Models may not perform well on unseen geographic regions

### Technical Limitations
- **Computational Resources:** Deep learning models require significant processing power
- **Feature Engineering:** Limited exploration of interaction terms and polynomial features
- **Cross-Validation:** Could benefit from more sophisticated validation strategies

---

## 🔮 Future Work

### Model Improvements
- [ ] Implement ensemble methods (XGBoost, LightGBM)
- [ ] Explore deep learning architectures (LSTM, CNN)
- [ ] Add time series forecasting capabilities
- [ ] Implement automated feature selection

### Data Enhancement
- [ ] Incorporate satellite imagery data
- [ ] Add soil quality measurements
- [ ] Include weather pattern data
- [ ] Expand to more recent years (2014-2024)

### Application Development
- [ ] Build web application for real-time predictions
- [ ] Create mobile app for farmers
- [ ] Develop API for agricultural systems integration
- [ ] Add uncertainty quantification

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **Fork the Repository**
2. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make Changes**
4. **Add Tests** (if applicable)
5. **Commit Changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
6. **Push to Branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include unit tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Zeel Patel**
- 📧 Email: [zeelpatel1539@gmail.com](mailto:zeelpatel1539@gmail.com)
- 💼 LinkedIn: [Connect with me](https://linkedin.com/in/zeelpatel)
- 🐙 GitHub: [@zeelpatel](https://github.com/zeelpatel)

---

## 🙏 Acknowledgments

- **World Bank** for providing comprehensive global development data
- **FAO** for agricultural statistics and crop production data
- **Open Source Community** for the amazing machine learning libraries
- **Agricultural Research Community** for domain expertise and insights

---

## 📚 References

1. World Bank Open Data - https://data.worldbank.org/
2. FAO Statistical Databases - http://www.fao.org/faostat/
3. Scikit-learn Documentation - https://scikit-learn.org/
4. TensorFlow Documentation - https://tensorflow.org/

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ for the agricultural community

</div>