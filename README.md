# 🌾 Crop Yield Prediction Using Machine Learning

This project explores multiple machine learning models to accurately predict crop yields using global agricultural and environmental datasets. The aim is to assist farmers and policymakers in making informed decisions based on data-driven insights.

---

## 📌 Problem Statement

Predicting crop production is complex due to high-dimensional, nonlinear relationships among variables like temperature, rainfall, and pesticide use. Traditional statistical models fail to handle such complexity. This project investigates whether machine learning models can effectively predict crop yields using these features.

---

## 🎯 Objectives

- Build and compare four machine learning models:
  - **Ridge Regression**
  - **Decision Tree Regressor**
  - **Feedforward Neural Network (FFNN)**
  - **PCA + Random Forest**
- Identify the best-performing model for accurate crop yield prediction.
- Provide practical insights for better agricultural planning.

---

## 📚 Dataset

**Data Sources:**
- 🌐 World Bank
- 🌾 Food and Agriculture Organization (FAO)

**Countries Covered:** 101  
**Time Span:** 1990 to 2013

**Features Used:**
- `Crop Yield (hg/ha)`
- `Rainfall (mm)`
- `Pesticides (tonnes)`
- `Temperature (°C)`

---

## 🛠️ Tools & Libraries

| Task                      | Tools / Libraries Used                       |
|---------------------------|---------------------------------------------|
| Data Handling             | `Pandas`, `NumPy`                           |
| Visualization             | `Matplotlib`, `Seaborn`                     |
| Preprocessing             | `StandardScaler`, `OneHotEncoder`          |
| Dimensionality Reduction  | `PCA` from `sklearn.decomposition`         |
| Model Training            | `sklearn`, `TensorFlow`, `Keras`           |
| Evaluation Metrics        | `MSE`, `MAE`, `R²`, `Accuracy`             |
| Hyperparameter Tuning     | `GridSearchCV`                             |

---

## 🧪 Models Implemented

### 1. Ridge Regression
Linear model with L2 regularization to reduce overfitting.

### 2. Decision Tree Regressor
Captures non-linear relationships using tree splits based on features like rainfall and pesticides.

### 3. Feedforward Neural Network (FFNN)
Deep learning model using:
- Activation: ReLU  
- Optimizer: Adam  
- Split: 60% Train / 20% Validation / 20% Test

### 4. PCA + Random Forest
Uses PCA for dimensionality reduction, followed by Random Forest for robust predictions.

---

## 📊 Model Performance

| Model                   | MSE    | MAE    | R²     | Accuracy (%) |
|------------------------|--------|--------|--------|--------------|
| Ridge Regression       | 523.25 | 18.36  | 0.84   | 92.15%       |
| PCA + Random Forest    | 452.14 | 15.92  | 0.89   | **93.87%**   |
| Decision Tree Regressor| 562.31 | 20.12  | 0.81   | 90.73%       |
| Feedforward Neural Net | 482.67 | 16.43  | 0.87   | 93.21%       |

✅ **Best Model**: PCA + Random Forest

---

## 📈 Visualizations

- 🔥 Correlation Heatmap
- 📉 Actual vs Predicted Density Plots
- 🧠 Feature Importance Rankings
- 📊 Evaluation Metric Bar Charts

---

## 💡 Key Insights

- **Rainfall**, **temperature**, and **pesticide use** are key predictive features.
- **Potatoes** and **India** are highly influential in yield prediction.
- PCA+Random Forest shows best accuracy due to robust handling of high-dimensional data.
- FFNN is effective for non-linear relationships, proving deep learning’s potential in agriculture.

---

## ⚠️ Limitations

- Lack of **soil** and **irrigation** data may limit prediction accuracy.
- **Ridge Regression** struggles with non-linearity.
- Advanced regularization or ensemble techniques offer better performance for complex datasets.

---

## 🔧 Project Structure

├── modelbuilding.ipynb # Main Jupyter Notebook with all steps
├── yield_df.csv # Final merged dataset (1990–2013)
├── README.md # Project documentation
└── figures/ # Generated plots and visualizations


---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/crop-yield-prediction.git
   cd crop-yield-prediction
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

3. ** Run the Notebook**
   ```bash
   jupyter notebook modelbuilding.ipynb

##📩 Contact
Zeel Patel
📧 zeelpatel1539@gmail.com
