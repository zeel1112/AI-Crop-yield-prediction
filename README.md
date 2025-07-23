🌾 Crop Yield Prediction Using Machine Learning
This project explores multiple machine learning models to accurately predict crop yields using global agricultural and environmental datasets. The aim is to assist farmers and policymakers in making informed decisions based on data-driven insights.

📌 Problem Statement
Predicting crop production is complex due to high-dimensional, nonlinear relationships among variables like temperature, rainfall, and pesticide use. Traditional statistical models fail to handle such complexity. This project investigates whether machine learning models can effectively predict crop yields using these features.

🎯 Objectives
Build and compare four machine learning models:

Ridge Regression

Decision Tree Regressor

Feedforward Neural Network (FFNN)

PCA + Random Forest

Identify the best-performing model for accurate crop yield prediction.

Provide practical insights for better agricultural planning.

📚 Dataset
Data is sourced from:

World Bank

Food and Agriculture Organization (FAO)

Covers data from 101 countries between 1990 to 2013.

Features Used:

Crop Yield (hg/ha)

Rainfall (mm)

Pesticides (tonnes)

Temperature (°C)

🛠️ Tools & Libraries
Task	Tools / Libraries Used
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Preprocessing	StandardScaler, OneHotEncoder
Dimensionality Reduction	PCA from sklearn.decomposition
Model Training	sklearn, TensorFlow, Keras
Evaluation Metrics	MSE, MAE, R², Accuracy
Hyperparameter Tuning	GridSearchCV

🧪 Models Implemented
Ridge Regression
Linear model with L2 regularization to reduce overfitting.

Decision Tree Regressor
Captures non-linear relationships using tree splits based on features like rainfall and pesticides.

Feedforward Neural Network (FFNN)
Deep learning model capable of learning complex patterns using ReLU activation and Adam optimizer.

PCA + Random Forest
Uses Principal Component Analysis to reduce dimensionality before applying Random Forest for robust predictions.

📊 Model Performance
Model	MSE	MAE	R²	Accuracy (%)
Ridge Regression	523.25	18.36	0.84	92.15%
PCA + Random Forest	452.14	15.92	0.89	93.87%
Decision Tree Regressor	562.31	20.12	0.81	90.73%
Feedforward Neural Net	482.67	16.43	0.87	93.21%

✅ Best Model: PCA + Random Forest

📈 Visualizations
Correlation Heatmap

Actual vs Predicted Density Plots

Feature Importance Rankings

Evaluation Metric Comparisons

💡 Key Insights
PCA + Random Forest outperforms all models due to its ability to handle high-dimensional data effectively.

FFNN is also strong, highlighting the power of deep learning for nonlinear problems.

Rainfall, temperature, and pesticide use are the top influential features.

Potatoes and India show high predictive importance in crop yield modeling.

⚠️ Limitations
Absence of key variables like soil quality and irrigation data may affect prediction accuracy.

Ridge Regression underperformed due to its linear assumptions.

🔧 Project Structure
bash
Copy
Edit
├── modelbuilding.ipynb        # Main Jupyter Notebook with all steps
├── yield_df.csv               # Final merged dataset (1990–2013)
├── README.md                  # Project documentation
└── figures/                   # Generated plots and visualizations
🚀 Getting Started
Clone this repo:

bash
Copy
Edit
git clone https://github.com/yourusername/crop-yield-prediction.git
cd crop-yield-prediction
Install requirements:

bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook modelbuilding.ipynb
📩 Contact
For questions or collaboration:
Zeel Patel
📧 zeel2002patel@gmail.com
🔗 LinkedIn
