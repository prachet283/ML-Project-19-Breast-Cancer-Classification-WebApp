# Breast Cancer Classification WebApp
This project aims to create a web application that can classify breast cancer as malignant or benign using machine learning techniques. The primary objective is to provide an accessible tool for medical professionals and researchers to predict the nature of breast tumors based on various diagnostic features.

# Overview
Breast cancer is one of the most common cancers among women worldwide. Early detection and accurate diagnosis are crucial for effective treatment and improved survival rates. This project leverages machine learning to build a predictive model and deploys it as a web application to assist in the classification of breast cancer tumors.

# Dataset
The dataset used for this project is the Breast Cancer Wisconsin (Diagnostic) Data Set, available from the UCI Machine Learning Repository. It consists of 569 samples, each with 30 features describing various properties of cell nuclei present in the tumor. The target variable indicates whether the tumor is malignant or benign.

Features include:

Radius
Texture
Perimeter
Area
Smoothness
Compactness
Concavity
Concave points
Symmetry
Fractal dimension

# Technology
The project utilizes the following technologies and libraries:

Python: Programming language
Pandas: Data manipulation and analysis
NumPy: Numerical computing
Matplotlib & Seaborn: Data visualization
Scikit-learn: Machine learning
Streamlit: Web application framework

# Exploratory Data Analysis (EDA)
EDA was performed to understand the distribution and relationships of the features in the dataset. Key steps in the EDA process include:

Checking for missing values and data types.
Statistical summary of the features.
Visualizing the distribution of features using histograms.
Correlation analysis using heatmaps to identify relationships between features.

# Model
The classification model was built using the Logistic Regression algorithm. The steps involved in model development include:

Data preprocessing: Standard scaling of features.
Splitting the dataset into training and testing sets.
Training the Logistic Regression model on the training data.
Evaluating the model on the test data using metrics such as accuracy, precision, recall, and F1-score.

# Web App
The web application was developed using Streamlit, providing an interactive interface for users to input diagnostic features and obtain predictions.

Features of the web app:

User-friendly input forms for entering diagnostic features.
"Predict" button to generate classification results.
Display of prediction results and model performance metrics.

# Results
The model achieves the following performance metrics on the test set:

Accuracy: 92.98%
These metrics indicate the model's effectiveness in classifying tumors as malignant or benign.

# Conclusion
The Breast Cancer Classification Web App is a powerful tool that leverages machine learning to assist in the early detection and diagnosis of breast cancer. By providing an accessible and user-friendly interface, it aims to support medical professionals and researchers in making informed decisions. Future improvements may include incorporating more advanced models and expanding the feature set to enhance prediction accuracy.

# Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

# Deployment
The application is deployed using Streamlit. You can access it here = https://ml-project-19-breast-cancer-classification-webapp-luox8ykrhlio.streamlit.app/

You can access updated it here = https://ml-project-19-breast-cancer-classification-webapp-9kzm5bbgrew6.streamlit.app/
