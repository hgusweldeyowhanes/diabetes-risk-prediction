# 🩺 Diabetes Risk Prediction System

A comprehensive machine learning web application that predicts diabetes risk using multiple AI algorithms with a user-friendly Django interface.

![Django](https://img.shields.io/badge/Django-4.2-green)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

![image](https://raw.githubusercontent.com/hgusweldeyowhanes/diabetes-risk-prediction/main/images/home-page.PNG)
![image](https://raw.githubusercontent.com/hgusweldeyowhanes/diabetes-risk-prediction/main/images/diabets_butten_input.PNG)
## 📖Table of Content
  * [Introduction](#🎯-introduction)
  * [Features](#🌟-features)
  * [Installation](#installation)
  * [Directory Tree](#📁-directory-tree)
  * [Usage](#💻-usage)
  * [Machine Learning Models](#🧠-machine-learning-models)
  * [API Documentation](#📊-api-documentation)
  * [Technology Stack](#🛠️-technology-stack)
  * [Dataset](#📈-dataset)
  * [Results](#📊-results)
  * [Conclusion](#🎯-conclusion)
  * [License](#📄-license)
  * [Disclaimer](#⚠️-disclaimer)

  ## 🎯 Introduction
 
The Diabetes Risk Prediction System is an intelligent web application that leverages machine learning to assess diabetes risk based on patient health parameters. Built with Django and Scikit-learn, it provides real-time predictions with comprehensive analytics and visualizations.
**Key Highlights:**
- Dual AI model approach for enhanced accuracy
- Real-time predictions with confidence scores
- Comprehensive visual analytics
- RESTful API for integration
- User-friendly web interface
## 🌟 Features

- **🤖 Dual AI Models**: SVM Ensemble + Random Forest algorithms
- **🎯 Real-time Predictions**: Instant diabetes risk assessment
- **📊 Visual Analytics**: ROC curves, confusion matrices, feature importance
- **💻 Web Interface**: Beautiful, responsive Django frontend
- **📈 Model Performance**: Accuracy metrics and confidence scores
- **🔄 Data Augmentation**: SMOTE for handling imbalanced data

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/diabetes-risk-prediction.git
   cd diabetes-risk-prediction
   ```
2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **setup database**
```bash
python manage.py migrate
```
5. **Run the development server**
```bash 
python manage.py runserver
```
6. **Acess Aplication**
```bash
Open http://localhost:8000/
```
## 📁 Directory Tree
```text
diabetes-risk-prediction/
├── manage.py
├── diabetes.csv                 # Dataset
├── requirements.txt 
├── templates/
|   ├── home.html
|   ├── predict_form.html
|   ├── predict_result.html      
├── diabetic-prediction/        # Django project
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
└── Diabetic_Prediction/               # Main application
    ├── views.py               # Web views and API
    ├── forms.py              # Input forms
    ├── urls.py              # App URLs
    └── utils/
        ├── ml_model.py      # Machine learning core
```
  
## 💻 Usage
### Web Interface
#### 1.Navigate to Home Page

 - Visit http://localhost:8000
 - Explore the features and overview

#### 2.Access Prediction Form

   - Click "Get Started" or visit /predict/

   - Fill in patient health parameters

#### 4.Submit and View Results

   - Get instant predictions from both models

   - View confidence scores and accuracy metrics

   - Explore visual analytics and charts


## 🧠 Machine Learning Models
### Algorithms Implemented
|Algorithm	| Type   	|Purpose	|Accuracy|
|-------------|---------|------------|--------|
|SVM Ensemble |	Ensemble Learning |	Primary prediction |	~75%|
|Random Forest | Ensemble Learning | Secondary prediction|	~82%|
|SMOTE	   |Data Augmentation	|Handle class imbalance	|-|

### Model Architecture
SVM Ensemble: Combines Linear, RBF, Polynomial, and Sigmoid kernels

Random Forest: Optimized with GridSearchCV for hyperparameter tuning

Feature Scaling: StandardScaler for normalization

Data Balancing: SMOTE for handling imbalanced datasets

## 📊 API Documentation
### Endpoints
| Method | Endpoint	| Description | Parameters|
|---------|---------|-------------|-----------|
|GET	 |/  | 	Home page | 	-  |
| GET	|/predict/	|Prediction form|	-|
|POST	|/predict/	|Submit prediction	|Form data|
|POST	|/api/predict/	|JSON API	|JSON data|
|GET	|/health/	|Health check|	-|
### Request Format
```json
{
  "Pregnancies": 2,
  "Glucose": 138,
  "BloodPressure": 62,
  "SkinThickness": 35,
  "Insulin": 0,
  "BMI": 33.6,
  "DiabetesPedigreeFunction": 0.127,
  "Age": 47
}
```
### Response Format
```json
{
  "svm_prediction": 1,
  "svm_probability": 0.856,
  "rf_prediction": 1,
  "rf_probability": 0.823,
  "analytics": {
    "svm_accuracy_test": 0.851,
    "rf_accuracy_test": 0.823,
    "plots": {  },
    "feature_importance": {  }
  }
}
```
## 🛠️ Technology Stack
### Backend Development
    Framework: Django 4.2

    Language: Python 3.9

    Database: SQLite
    Server: Django Development Server
### Machine Learning
    Library: Scikit-learn 1.3.0

    Data Processing: Pandas, NumPy

    Visualization: Matplotlib, Seaborn

    Data Augmentation: Imbalanced-learn
### Frontend Development
    UI Framework: Bootstrap 5

    Styling: CSS3, Custom CSS

    Interactivity: JavaScript

    Charts: Matplotlib (server-side)
## 📈 Dataset
### Overview
    Source:  Diabetes Database (Kaggle)

    Records: 2000 patient entries

    Features: 8 medical parameters

    Target: Binary classification (Diabetic/Non-Diabetic)
### Feature Description
| Feature	| Description	| Range  |
|--------------|----------------|-------------|
|Pregnancies	|Number of pregnancies	|0-17|
|Glucose	|Plasma glucose concentration	|0-199|
|BloodPressure	|Diastolic blood pressure (mm Hg)	|0-122|
|SkinThickness	|Triceps skin fold thickness (mm)	|0-99|
|Insulin	|2-Hour serum insulin (mu U/ml)	|0-846|
|BMI	|Body mass index	|0-67.1|
|DiabetesPedigreeFunction	|Diabetes pedigree function	|0.08-2.42|
|Age                 |    Age in years	     |  21-81       |

## 📊 Results
![image](https://raw.githubusercontent.com/hgusweldeyowhanes/diabetes-risk-prediction/main/images/model_pred.PNG)
![image](https://raw.githubusercontent.com/hgusweldeyowhanes/diabetes-risk-prediction/main/images/model_analytics.PNG)
![image](https://raw.githubusercontent.com/hgusweldeyowhanes/diabetes-risk-prediction/main/images/feature-importance.PNG)

### Model Performance
|Metric	| SVM Ensemble	| Random Forest|
|---------|--------------|-------------|
|Accuracy	|75%	|82%|
|Precision	|76%	|84%|
|Recall	|74%	|81%|
|F1-Score	|75%	|82%|

## 🎯 Conclusion
### Key Achievements
✅ Successful Implementation: Built a fully functional diabetes prediction system

✅ Dual Model Approach: Enhanced reliability through ensemble methods

✅ User-Friendly Interface: Intuitive web application for easy usage

✅ Comprehensive Analytics: Detailed insights into model performance

✅ REST API: Programmatic access for integration
### Future Enhancements
    Deploy to cloud platform (Heroku/AWS)

    Add user authentication system

    Implement patient history tracking

    Add more machine learning models

    Create mobile application

    Enhance visualization with interactive charts
## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
## ⚠️ Disclaimer
This project is developed for educational and demonstration purposes only.
