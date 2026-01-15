# Predicting Diabetes-Related Complications Using Machine Learning

## 📋 Project Overview
This project implements multiple machine learning algorithms to predict diabetes-related complications using health data from the Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset. The ensemble Voting Classifier achieved the highest accuracy of **88.5%** in predicting individuals at risk of diabetes complications.

## 🎯 Key Features
- **Multiple ML Algorithms**: Logistic Regression, Random Forest, SVM, Naïve Bayes, Decision Trees  
- **Ensemble Learning**: Voting Classifier for improved accuracy  
- **Advanced Preprocessing**: SMOTE for class balancing, feature engineering  
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC analysis  

## 📊 Dataset
- **Source**: BRFSS 2015 (Behavioral Risk Factor Surveillance System)  
- **Samples**: 253,680 survey responses  
- **Features**: 21 health indicators including BMI, blood pressure, cholesterol, lifestyle factors  
- **Target**: Binary classification of diabetes-related complications  

## 🔧 Data Preprocessing
1. **Missing Values**: Median imputation  
2. **Feature Engineering**:  
   - Hypertension Risk = HighBP + HighChol  
   - Physical Health Score = PhysHlth + DiffWalk  
3. **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique)  
4. **Feature Selection**: Pearson correlation analysis  

## 🏗️ Machine Learning Models
### Algorithms Implemented:
1. Logistic Regression  
2. Random Forest  
3. Support Vector Machine (SVM)  
4. Naïve Bayes  
5. Decision Tree  
6. Voting Classifier (Ensemble)  

### Model Performance Comparison
| Model              | Accuracy | Precision | Recall | F1-Score | AUC Score |
|--------------------|----------|-----------|--------|----------|-----------|
| Random Forest      | 88.45%   | 0.88      | 0.88   | 0.88     | 0.96      |
| Decision Tree      | 83.96%   | 0.84      | 0.84   | 0.84     | 0.84      |
| Logistic Regression| 76.12%   | 0.76      | 0.76   | 0.76     | 0.76      |
| Naïve Bayes        | 73.01%   | 0.73      | 0.73   | 0.73     | 0.73      |
| **Voting Classifier** | **88.50%** | **0.88** | **0.88** | **0.88** | **0.97** |

## 📁 Project Structure
diabetes-complications-prediction/ │ ├── data/ │   ├── raw/BRFSS2015.csv │   ├── processed/cleaned_data.csv │   └── splits/train_test_split/ │ ├── notebooks/ │   ├── 01_EDA_Data_Analysis.ipynb │   ├── 02_Feature_Engineering.ipynb │   ├── 03_Model_Training.ipynb │   └── 04_Model_Evaluation.ipynb │ ├── src/ │   ├── data_preprocessing.py │   ├── feature_engineering.py │   ├── model_training.py │   ├── model_evaluation.py │   └── utils.py │ ├── models/ │   ├── random_forest.pkl │   ├── voting_classifier.pkl │   └── all_models/ │ ├── results/ │   ├── confusion_matrices/ │   ├── roc_curves/ │   ├── feature_importance/ │   └── metrics_report.json │ ├── requirements.txt ├── README.md └── LICENSE

🎯 Applications
- Early detection of diabetes-related complications
- Support for healthcare decision-making
- Risk stratification in public health research
🔮 Future Work
- [ ] Expand dataset with more recent BRFSS surveys
- [ ] Integrate deep learning models for comparison
- [ ] Develop web-based prediction tool
- [ ] Real-time health monitoring integration
