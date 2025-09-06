# 🩺 Diabetic Patient Prediction API

This is a **Diabetic Patient Prediction API application**.  

## 📊 Dataset
In the production process, I used the **Kaggle dataset** (already included in this repo in case you want to work on it).  

## 🤖 Model
I trained a **Support Vector Classifier (SVC)** and was able to achieve **96% accuracy** without any signs of underfitting or overfitting.  

The model is wrapped in a **preprocessing pipeline**, so if you want to use it, you don’t need to perform any preprocessing operations. Just provide the input in the same format as the training dataset, and it will generate predictions.  

## 🚀 Deployment
In the end, I used **FastAPI** to deploy the model.
