from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pickle
from pydantic import BaseModel,Field
import numpy as np
import pandas as pd
from enum import Enum

app = FastAPI(
    title="Diabetes Prediction API",
    description = "An API that predicts diabetes using a trained ML model",
    version = "1.0.0"
)

# app.mount("/static",StaticFiles(directory = "static"),name = "static")

templates = Jinja2Templates(directory = "templates")

# Loadaing the machine learning model
with open("diabeties.pkl","rb") as f:
    model = pickle.load(f)

@app.get("/",response_class = HTMLResponse)
def form_page(request : Request):
    return templates.TemplateResponse("index.html",{"request":request})


# class Gender(str,Enum):
#     Male = "Male"
#     Female = "Female"

# class smokinghistory(str,Enum):
#     never = "never"
#     former = "former"
#     current = "current"
#     No_Info = "No Info"


# class InputData(BaseModel):
#     gender : Gender = Field(..., description = "Gender of the Patient:(Male/Female)")
#     age : int = Field(...,example = 20, description = "Enter the patienct's age")
#     hypertension : int = Field(...,example = 0, description = "0 = No, 1 = Yes")
#     heart_disease  : int = Field(...,example = 0, description = "0 = No, 1 = Yes")
#     smoking_history : smokinghistory = Field(..., description = "Smoking History;(never,formrer,current or No Info)")
#     bmi : float = Field(...,example = 17.5, description = "Enter your Body Mass Index(BMI)")
#     HbA1c_level : float = Field(...,example = 5.8, description = "Average Blood Sugar Level")
#     blood_glucose_level : int = Field(...,example =140, description = "Blood Glucose Level")



@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            gender: str = Form(...),
            age: int = Form(...),
            hypertension: int = Form(...),
            heart_disease: int = Form(...),
            smoking_history: str = Form(...),
            bmi: float = Form(...),
            HbA1c_level: float = Form(...),
            blood_glucose_level: int = Form(...)):
    
    input_df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": HbA1c_level,
        "blood_glucose_level": blood_glucose_level
    }])

    prediction = model.predict(input_df)[0]

    result = "You have Diabetes" if prediction == 1 else "You don't have Diabeties"

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": result
    })