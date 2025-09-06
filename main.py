from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd


app = FastAPI(
    title="Diabetes Prediction API",
    description = "An API that predicts diabetes using a trained ML model",
    version = "1.0.0"
)


templates = Jinja2Templates(directory = "templates")

# Loadaing the machine learning model
with open("diabeties.pkl","rb") as f:
    model = pickle.load(f)

@app.get("/",response_class = HTMLResponse)
def form_page(request : Request):
    return templates.TemplateResponse("index.html",{"request":request})




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