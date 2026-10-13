from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
import joblib
import numpy as np
from typing import Annotated
import sys
import os
from fastapi.responses import HTMLResponse

app = FastAPI()

# Compatibility workaround for some saved scikit-learn models
from sklearn._loss import _loss as sklearn_loss
sys.modules.setdefault("_loss", sklearn_loss)

model_data = joblib.load("heart_model.joblib")
model = model_data["model"]
feature_names = model_data["features"]
templates = Jinja2Templates(directory="templates")

class HeartData(BaseModel):
    age: Annotated[int, "Age of the patient"] = Field(ge=0, le=120)
    sex: Annotated[int, "Sex of the patient (1 = male, 0 = female)"]
    cp: Annotated[int, "Chest pain type"]
    trestbps: Annotated[float, "Resting blood pressure"]
    chol: Annotated[float, "Serum cholesterol"]
    fbs: Annotated[int, "Fasting blood sugar"]
    restecg: Annotated[int, "Resting electrocardiographic results"] 
    thalach: Annotated[float, "Maximum heart rate achieved"]
    exang: Annotated[int, "Exercise-induced angina"]
    oldpeak: Annotated[float, "ST depression induced by exercise"]
    slope: Annotated[int, "Slope of the peak exercise ST segment"]
    ca: Annotated[int, "Number of major vessels colored by fluoroscopy"]
    thal: Annotated[int, "Thalassemia"]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()

    float_fields = {"trestbps", "chol", "thalach", "oldpeak"}
    raw_data = {}

    try:
        # Convert form values into numbers
        for key, value in form_data.items():
            if value == "" or value is None:
                continue

            if key in float_fields:
                raw_data[key] = float(value)
            else:
                raw_data[key] = int(value)

        # Validate the input fields
        validated_data = HeartData(**raw_data)
        input_dict = validated_data.model_dump()

        # Arrange columns in the exact order expected by the model
        input_df = pd.DataFrame([input_dict], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_df)[0]

        result = (
            "Positive for heart disease"
            if prediction == 1
            else "Negative for heart disease"
        )

        return templates.TemplateResponse(
            request,
            "result.html",
            {"result": result}
        )

    except (ValidationError, ValueError, TypeError) as e:
        return templates.TemplateResponse(
            request,
            "result.html",
            {
                "result": f"Please fill out all fields correctly. Error: {e}"
            },
            status_code=400
        )
