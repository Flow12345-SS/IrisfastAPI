import os
import joblib
import numpy as np
import threading
import webbrowser
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

# Open browser automatically
def open_browser():
    webbrowser.open("http://127.0.0.1:8000")

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=open_browser).start()
    yield

app = FastAPI(lifespan=lifespan)

# Correct absolute path for templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # app/
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load model and scaler
MODEL_DIR = os.path.join(os.path.dirname(BASE_DIR), "model")
model = joblib.load(os.path.join(MODEL_DIR, "iris_best_model.joblib"))
scaler = joblib.load(os.path.join(MODEL_DIR, "iris_scaler.joblib"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction_index = model.predict(input_scaled)[0]
    iris_classes = {0: "setosa", 1: "versicolor", 2: "virginica"}
    prediction = iris_classes.get(prediction_index, str(prediction_index))
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})
