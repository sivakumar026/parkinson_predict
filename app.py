import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins (OK for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load model & scaler (ONCE)
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# -----------------------------
# Pydantic Input Schema
# -----------------------------
class ParkinsonsInput(BaseModel):
    MDVP_Fo_Hz: float
    MDVP_Fhi_Hz: float
    MDVP_Flo_Hz: float
    MDVP_Jitter_percent: float
    MDVP_Jitter_Abs: float
    MDVP_RAP: float
    MDVP_PPQ: float
    Jitter_DDP: float
    MDVP_Shimmer: float
    MDVP_Shimmer_dB: float
    Shimmer_APQ3: float
    Shimmer_APQ5: float
    MDVP_APQ: float
    Shimmer_DDA: float
    NHR: float
    HNR: float
    RPDE: float
    DFA: float
    spread1: float
    spread2: float
    D2: float
    PPE: float


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: ParkinsonsInput):

    # 1️⃣ Convert input → numpy array (correct order)
    features = np.array([[
        data.MDVP_Fo_Hz,
        data.MDVP_Fhi_Hz,
        data.MDVP_Flo_Hz,
        data.MDVP_Jitter_percent,
        data.MDVP_Jitter_Abs,
        data.MDVP_RAP,
        data.MDVP_PPQ,
        data.Jitter_DDP,
        data.MDVP_Shimmer,
        data.MDVP_Shimmer_dB,
        data.Shimmer_APQ3,
        data.Shimmer_APQ5,
        data.MDVP_APQ,
        data.Shimmer_DDA,
        data.NHR,
        data.HNR,
        data.RPDE,
        data.DFA,
        data.spread1,
        data.spread2,
        data.D2,
        data.PPE
    ]])

    # 2️⃣ Apply scaler
    scaled_features = scaler.transform(features)

    # 3️⃣ Predict
    prediction = model.predict(scaled_features)
    prediction=prediction[0]
    return {
        "prediction": int(prediction),
        "result": "Parkinson detected" if prediction == 1 else "Healthy"
    }
