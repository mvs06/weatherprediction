# ==========================================
# To run this API locally, install the required packages:
# pip install fastapi uvicorn xgboost pandas pydantic numpy twilio
#
# To start the local server, run:
# uvicorn main:app --reload --port 8080
# ==========================================

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
EXPECTED_COLS = [
    "temperature_2m_lag1",
    "relative_humidity_2m_lag1",
    "surface_pressure_lag1",
    "cloud_cover_lag1",
    "dew_point_lag1",
    "soil_moisture_lag1",
    "precipitation_lag1",
    "temperature_2m_lag24",
    "relative_humidity_2m_lag24",
    "surface_pressure_lag24",
    "cloud_cover_lag24",
    "dew_point_lag24",
    "soil_moisture_lag24",
    "precipitation_lag24",
]

app = FastAPI(
    title="Weather Prediction API",
    description="XGBoost weather predictor with optional SMS delivery.",
)


def env_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise HTTPException(
            status_code=500,
            detail=f"Missing required environment variable: {name}",
        )
    return value


def load_model(model: Any, filename: str) -> Any:
    model_path = BASE_DIR / filename
    try:
        model.load_model(model_path.as_posix())
    except Exception:
        print(
            f"Warning: Model not found at {model_path}. "
            "Prediction endpoints will fail until the file is available."
        )
    return model


model = load_model(xgb.XGBRegressor(), "weather_xgboost_model.json")
temp_model = load_model(xgb.XGBRegressor(), "temp_xgboost_model.json")
model_3d = load_model(xgb.XGBRegressor(), "weather_xgboost_3d_model.json")
temp_model_3d = load_model(xgb.XGBRegressor(), "temp_xgboost_3d_model.json")
prob_model = load_model(xgb.XGBClassifier(), "rain_prob_xgboost_model.json")
prob_model_3d = load_model(xgb.XGBClassifier(), "rain_prob_xgboost_3d_model.json")


class WeatherInput(BaseModel):
    temperature_2m_lag1: float
    relative_humidity_2m_lag1: float
    surface_pressure_lag1: float
    cloud_cover_lag1: float
    dew_point_lag1: float
    soil_moisture_lag1: float
    precipitation_lag1: float
    temperature_2m_lag24: float
    relative_humidity_2m_lag24: float
    surface_pressure_lag24: float
    cloud_cover_lag24: float
    dew_point_lag24: float
    soil_moisture_lag24: float
    precipitation_lag24: float


class PredictionSMSRequest(WeatherInput):
    phone_number: str = Field(..., min_length=8)
    location_name: Optional[str] = "Requested location"


class HistoricalWeatherInput(BaseModel):
    hourly_times: List[str]
    temperature_2m: List[float]
    relative_humidity_2m: List[float]
    surface_pressure: List[float]
    cloud_cover: List[float]
    dew_point: List[float]
    soil_moisture: List[float]
    precipitation: List[float]


def normalize_precipitation(prediction: float) -> float:
    cleaned_value = float(prediction)
    if cleaned_value < 0.1:
        return 0.0
    return round(cleaned_value, 2)


def build_input_frame(data: WeatherInput) -> pd.DataFrame:
    input_data = pd.DataFrame([data.model_dump()])
    return input_data[EXPECTED_COLS]


def calculate_prediction(data: WeatherInput) -> Dict[str, Any]:
    input_data = build_input_frame(data)

    try:
        prediction = model.predict(input_data)[0]
        temp_prediction = temp_model.predict(input_data)[0]
        rain_prob = prob_model.predict_proba(input_data)[0][1] * 100

        prediction_3d = model_3d.predict(input_data)[0]
        temp_prediction_3d = temp_model_3d.predict(input_data)[0]
        rain_prob_3d = prob_model_3d.predict_proba(input_data)[0][1] * 100
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    pred_value = normalize_precipitation(prediction)
    pred_value_3d = normalize_precipitation(prediction_3d)

    return {
        "predicted_precipitation_mm": pred_value,
        "chance_of_rain_percent": round(float(rain_prob), 1),
        "will_it_rain": pred_value > 0,
        "predicted_temperature_2m": round(float(temp_prediction), 2),
        "predicted_precipitation_mm_3d": pred_value_3d,
        "chance_of_rain_percent_3d": round(float(rain_prob_3d), 1),
        "will_it_rain_3d": pred_value_3d > 0,
        "predicted_temperature_2m_3d": round(float(temp_prediction_3d), 2),
    }


def format_prediction_sms(prediction: Dict[str, Any], location_name: str) -> str:
    return "\n".join(
        [
            f"Weather prediction for {location_name}",
            "",
            "Next 24 hours:",
            f"Temperature: {prediction['predicted_temperature_2m']} C",
            f"Rain chance: {prediction['chance_of_rain_percent']}%",
            f"Expected rain: {prediction['predicted_precipitation_mm']} mm",
            f"Rain expected: {'Yes' if prediction['will_it_rain'] else 'No'}",
            "",
            "3 days ahead:",
            f"Temperature: {prediction['predicted_temperature_2m_3d']} C",
            f"Rain chance: {prediction['chance_of_rain_percent_3d']}%",
            f"Expected rain: {prediction['predicted_precipitation_mm_3d']} mm",
            f"Rain expected: {'Yes' if prediction['will_it_rain_3d'] else 'No'}",
        ]
    )


def send_sms(phone_number: str, message: str) -> str:
    try:
        from twilio.base.exceptions import TwilioRestException
        from twilio.rest import Client
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Twilio is not installed. Run: pip install twilio",
        ) from exc

    account_sid = env_required("TWILIO_ACCOUNT_SID")
    auth_token = env_required("TWILIO_AUTH_TOKEN")
    from_number = env_required("TWILIO_FROM_NUMBER")

    client = Client(account_sid, auth_token)
    try:
        sms = client.messages.create(body=message, from_=from_number, to=phone_number)
    except TwilioRestException as exc:
        raise HTTPException(status_code=502, detail=f"Twilio error: {exc.msg}") from exc

    return sms.sid


@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "message": "Weather ML API is running!"}


@app.post("/predict")
def predict_rain(data: WeatherInput) -> Dict[str, Any]:
    return calculate_prediction(data)


@app.post("/predict-and-send")
def predict_and_send(payload: PredictionSMSRequest) -> Dict[str, Any]:
    prediction = calculate_prediction(payload)
    sms_message = format_prediction_sms(
        prediction=prediction,
        location_name=payload.location_name or "Requested location",
    )
    message_sid = send_sms(payload.phone_number, sms_message)

    return {
        "status": "sent",
        "phone_number": payload.phone_number,
        "message_sid": message_sid,
        "sms_message": sms_message,
        "prediction": prediction,
    }


@app.post("/forecast_3d")
def forecast_3d(data: HistoricalWeatherInput) -> Dict[str, List[Dict[str, Any]]]:
    df = pd.DataFrame(
        {
            "time": data.hourly_times,
            "temperature_2m": data.temperature_2m,
            "relative_humidity_2m": data.relative_humidity_2m,
            "surface_pressure": data.surface_pressure,
            "cloud_cover": data.cloud_cover,
            "dew_point": data.dew_point,
            "soil_moisture": data.soil_moisture,
            "precipitation": data.precipitation,
        }
    )

    features = [
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "cloud_cover",
        "dew_point",
        "soil_moisture",
        "precipitation",
    ]
    for feature in features:
        df[f"{feature}_lag1"] = df[feature].shift(1)
        df[f"{feature}_lag24"] = df[feature].shift(24)

    df_valid = df.dropna().tail(72).copy()

    if len(df_valid) < 72:
        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough data points provided. Provide at least 4 days "
                "(96 hours) of historical data to get a 3-day output."
            ),
        )

    X = df_valid[EXPECTED_COLS]

    preds_precip = model_3d.predict(X)
    preds_temp = temp_model_3d.predict(X)
    preds_prob = prob_model_3d.predict_proba(X)[:, 1] * 100

    df_valid["time"] = pd.to_datetime(df_valid["time"])
    future_times = df_valid["time"] + pd.Timedelta(days=3)

    forecasts = []
    for index, future_time in enumerate(future_times):
        precip_value = normalize_precipitation(preds_precip[index])
        forecasts.append(
            {
                "time": future_time.isoformat(),
                "temperature_2m": round(float(preds_temp[index]), 2),
                "precipitation_mm": precip_value,
                "chance_of_rain_percent": round(float(preds_prob[index]), 1),
            }
        )

    return {"forecast": forecasts}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
