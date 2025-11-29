from typing import Union, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load

app = FastAPI()

# Allow the React dev server (http://localhost:3000) to talk to this API while developing.
# Adjust or add origins as needed for production.
origins = [
    "http://localhost",
    "https://insurance-prediction-rho.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = load("MultiRegresion.pkl")
except Exception as exc:  
    model = None
    _load_error = exc


@app.get("/")
def read_root() -> dict:
    return {"hello": "world"}

@app.get("/items/{item_id}")
def read_item(
    item_id: int,
    age: int,
    height: float,
    weight: float,
    q: Union[str, None] = None
) -> Any:
    try:
        raw_pred = model.predict([[age, height, weight]])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to predict: {exc}")

    return {"item_id": item_id, "prediction": raw_pred.tolist()}
