import joblib
from fastapi import FastAPI, HTTPException
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd

app = FastAPI() 

model = joblib.load("model/model-car-predict.joblib")

categorical_features = ["Brand", "Model", "Fuel_Type", "Transmission"]
numerical_features = ["Year", "Engine_Size", "Mileage", "Doors", "Owner_Count"]

encoders = {}
for col in categorical_features:
    encoders[col] =  joblib.load(f"encoding/{col.lower()}-encoding.joblib")
    
scalers = {}
for col in numerical_features:
    scalers[col] =  joblib.load(f"scaler/{col.lower()}-scaler.joblib")

@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/")
def predict(data: dict):
        # Preprocessing data
        feature_list = preprocess_data(data)
        
        # Model predictionS
        prediction = model.predict(feature_list)
        
        # result = pd.DataFrame(prediction, columns=["Predicted_Price"])
        
        return { "prediction": list(prediction) }
        # feat = data["features"][0].keys()
        # return { "features": list(feat)}

def preprocess_data(data: dict)->any:
    
    features = data["features"]
    
    feature_list = []
    for f in features:
        feature_list.append([
            f["Brand"],
            f["Model"],
            f["Year"],
            f["Engine_Size"],
            f["Fuel_Type"],
            f["Transmission"],
            f["Mileage"],
            f["Doors"],
            f["Owner_Count"]])
        
    return feature_list


def handle_input(data: dict)->any:

    cols = ["Brand", "Model", "Year", "Engine_Size", "Fuel_Type", "Transmission", "Mileage", "Doors", "Owner_Count"]
    df = pd.DataFrame(data, columns=cols)
        
    for col in categorical_features:
        df[col] = encoders[col].transform(df[col])
        
    for col in numerical_features:
        df[col] = scalers[col].transform(df[col])
        
    return df