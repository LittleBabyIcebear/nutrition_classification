import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import io

app = FastAPI()

# Muat Objek yang Telah Disimpan
try:
    model = joblib.load('knn_model.joblib')
    scaler = joblib.load('scaler.joblib')
    encoders = joblib.load('encoders.joblib')
    inverse_mapping = joblib.load('inverse_status_gizi_mapping.joblib')
except FileNotFoundError:
    raise RuntimeError("Joblib file not found")


# Model Input Data
class BalitaInput(BaseModel):
    umur_bulan: int
    jenis_kelamin: str
    tinggi_badan_cm: float
    berat_badan_kg: float


# Endpoint Single
@app.post("/predict")
def predict_status_gizi(data: BalitaInput):
    input_df = pd.DataFrame([data.dict()])

    input_df = input_df.rename(columns={
        'umur_bulan': 'Umur (bulan)',
        'jenis_kelamin': 'Jenis Kelamin',
        'tinggi_badan_cm': 'Tinggi Badan (cm)',
        'berat_badan_kg': 'Prediksi Berat Badan (kg)'
    })

    jenis_kelamin_encoder = encoders['Jenis Kelamin']
    input_df['Jenis Kelamin'] = jenis_kelamin_encoder.transform(input_df['Jenis Kelamin'])

    numerical_cols = ['Umur (bulan)', 'Tinggi Badan (cm)', 'Prediksi Berat Badan (kg)']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    input_df = input_df[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)', 'Prediksi Berat Badan (kg)']]

    prediction_numeric = model.predict(input_df)[0]
    prediction_label = inverse_mapping[prediction_numeric]

    return {"prediksi_status_gizi": prediction_label}


# -Endpoint Massal
@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Endpoint untuk memprediksi status gizi balita secara massal dari file CSV.
    File CSV harus memiliki kolom dalam urutan:
    1. Umur (bulan)
    2. Jenis Kelamin
    3. Tinggi Badan (cm)
    4. Prediksi Berat Badan (kg)
    """
    try:
        # Baca file CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Validasi kolom
        expected_columns = ['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)', 'Prediksi Berat Badan (kg)']
        if list(df.columns) != expected_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Kolom CSV harus: {expected_columns}"
            )
        
        # Validasi nilai Jenis Kelamin
        valid_genders = ['laki-laki', 'perempuan']
        if not df['Jenis Kelamin'].isin(valid_genders).all():
            raise HTTPException(
                status_code=400,
                detail=f"Jenis Kelamin harus: {valid_genders}"
            )
        
        # Preprocessing
        df_processed = df.copy()
        
        jenis_kelamin_encoder = encoders['Jenis Kelamin']
        df_processed['Jenis Kelamin'] = jenis_kelamin_encoder.transform(df_processed['Jenis Kelamin'])
        
        numerical_cols = ['Umur (bulan)', 'Tinggi Badan (cm)', 'Prediksi Berat Badan (kg)']
        df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
        
        df_processed = df_processed[['Umur (bulan)', 'Jenis Kelamin', 'Tinggi Badan (cm)', 'Prediksi Berat Badan (kg)']]
        
        # Prediksi
        predictions_numeric = model.predict(df_processed)
        predictions_label = [inverse_mapping[pred] for pred in predictions_numeric]
        
        # Format output
        results = []
        for idx, pred in enumerate(predictions_label):
            results.append({
                "index": idx,
                "prediksi_status_gizi": pred
            })
        
        return {
            "total": len(results),
            "prediksi": results
        }
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="File CSV kosong")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Endpoint root
@app.get("/")
def read_root():
    return {"message": "API Prediksi Status Gizi Balita"}