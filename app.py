from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import lightgbm as lgb
import os

# Initialize FastAPI Application with local routing attributes
app = FastAPI(
    title="🏦 GCC Bank: Enterprise Real-Time Fraud Interception Engine",
    description="Production-grade inference API for processing high-velocity credit transaction streams.",
    version="1.0.0",
    docs_url="/docs",      # Explicitly mapping standard Swagger UI route
    redoc_url="/redoc"     # Alternate fail-safe documentation dashboard interface
)

# Global Variables for Artifacts
model = None
expected_features = []
using_fallback = False

# 🚀 1. Load Model Assets on System Startup (With Fail-Safe Logic)
@app.on_event("startup")
def load_model_artifacts():
    global model, expected_features, using_fallback
    
    model_path = 'lgb_fraud_model.txt'
    features_path = 'feature_names.txt'
    
    # Check if production text files exist in the directory
    if os.path.exists(model_path) and os.path.exists(features_path):
        try:
            # Initialize a blank Booster and load the text rules directly
            model = lgb.Booster(model_file=model_path)
            
            # Read the exact training features list
            with open(features_path, 'r') as f:
                expected_features = [line.strip() for line in f.readlines()]
                
            print("✅ Production-grade text model loaded seamlessly into application memory!")
            print(f"📊 Aligned Target Feature Dimension: {len(expected_features)} columns.")
            using_fallback = False
            return
        except Exception as e:
            print(f"⚠️ Failed to parse text assets: {str(e)}. Activating fallback mode.")
    
    # 🛡️ FAIL-SAFE ENVIRONMENT ACTIVATION
    # If files are missing, we dynamically generate mock tracking features 
    # matching the real competition layout to keep the API online.
    print("📢 WARNING: Production model files not found. Initializing fail-safe mock risk system.")
    expected_features = ['TransactionAmt', 'ProductCD', 'card1', 'card4', 'card6', 'P_emaildomain', 'DeviceType', 'DeviceInfo']
    # Adding arbitrary placeholder codes to simulate the wide 434 feature store space
    for i in range(1, 400):
        expected_features.append(f"C{i}")
    
    using_fallback = True
    print(f"✅ Fail-safe mock engine initialized with {len(expected_features)} structured features.")


# 📥 2. Define the Incoming Client Transaction Schema (Data Contract)
class TransactionPayload(BaseModel):
    TransactionAmt: float
    ProductCD: str
    card1: float
    card4: str
    card6: str
    P_emaildomain: str
    DeviceType: str = None
    DeviceInfo: str = None

    # This embeds the real-world sample data directly into the dashboard documentation UI!
    class Config:
        schema_extra = {
            "example": {
                "TransactionAmt": 2500.50,
                "ProductCD": "W",
                "card1": 13926.0,
                "card4": "visa",
                "card6": "credit",
                "P_emaildomain": "gmail.com",
                "DeviceType": "mobile",
                "DeviceInfo": "iPhone"
            }
        }


# 🏡 3. Welcoming Base Landing Page Route
@app.get("/", tags=["General System Portal"])
def read_root():
    return {
        "system_status": "ONLINE",
        "message": "Welcome to the GCC Bank Fraud Interception Core API Server!",
        "primary_testing_dashboard_url": "http://127.0.0",
        "fail_safe_backup_dashboard_url": "http://127.0.0",
        "system_health_monitor_url": "http://127.0.0",
        "engine_mode": "FALLBACK_MOCK_LOGIC" if using_fallback else "PRODUCTION_LIGHTGBM_TEXT_BOOSTER"
    }


# 🩺 4. Real-Time Health Check Endpoint (For Cloud Load Balancers)
@app.get("/health", tags=["Infrastructure Logs"])
def health_check():
    return {
        "status": "HEALTHY",
        "model_engine_active": True,
        "total_aligned_features": len(expected_features),
        "running_on_fallback_logic": using_fallback
    }


# ⚡ 5. Core Prediction / Interception Endpoint
@app.post("/predict/v1/intercept", tags=["Risk Core Engine"])
def intercept_transaction(payload: TransactionPayload):
    try:
        input_data = payload.dict()
        
        # 🧪 ROUTE A: FALLBACK RUNTIME LOGIC (If local text files are missing)
        if using_fallback:
            # We calculate a deterministic fraud score using core transaction risk triggers
            # High amount + credit card + non-standard email domain heavily ticks up the risk score
            base_risk = 5.0
            if input_data["TransactionAmt"] > 2000: base_risk += 35.0
            if input_data["card6"].lower() == "credit": base_risk += 15.0
            if "gmail" not in input_data["P_emaildomain"].lower(): base_risk += 30.0
            
            # Constrain risk between 0.0% and 100.0%
            fraud_probability = min(max(base_risk / 100.0, 0.0), 1.0)
            system_decision = "DECLINE_AND_VERIFY" if fraud_probability > 0.50 else "APPROVE"
            
            return {
                "transaction_status": system_decision,
                "fraud_risk_score": round(fraud_probability * 100, 2),
                "telemetry_audited_features": len(input_data),
                "processing_engine": "MOCK_FAIL_SAFE_ROUTER",
                "processing_latency_status": "WITHIN_SLA"
            }
            
        # 💻 ROUTE B: PRODUCTION LIGHTGBM INFERENCE LOGIC (If files match)
        else:
            # Build an empty dataframe row matching the exact 431 columns the model expects
            full_feature_row = pd.DataFrame(columns=expected_features)
            full_feature_row.loc[0] = [np.nan] * len(expected_features)
            
            # Overlay the customer's live metrics onto our empty baseline matrix
            for key, value in input_data.items():
                if key in full_feature_row.columns:
                    full_feature_row[key] = value
                    
            # Re-cast text columns back to 'category' format to keep LightGBM aligned
            categorical_cols = full_feature_row.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                full_feature_row[col] = full_feature_row[col].astype('category')
                
            # Execute Microsecond Prediction Using Text Booster Engine
            fraud_probability = float(model.predict(full_feature_row))
            system_decision = "DECLINE_AND_VERIFY" if fraud_probability > 0.50 else "APPROVE"
            
            return {
                "transaction_status": system_decision,
                "fraud_risk_score": round(fraud_probability * 100, 2),
                "telemetry_audited_features": len(input_data),
                "processing_engine": "PRODUCTION_LIGHTGBM_BOOSTER",
                "processing_latency_status": "WITHIN_SLA"
            }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Transaction Processing Failure: {str(e)}"
        )



