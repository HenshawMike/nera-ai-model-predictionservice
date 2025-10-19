"""
FastAPI application for serving real estate price predictions
"""
import os
import logging
from typing import Dict, Any, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import Config
from .preprocessing import DataPreprocessor
from .train import PricePredictor
from .supabase_client import SupabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NERA Real Estate Price Prediction API",
    description="API for predicting real estate prices in Nigeria",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and preprocessor
model = None
preprocessor = None
supabase = None

# Request and Response Models
class PropertyFeatures(BaseModel):
    """Input features for price prediction"""
    state: str = Field(..., description="State where the property is located")
    lga: str = Field(..., description="Local Government Area")
    district: str = Field(..., description="District or neighborhood")
    bedrooms: int = Field(..., ge=0, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=0, description="Number of bathrooms")
    toilets: int = Field(..., ge=0, description="Number of toilets")
    property_type: str = Field(..., description="Type of property (e.g., 'apartment', 'duplex')")

class PricePrediction(BaseModel):
    """Prediction response model"""
    predicted_price: float = Field(..., description="Predicted price in NGN")
    confidence_interval: Dict[str, float] = Field(
        ...,
        description="Confidence interval for the prediction"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the prediction"
    )

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str
    model_loaded: bool
    model_version: Optional[str] = None

# Dependency to get the Supabase client
def get_supabase() -> SupabaseManager:
    """Dependency to get the Supabase client"""
    global supabase
    if supabase is None:
        supabase = SupabaseManager()
    return supabase

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the model and preprocessor on startup"""
    global model, preprocessor
    
    try:
        # Load the preprocessor
        preprocessor = DataPreprocessor.load_preprocessor(Config.MODEL_PATH)
        
        # Load the model
        model = PricePredictor.load_model(Config.MODEL_PATH)
        
        logger.info("Successfully loaded model and preprocessor")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Don't raise here to allow the app to start without a model
        # The /predict endpoint will return a 503 if the model isn't loaded

# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": Config.MODEL_VERSION if model else None
    }

# Prediction endpoint
@app.post("/predict", response_model=PricePrediction)
async def predict_price(
    property_data: PropertyFeatures,
    supabase: SupabaseManager = Depends(get_supabase)
):
    """
    Predict the price of a property based on its features
    
    Args:
        property_data: Property features for prediction
        
    Returns:
        PricePrediction: Predicted price with confidence interval
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([property_data.dict()])
        
        # Preprocess input
        X, _ = preprocessor.preprocess_data(input_data)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Calculate confidence interval (simplified for this example)
        # In production, you might want to use quantile regression or another method
        confidence_interval = {
            'lower_bound': prediction * 0.9,  # 10% below
            'upper_bound': prediction * 1.1,  # 10% above
            'confidence_level': 0.9
        }
        
        # Log the prediction (optional)
        logger.info(
            f"Prediction made for {property_data.bedrooms}-bedroom "
            f"{property_data.property_type} in {property_data.district}, "
            f"{property_data.lga}, {property_data.state}: ₦{prediction:,.2f}"
        )
        
        return {
            'predicted_price': float(prediction),
            'confidence_interval': confidence_interval,
            'metadata': {
                'model_version': Config.MODEL_VERSION,
                'features_used': property_data.dict(),
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error making prediction: {str(e)}"
        )

# Model info endpoint
@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        'model_type': 'LightGBM',
        'model_version': Config.MODEL_VERSION,
        'features': {
            'categorical': Config.CATEGORICAL_FEATURES,
            'numeric': Config.NUMERIC_FEATURES
        },
        'target': Config.TARGET,
        'feature_importance': model.feature_importance_ if hasattr(model, 'feature_importance_') else None,
        'metrics': model.training_metrics_ if hasattr(model, 'training_metrics_') else None
    }
