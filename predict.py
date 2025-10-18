import os
import json
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, List
import logging
from pathlib import Path
import sys
import joblib

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Now import the modules from the model package
from clean_real_estate_data import RealEstateDataCleaner
from model.preprocessing import DataPreprocessor, NEW_NUMERIC_FEATURES
from model.config import Config
from scipy.sparse import hstack
import lightgbm as lgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI router
router = APIRouter()

# Configuration
UPLOAD_DIR = "uploads"
# Define paths for both model and preprocessor (relative to this file)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "model_v1", "model.joblib")
PREPROCESSOR_PATH = os.path.join(SCRIPT_DIR, "models", "v1", "preprocessor.joblib")


class PredictionService:
    def __init__(self):
        self.model = self._load_model()
        self.preprocessor = self._load_preprocessor()

    def _load_model(self) -> lgb.Booster:
        """Load the trained LightGBM model."""
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            predictor = joblib.load(MODEL_PATH)
            if isinstance(predictor, lgb.Booster):
                return predictor
            # If the saved object is the PricePredictor class, extract the model
            elif hasattr(predictor, 'model') and isinstance(predictor.model, lgb.Booster):
                return predictor.model
            else:
                 raise TypeError(f"Loaded object from {MODEL_PATH} is not a LightGBM Booster or a PricePredictor instance.")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_preprocessor(self) -> DataPreprocessor:
        """Load the saved DataPreprocessor instance."""
        try:
            if not os.path.exists(PREPROCESSOR_PATH):
                raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
            return DataPreprocessor.load_preprocessor(model_dir=os.path.dirname(PREPROCESSOR_PATH))
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            raise

    async def predict(self, file_path: str) -> Dict[str, Any]:
        """
        Generate predictions for the given file.
        """
        try:
            logger.info(f"Starting cleaning for file: {file_path}")
            cleaner = RealEstateDataCleaner(file_path)
            cleaned_file_path = cleaner.run_cleaning_pipeline()
            logger.info(f"Data cleaned successfully. Cleaned file at: {cleaned_file_path}")
            
            df_cleaned = pd.read_csv(cleaned_file_path)
            df_for_results = df_cleaned.copy()

            features = self._prepare_features(df_cleaned)
            
            logger.info("Generating predictions...")
            # The model was trained on log-transformed prices, so we predict on that scale
            log_predictions = self.model.predict(features)
            # Inverse transform to get actual price predictions
            predictions = np.expm1(log_predictions)
            logger.info("Predictions generated.")
            
            results = self._format_results(df_for_results, predictions)
            
            return {
                "status": "success",
                "predictions": results,
                "summary": self._generate_summary(results)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Error generating predictions: {str(e)}"
            )
    
    def _prepare_features(self, df: pd.DataFrame):
        """
        Prepare features for model prediction using the loaded preprocessor.
        """
        logger.info("Preparing features for prediction...")

        # --- Replicate the feature engineering from DataPreprocessor ---
        if 'price_numeric' not in df.columns:
            df['price_numeric'] = 0

        df['rooms_count'] = df['bedrooms'] + df['bathrooms'] + df['toilets']
        df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 0.5)
        df['bedrooms_x_bathrooms'] = df['bedrooms'] * df['bathrooms']
        df['price_per_bedroom'] = df['price_numeric'] / df['bedrooms'].clip(lower=1)

        # Use a try-except block for groupby transformations as some groups might not exist in smaller datasets
        try:
            df['lga_median_bedrooms'] = df.groupby('lga')['bedrooms'].transform('median')
            df['lga_median_bathrooms'] = df.groupby('lga')['bathrooms'].transform('median')
            df['district_median_bedrooms'] = df.groupby('district_or_estate')['bedrooms'].transform('median')
            df['district_median_bathrooms'] = df.groupby('district_or_estate')['bathrooms'].transform('median')
        except Exception:
            logger.warning("Could not create all location-based aggregate features. This might happen with new locations not seen in training.")
            pass # Silently pass if a group does not exist

        # Fill NaNs created by groupby, using the overall median from the training data
        for col in NEW_NUMERIC_FEATURES:
            if col in df.columns:
                impute_val = self.preprocessor.imputation_values.get(col, df[col].median())
                df[col] = df[col].fillna(impute_val)

        # Ensure all feature columns from training exist
        feature_cols = Config.CATEGORICAL_FEATURES + Config.NUMERIC_FEATURES + NEW_NUMERIC_FEATURES
        for col in feature_cols:
            if col not in df.columns:
                df[col] = self.preprocessor.imputation_values.get(col, 0)

        # Standardize text columns to match training
        for col in Config.CATEGORICAL_FEATURES + ['title']:
             if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
        
        # --- Apply the fitted encoders from the preprocessor ---
        logger.info("Applying fitted TargetEncoder and TfidfVectorizer...")

        # Separate features for encoding, matching the logic from generate_preprocessor.py
        X_categorical = df[Config.CATEGORICAL_FEATURES]
        X_numeric_and_new = df[Config.NUMERIC_FEATURES + NEW_NUMERIC_FEATURES]
        X_title = df['title']

        # The preprocessor's encoder was fitted *only* on categorical features.
        # We must transform them in isolation.
        logger.info(f"Transforming {len(Config.CATEGORICAL_FEATURES)} categorical features...")
        X_categorical_encoded = self.preprocessor.encoder.transform(X_categorical)

        # Reconstruct the full feature dataframe in the correct order to match the model's training data.
        X_other_combined = pd.concat([X_categorical_encoded, X_numeric_and_new], axis=1)

        # Transform the title using the TF-IDF vectorizer
        logger.info("Transforming 'title' feature with TfidfVectorizer...")
        X_title_tfidf = self.preprocessor.tfidf_vectorizer.transform(X_title)

        # Combine features into a single sparse matrix
        # Note: hstack works with pandas DataFrames and sparse matrices
        X_combined = hstack([X_other_combined, X_title_tfidf]).tocsr()

        logger.info(f"Feature preparation complete. Shape: {X_combined.shape}")

        return X_combined

    def _format_results(self, df: pd.DataFrame, predictions: np.ndarray) -> List[Dict]:
        """Format prediction results"""
        results = []
        df_copy = df.copy()
        df_copy['predicted_price'] = predictions

        for idx, row in df_copy.iterrows():
            # Combine location parts for a readable location string, ensuring all parts are strings
            raw_location_parts = [
                row.get('district_or_estate'),
                row.get('lga'),
                row.get('state')
            ]
            location = ', '.join(str(part) for part in raw_location_parts if part and pd.notna(part))
            
            # Calculate confidence score (0-1)
            actual_price = float(row.get('price_numeric', 0))
            predicted_price = float(row.get('predicted_price', 0))
            
            if actual_price > 0:
                # If we have actual price, base confidence on percentage difference
                price_diff = abs(predicted_price - actual_price) / actual_price
                # Convert to 0-1 scale (smaller difference = higher confidence)
                confidence = max(0, min(1, 1 - min(price_diff, 1.0)))
            else:
                # Base confidence on data quality and completeness
                confidence = 0.0  # Default confidence when no actual price is available
                
                # Adjust confidence based on data quality
                if pd.notna(row.get('bedrooms')) and pd.notna(row.get('bathrooms')):
                    confidence += 0.1
                if pd.notna(row.get('toilets')):
                    confidence += 0.05
                if location.strip():
                    confidence += 0.05
                    
                # Ensure confidence is between 0.7 and 0.95
                confidence = max(0.7, min(0.95, confidence))

            results.append({
                "property_id": idx + 1,
                "title": row.get('title', ''),
                "location": location,
                "property_type": row.get('property_type_standardized', ''),
                "bedrooms": int(row.get('bedrooms', 0)) if pd.notna(row.get('bedrooms')) else None,
                "bathrooms": int(row.get('bathrooms', 0)) if pd.notna(row.get('bathrooms')) else None,
                "toilets": int(row.get('toilets', 0)) if pd.notna(row.get('toilets')) else None,
                "predicted_price": predicted_price,
                "actual_price": actual_price if actual_price > 0 else None,
                "price_difference": predicted_price - actual_price if actual_price > 0 else None,
                "confidence": round(confidence, 2)  # Round to 2 decimal places
            })
        return results
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics"""
        if not results:
            return {}
            
        prices = [r["actual_price"] for r in results if r["actual_price"] is not None]
        predicted_prices = [r["predicted_price"] for r in results if r["predicted_price"] is not None]
        
        return {
            "total_properties": len(results),
            "avg_actual_price": sum(prices) / len(prices) if prices else 0,
            "avg_predicted_price": sum(predicted_prices) / len(predicted_prices) if predicted_prices else 0,
            "min_price": min(prices) if prices else 0,
            "max_price": max(prices) if prices else 0,
        }

# Initialize prediction service
prediction_service = PredictionService()

@router.post("/predict")
async def predict_properties(data: Dict[str, Any]):
    """
    Generate predictions for property data
    
    Expected JSON payload:
    {
        "filename": "path/to/uploaded/file.csv"
    }
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, data.get("filename", ""))
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail="File not found. Please upload the file first."
            )
        
        # Generate predictions
        result = await prediction_service.predict(file_path)
        
        return {
            "status": "success",
            "data": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing prediction request: {str(e)}"
        )
