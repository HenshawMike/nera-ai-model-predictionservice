"""
Configuration settings for the NERA Real Estate Price Prediction model
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

class Config:
    """Application configuration"""
    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    # Debug mode
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Model
    MODEL_BUCKET = "model_registry"
    MODEL_PATH = str(BASE_DIR / "models")
    MODEL_DIR = Path(MODEL_PATH)
    MODEL_VERSION = "v1"
    
    # Model training parameters
    MODEL_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'reg_alpha': 0.1,         # L1 regularization
        'reg_lambda': 0.1,        # L2 regularization
        'n_estimators': 2000,     # Increased number of trees
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1,
        'importance_type': 'gain',
    }
    
    # Training settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EARLY_STOPPING_ROUNDS = 100
    
    # Feature engineering settings
    OUTLIER_LOWER_PERCENTILE = 0.01
    OUTLIER_UPPER_PERCENTILE = 0.99
    
    # Database
    PROPERTIES_TABLE = "abuja_properties"
    MODEL_RUNS_TABLE = "model_runs"
    
    # Required fields for prediction
    REQUIRED_FIELDS = [
        'title', 'lga', 'district_or_estate', 'bedrooms',
        'bathrooms', 'toilets', 'property_type_standardized', 'price_numeric'
    ]
    
    # Categorical features
    CATEGORICAL_FEATURES = ['lga', 'district_or_estate', 'property_type_standardized']
    
    # Numeric features
    NUMERIC_FEATURES = ['bedrooms', 'bathrooms', 'toilets']
    
    # Target variable
    TARGET = 'price_numeric'
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        missing = []
        if not cls.SUPABASE_URL:
            missing.append("SUPABASE_URL")
        if not cls.SUPABASE_SERVICE_ROLE_KEY:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
            
        if missing:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing)}. "
                "Please set these environment variables in your .env file."
            )
        
        # Create model directory if it doesn't exist
        os.makedirs(cls.MODEL_PATH, exist_ok=True)

# Validate configuration on import
Config.validate()
