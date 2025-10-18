"""
Script to download and verify model files for the prediction service.
This script should be run during the build process.
"""
import os
import sys
import logging
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model paths - using relative paths from the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "model_v1", "model.joblib")
PREPROCESSOR_PATH = os.path.join(SCRIPT_DIR, "models", "v1", "preprocessor.joblib")

def ensure_model_files():
    """Ensure model files exist in the expected locations."""
    # Create model directories if they don't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
    
    # Check if model files exist
    model_exists = os.path.exists(MODEL_PATH)
    preprocessor_exists = os.path.exists(PREPROCESSOR_PATH)
    
    if model_exists and preprocessor_exists:
        logger.info("Model files already exist in the expected locations.")
        return True
    
    # If we're in production and files are missing, we should download them
    if os.getenv('ENVIRONMENT') == 'production':
        logger.error("Model files are missing in production environment!")
        logger.error(f"Expected model at: {MODEL_PATH}")
        logger.error(f"Expected preprocessor at: {PREPROCESSOR_PATH}")
        return False
    
    # In development, we can copy from a local path if available
    # This is a fallback and should be replaced with actual download logic
    local_model_path = os.path.join("local_models", "model.joblib")
    local_preprocessor_path = os.path.join("local_models", "preprocessor.joblib")
    
    if os.path.exists(local_model_path) and not model_exists:
        shutil.copy2(local_model_path, MODEL_PATH)
        logger.info(f"Copied model to {MODEL_PATH}")
    
    if os.path.exists(local_preprocessor_path) and not preprocessor_exists:
        shutil.copy2(local_preprocessor_path, PREPROCESSOR_PATH)
        logger.info(f"Copied preprocessor to {PREPROCESSOR_PATH}")
    
    return os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH)

if __name__ == "__main__":
    success = ensure_model_files()
    if not success:
        logger.error("Failed to ensure model files are available")
        sys.exit(1)
    logger.info("Model files are ready")
    sys.exit(0)
