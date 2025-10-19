"""
Supabase client and utilities for the NERA Real Estate Price Prediction model
"""
import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from supabase import create_client, Client as SupabaseClient
from .config import Config

logger = logging.getLogger(__name__)

class SupabaseManager:
    """Manages interactions with Supabase"""
    
    def __init__(self):
        """Initialize Supabase client"""
        self.client: SupabaseClient = create_client(
            Config.SUPABASE_URL,
            Config.SUPABASE_SERVICE_ROLE_KEY
        )
    
    def fetch_properties_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch properties data from Supabase with pagination support
        
        Args:
            limit: Maximum number of rows to fetch. If None, fetches all rows.
            
        Returns:
            pd.DataFrame: DataFrame containing properties data
        """
        try:
            logger.info(f"Fetching data from {Config.PROPERTIES_TABLE}...")
            page_size = 1000  # Number of records per page
            offset = 0
            all_data = []
            
            while True:
                # Build the query with pagination
                query = self.client.table(Config.PROPERTIES_TABLE)\
                    .select("*")\
                    .range(offset, offset + page_size - 1)
                
                # Execute the query
                response = query.execute()
                
                if not response.data:
                    break  # No more data
                
                all_data.extend(response.data)
                logger.info(f"Fetched {len(all_data)} records so far...")
                
                # Apply limit if specified
                if limit is not None and len(all_data) >= limit:
                    all_data = all_data[:limit]
                    break
                    
                # Move to next page
                offset += page_size
                
                # If we got fewer records than page size, we've reached the end
                if len(response.data) < page_size:
                    break
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Log available columns for debugging
            logger.info(f"Available columns in {Config.PROPERTIES_TABLE}: {df.columns.tolist()}")
            
            # Ensure all required columns exist
            missing_columns = [col for col in Config.REQUIRED_FIELDS if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Required columns not found in the database: {missing_columns}")
            
            # Log data loading progress for larger datasets
            logger.info(f"Loaded {len(df)} records from {Config.PROPERTIES_TABLE}")
            if len(df) > 1000:
                logger.info(f"Processing large dataset ({len(df)} records). This may take a while...")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching properties data: {str(e)}")
            raise
    
    def save_model_run_metadata(
        self, 
        model_version: str,
        metrics: Dict[str, float],
        feature_importance: Dict[str, float],
        training_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Save model run metadata to the database
        
        Args:
            model_version: Version identifier for the model
            metrics: Dictionary of evaluation metrics
            feature_importance: Dictionary of feature importances
            training_params: Dictionary of training parameters
            
        Returns:
            Dict containing the inserted record
        """
        try:
            record = {
                'model_name': 'real_estate_price_predictor',
                'version': model_version,
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_parameters': training_params,
                'status': 'completed'
            }
            
            response = (
                self.client.table(Config.MODEL_RUNS_TABLE)
                .insert(record)
                .execute()
            )
            
            if not response.data:
                raise ValueError("Failed to save model run metadata")
                
            return response.data[0]
            
        except Exception as e:
            logger.error(f"Error saving model run metadata: {str(e)}")
            raise
    
    def upload_model_artifact(
        self, 
        file_path: str, 
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload a model artifact to Supabase Storage
        
        Args:
            file_path: Path to the file to upload
            content_type: MIME type of the file
            
        Returns:
            str: Public URL of the uploaded file
        """
        try:
            # Ensure the bucket exists
            try:
                self.client.storage.get_bucket(Config.MODEL_BUCKET)
            except Exception:
                # Create the bucket if it doesn't exist
                self.client.storage.create_bucket(
                    Config.MODEL_BUCKET,
                    public=False
                )
            
            # Generate storage path
            file_name = file_path.split("/")[-1]
            storage_path = f"{Config.MODEL_PATH}/{Config.MODEL_VERSION}/{file_name}"
            
            # Upload the file
            with open(file_path, 'rb') as f:
                self.client.storage \
                    .from_(Config.MODEL_BUCKET) \
                    .upload(file=storage_path, path=file_path, file_options={"content-type": content_type})
            
            # Get the public URL
            url = self.client.storage \
                .from_(Config.MODEL_BUCKET) \
                .get_public_url(storage_path)
                
            return url
            
        except Exception as e:
            logger.error(f"Error uploading model artifact: {str(e)}")
            raise
    
    def download_latest_model(self, local_path: str) -> str:
        """
        Download the latest model from Supabase Storage
        
        Args:
            local_path: Local path to save the downloaded model
            
        Returns:
            str: Path to the downloaded model file
        """
        try:
            # List files in the model path
            files = self.client.storage \
                .from_(Config.MODEL_BUCKET) \
                .list(Config.MODEL_PATH)
            
            if not files:
                raise FileNotFoundError("No model files found in storage")
            
            # Find the latest model version
            latest_file = max(files, key=lambda x: x['created_at'])
            storage_path = f"{Config.MODEL_PATH}/{latest_file['name']}"
            
            # Download the file
            with open(local_path, 'wb') as f:
                res = self.client.storage \
                    .from_(Config.MODEL_BUCKET) \
                    .download(storage_path)
                f.write(res)
                
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise
