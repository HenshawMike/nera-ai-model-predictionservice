"""
Data preprocessing module for NERA Real Estate Price Prediction
"""
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
import joblib
import os
from .config import Config
from .supabase_client import SupabaseClient
from category_encoders import TargetEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

NEW_NUMERIC_FEATURES = [
    'rooms_count',
    'bed_bath_ratio',
    'bedrooms_x_bathrooms',
    'lga_median_bedrooms',
    'lga_median_bathrooms',
    'district_median_bedrooms',
    'district_median_bathrooms',
    'price_per_bedroom',
]

class DataPreprocessor:
    """Handles data preprocessing for the real estate price prediction model"""
    
    def __init__(self, model_dir: str = 'models'):
        """
        Initialize the data preprocessor
        
        Args:
            model_dir: Directory to save preprocessing artifacts
        """
        self.model_dir = model_dir
        self.imputation_values: Dict[str, Any] = {}
        self.encoder = TargetEncoder(cols=Config.CATEGORICAL_FEATURES, handle_unknown='value', return_df=True)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the input DataFrame with enhanced cleaning and feature engineering
        
        Args:
            df: Input DataFrame containing property data
            
        Returns:
            Tuple containing features (X) and target (y)
        """
        try:
            # Make a copy to avoid modifying the original DataFrame
            df = df.copy()
            
            # Ensure required columns exist
            self._validate_columns(df)
            
            # Convert numeric columns
            df = self._convert_numeric_columns(df)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Clean text columns
            df = self._clean_text_columns(df)
            
            # Handle outliers in the target variable
            df = self._handle_outliers(df)
            
            # Feature Engineering
            df = self._create_features(df)

            # Extract features and target
            # The 'title' column will be handled separately in the training script
            feature_cols = Config.CATEGORICAL_FEATURES + Config.NUMERIC_FEATURES + NEW_NUMERIC_FEATURES
            X = df[['title'] + feature_cols]
            y = df[Config.TARGET]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error during data preprocessing: {str(e)}")
            raise
    
    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Validate that all required columns exist in the DataFrame"""
        missing_columns = [col for col in Config.REQUIRED_FIELDS if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric columns to appropriate dtypes"""
        for col in Config.NUMERIC_FEATURES + [Config.TARGET]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame"""
        # Fill missing 'lga' with 'district_or_estate' where possible
        if 'lga' in df.columns and 'district_or_estate' in df.columns:
            df['lga'].fillna(df['district_or_estate'], inplace=True)

        # Calculate imputation values (median for numeric, mode for categorical)
        if not self.imputation_values:
            for col in Config.NUMERIC_FEATURES:
                # Group by a more stable feature like 'property_type_standardized'
                if not df.empty:
                    self.imputation_values[col] = df.groupby(
                        'property_type_standardized'
                    )[col].transform('median').median()
                else:
                    self.imputation_values[col] = 0
                
            for col in Config.CATEGORICAL_FEATURES:
                self.imputation_values[col] = df[col].mode()[0] if not df[col].empty else 'unknown'
        
        # Apply imputation
        for col in Config.NUMERIC_FEATURES + [Config.TARGET]:
            if col in df.columns:
                df[col] = df[col].fillna(self.imputation_values.get(col, 0))
                
        for col in Config.CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(self.imputation_values.get(col, 'unknown'))
        
        return df
    
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize text columns"""
        for col in Config.CATEGORICAL_FEATURES + ['title']:
            if col in df.columns:
                # Convert to string, strip whitespace, and convert to lowercase
                df[col] = df[col].astype(str).str.strip().str.lower()

        # Group rare categories in 'district_or_estate'
        if 'district_or_estate' in df.columns:
            counts = df['district_or_estate'].value_counts()
            mask = df['district_or_estate'].isin(counts[counts < 5].index)
            df.loc[mask, 'district_or_estate'] = 'other'

        return df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features"""
        # Basic feature
        df['rooms_count'] = df['bedrooms'] + df['bathrooms'] + df['toilets']

        # Interaction features
        df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms'].replace(0, 0.5)
        df['bedrooms_x_bathrooms'] = df['bedrooms'] * df['bathrooms']
        df['price_per_bedroom'] = df['price_numeric'] / df['bedrooms'].clip(lower=1)

        # Location-based aggregate features
        df['lga_median_bedrooms'] = df.groupby('lga')['bedrooms'].transform('median')
        df['lga_median_bathrooms'] = df.groupby('lga')['bathrooms'].transform('median')
        df['district_median_bedrooms'] = df.groupby('district_or_estate')['bedrooms'].transform('median')
        df['district_median_bathrooms'] = df.groupby('district_or_estate')['bathrooms'].transform('median')

        # Fill any NaNs created by the groupby operations
        for col in ['lga_median_bedrooms', 'lga_median_bathrooms', 'district_median_bedrooms', 'district_median_bathrooms']:
            df[col].fillna(df[col.split('_median_')[1]].median(), inplace=True)

        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in the target variable using quantile-based clipping."""
        if Config.TARGET not in df.columns:
            return df
            
        df_clean = df.copy()
        
        # Handle zeros and negative prices
        df_clean = df_clean[df_clean['price_numeric'] > 0].copy()
        
        # Remove extreme outliers (top and bottom 1%)
        lower = df_clean['price_numeric'].quantile(0.01)
        upper = df_clean['price_numeric'].quantile(0.99)
        
        clipped_count_lower = (df_clean['price_numeric'] < lower).sum()
        clipped_count_upper = (df_clean['price_numeric'] > upper).sum()
        
        df_clean = df_clean[
            (df_clean['price_numeric'] >= lower) &
            (df_clean['price_numeric'] <= upper)
        ].copy()
        
        logger.info(f"Removed {clipped_count_lower} records with prices below {lower:.2f}")
        logger.info(f"Removed {clipped_count_upper} records with prices above {upper:.2f}")
        
        return df_clean
    
    def encode_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical features using Target Encoding.
        The encoder is fit on the training data and then used to transform both train and test sets.
        """
        # Fit the encoder on the training data
        self.encoder.fit(X_train, y_train)

        # Transform the training and test data
        X_train_encoded = self.encoder.transform(X_train)
        X_test_encoded = self.encoder.transform(X_test)

        return X_train_encoded, X_test_encoded

    def split_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple containing X_train, X_test, y_train, y_test
        """
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=X[['state', 'property_type']] if 'state' in X.columns and 'property_type' in X.columns else None
        )
    
    def save_preprocessor(self, file_path: Optional[str] = None) -> str:
        """
        Save the preprocessor state
        
        Args:
            file_path: Path to save the preprocessor. If None, uses default path.
            
        Returns:
            str: Path to the saved preprocessor
        """
        if file_path is None:
            file_path = os.path.join(self.model_dir, 'preprocessor.joblib')
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save all preprocessing artifacts
        artifacts = {
            'imputation_values': self.imputation_values,
            'encoder': self.encoder,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        joblib.dump(artifacts, file_path)
        return file_path
    
    @classmethod
    def load_preprocessor(cls, model_dir: str = 'models') -> 'DataPreprocessor':
        """
        Load a saved preprocessor
        
        Args:
            model_dir: Directory containing the saved preprocessor
            
        Returns:
            DataPreprocessor: Loaded preprocessor instance
        """
        file_path = os.path.join(model_dir, 'preprocessor.joblib')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessor file not found at {file_path}")
            
        preprocessor = cls(model_dir=model_dir)
        artifacts = joblib.load(file_path)
        preprocessor.imputation_values = artifacts['imputation_values']
        preprocessor.encoder = artifacts['encoder']
        preprocessor.tfidf_vectorizer = artifacts.get('tfidf_vectorizer', TfidfVectorizer())
        return preprocessor


def load_and_preprocess_data(supabase_client):
    """
    Load and preprocess data from Supabase
    
    Args:
        supabase_client: Initialized Supabase client (SupabaseManager instance)
        
    Returns:
        A tuple of (Preprocessed DataFrame, DataPreprocessor instance)
    """
    try:
        logger.info("Loading data from Supabase...")
        df = supabase_client.fetch_properties_data()

        if df is None or df.empty:
            raise ValueError("No data retrieved from Supabase")
            
        logger.info(f"Loaded {len(df)} records")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Preprocess data
        X, y = preprocessor.preprocess_data(df)
        
        # Combine features and target
        X[Config.TARGET] = y
        
        return X, preprocessor
        
    except Exception as e:
        logger.error(f"Error in load_and_preprocess_data: {str(e)}")
        raise


def split_data(df, test_size=0.2, random_state=42):
    """
    Split data into train and test sets
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    try:
        # Separate features and target
        feature_cols = Config.CATEGORICAL_FEATURES + Config.NUMERIC_FEATURES + NEW_NUMERIC_FEATURES
        X = df[['title'] + feature_cols]
        y = df[Config.TARGET]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        logger.info(f"Split data into {len(X_train)} training and {len(X_test)} test samples")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error in split_data: {str(e)}")
        raise
