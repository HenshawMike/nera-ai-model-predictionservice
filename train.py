"""
Model training and evaluation for NERA Real Estate Price Prediction
"""
import logging
import os
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Config
from category_encoders import TargetEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PricePredictor:
    """
    A class to handle training, prediction, and evaluation of LightGBM models for price prediction.
    Includes cross-validation and hyperparameter tuning with Optuna.
    """
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the PricePredictor with optional model parameters.
        
        Args:
            model_params: Dictionary of LightGBM parameters. If None, uses default from Config.
        """
        self.model = None
        self.model_params = model_params or Config.MODEL_PARAMS
        self.feature_importance_ = None
        self.training_metrics_ = {}
        self.log_target = True  # Whether to use log transformation on target
        self.cv_scores_ = None  # To store cross-validation scores
        self.best_iteration_ = None  # To store best iteration from training
        self.preprocessor = None

    def run_cv_with_encoding(self, X, y, params, n_splits):
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=Config.RANDOM_STATE)
        metrics_list = []

        logger.info(f"Starting {n_splits}-fold cross-validation for tuning...")

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Separate title from other features
            X_train_title = X_train['title']
            X_val_title = X_val['title']
            X_train_other = X_train.drop(columns=['title'])
            X_val_other = X_val.drop(columns=['title'])

            # TF-IDF Vectorization for title
            tfidf_vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
            X_train_title_tfidf = tfidf_vectorizer.fit_transform(X_train_title)
            X_val_title_tfidf = tfidf_vectorizer.transform(X_val_title)

            # Target Encoding for categorical features
            encoder = TargetEncoder(cols=Config.CATEGORICAL_FEATURES, handle_unknown='value', return_df=True)
            X_train_other_encoded = encoder.fit_transform(X_train_other, y_train)
            X_val_other_encoded = encoder.transform(X_val_other)

            # Combine features
            X_train_combined = hstack([X_train_other_encoded, X_train_title_tfidf]).tocsr()
            X_val_combined = hstack([X_val_other_encoded, X_val_title_tfidf]).tocsr()
            
            tfidf_feature_names = [f"tfidf_{name}" for name in tfidf_vectorizer.get_feature_names_out().tolist()]
            feature_names = list(X_train_other_encoded.columns) + tfidf_feature_names

            # Train a temporary model on this fold
            temp_predictor = PricePredictor(model_params=params)
            metrics = temp_predictor.train(X_train_combined, y_train, X_val_combined, y_val, log_target=True, feature_names=feature_names)
            metrics_list.append(metrics)

        # Calculate average metrics across folds
        avg_metrics = {}
        for key in metrics_list[0].keys():
            avg_metrics[f'avg_{key}'] = np.mean([m[key] for m in metrics_list])
            avg_metrics[f'std_{key}'] = np.std([m[key] for m in metrics_list])
        
        return avg_metrics

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, preprocessor, n_trials: int = 50,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Features (before encoding)
            y: Target values
            preprocessor: The data preprocessor instance
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary of best parameters and metrics
        """
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'random_state': Config.RANDOM_STATE,
                'verbosity': -1,
                'n_jobs': -1,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_data_in_leaf', 20, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            # Run cross-validation with proper encoding
            metrics = self.run_cv_with_encoding(X, y, params, n_splits=cv_folds)
            
            return metrics['avg_val_rmse']
        
        # Run optimization
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=Config.RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials)
        
        # Update model with best parameters
        self.model_params.update(study.best_params)
        
        # Train the final model on the full training data
        logger.info("Training final model with best hyperparameters on the full training data...")
        # 1. Fit the final encoders on the full training data
        preprocessor.encoder.fit(X.drop(columns=['title']), y)
        preprocessor.tfidf_vectorizer.fit(X['title'])

        # 2. Transform the full training data
        X_other_encoded = preprocessor.encoder.transform(X.drop(columns=['title']))
        X_title_tfidf = preprocessor.tfidf_vectorizer.transform(X['title'])

        # 3. Combine features for final model training
        X_combined = hstack([X_other_encoded, X_title_tfidf]).tocsr()

        # 4. Get feature names for the final model
        tfidf_feature_names = [f"tfidf_{name}" for name in preprocessor.tfidf_vectorizer.get_feature_names_out().tolist()]
        feature_names = list(X_other_encoded.columns) + tfidf_feature_names

        # 5. Train the final model
        self.train(X_combined, y, log_target=True, feature_names=feature_names)
        # 6. Store the fitted preprocessor
        self.preprocessor = preprocessor
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
        }
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
             X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
             params: Optional[Dict[str, Any]]=None, log_target: bool = True, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Train the LightGBM model with optional validation set and early stopping.
        
        Args:
            X_train: Training features
            y_train: Training target values
            X_val: Optional validation features
            y_val: Optional validation target values
            log_target: Whether to apply log1p transformation to target
            feature_names: Optional list of feature names
            
        Returns:
            Dictionary of training metrics
        """
        self.log_target = log_target
        
        # Apply log transformation if specified
        if log_target:
            y_train = np.log1p(y_train)
            if y_val is not None:
                y_val = np.log1p(y_val)
        
        # Convert data to LightGBM Dataset format
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
        
        # Prepare validation data if provided
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names, free_raw_data=False)
            valid_sets.append(valid_data)
            valid_names.append('val')
        
        # Initialize callbacks
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
                verbose=True
            ),
            lgb.log_evaluation(100),
        ]
        
        # Train the model
        self.model = lgb.train(
            params=self.model_params,
            train_set=train_data,
            num_boost_round=Config.MODEL_PARAMS.get('n_estimators', 2000),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # Store best iteration
        self.best_iteration_ = self.model.best_iteration
        
        # Store feature importance
        self.feature_importance_ = dict(
            zip(self.model.feature_name(), 
                self.model.feature_importance(importance_type='gain'))
        )
        
        # Calculate and store metrics
        metrics = {}
        # Get predictions on the log scale to match y_train/y_val
        y_pred_train_log = self.predict(X_train, apply_inverse_transform=False)
        metrics.update(self._calculate_metrics(y_train, y_pred_train_log, 'train'))
        
        if X_val is not None and y_val is not None:
            y_pred_val_log = self.predict(X_val, apply_inverse_transform=False)
            metrics.update(self._calculate_metrics(y_val, y_pred_val_log, 'val'))
        
        self.training_metrics_ = metrics
        return metrics

    def _calculate_metrics(self, y_true, y_pred, prefix=''):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Per user request, use a custom MAPE function to avoid exploding values
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6)))

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {f"{prefix}_rmse": rmse, f"{prefix}_mae": mae, f"{prefix}_r2": r2, f"{prefix}_mape": mape}
        for k, v in metrics.items():
            if 'r2' in k: 
                logger.info(f"{k}: {v:.4f}")
            else: 
                logger.info(f"{k}: {v:.2f}")
        return metrics

    def predict(self, X: pd.DataFrame, apply_inverse_transform: Optional[bool] = None) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features for prediction
            apply_inverse_transform: Whether to apply inverse log transformation.
                                   If None, uses self.log_target.
            
        Returns:
            Array of predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        if apply_inverse_transform is None:
            apply_inverse_transform = self.log_target
            
        # Make predictions
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        # Apply inverse log transformation if needed
        if apply_inverse_transform:
            predictions = np.expm1(predictions)
            
        return predictions

    def save_model(self, path=None):
        if path is None: 
            path = os.path.join(self.model_dir, 'model.txt')
        self.model.save_model(path)
        return path

def train_and_tune_model(X_train, y_train, preprocessor, n_trials=50, cv_folds=5, **kwargs):
    """
    Train and tune the model using Optuna.
    """
    predictor = PricePredictor()

    # Reduce n_trials for faster execution in this context.
    # In a real scenario, this would be higher.
    tuning_results = predictor.tune_hyperparameters(X_train, y_train, preprocessor, n_trials=n_trials, cv_folds=cv_folds)

    logger.info("Best hyperparameters found:")
    logger.info(json.dumps(tuning_results['best_params'], indent=2))

    # Return the predictor instance which contains the trained model and fitted preprocessor
    return predictor

def evaluate_model(model, X_test, y_test, log_target=True):
    """
    Evaluate model performance with RMSE in millions of Naira

    Args:
        model: Trained model
        X_test: Test features
        y_test: True target values (in original Naira)
        log_target: Whether the target was log-transformed during training

    Returns:
        dict: Dictionary of evaluation metrics with RMSE in millions
    """
    # Make predictions
    preds = model.predict(X_test)

    # Convert predictions back to original scale if log was used
    if log_target:
        preds = np.expm1(preds)

        # Ensure y_test is in original scale
        if np.any(y_test < 0):
            logger.warning("Negative values found in y_test. Make sure it's in original scale.")

    # Convert both actual and predicted values to millions
    SCALE_FACTOR = 1_000_000  # 1 million Naira
    y_test_millions = y_test / SCALE_FACTOR
    preds_millions = preds / SCALE_FACTOR

    # Calculate metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test_millions, preds_millions)),  # Now in millions
        'mape': np.mean(np.abs((y_test - preds) / np.maximum(np.abs(y_test), 1e-6))),  # Keep MAPE in original scale
        'r2': r2_score(y_test, preds)  # R² is scale-invariant
    }

    # Log the scale of values for debugging
    logger.info(f"Prediction range: {preds.min():.2f} to {preds.max():.2f}")
    logger.info(f"Actual range: {y_test.min():.2f} to {y_test.max():.2f}")

    return metrics


def save_model_artifacts(model, preprocessor, metrics):
    model_dir = Config.MODEL_DIR / f"model_{Config.MODEL_VERSION}"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / 'model.joblib')
    with open(model_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    return str(model_dir)


def load_model(model_dir: str = 'models') -> 'PricePredictor':
    """
    Load a trained model from disk
    
    Args:
        model_dir: Directory containing the saved model
        
    Returns:
        PricePredictor: Loaded model instance
    """
    model_path = os.path.join(model_dir, 'model.txt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    predictor = PricePredictor(model_dir=model_dir)
    predictor.model = lgb.Booster(model_file=model_path)
    
    # Load feature importance if available
    importance_path = os.path.join(model_dir, 'feature_importance.json')
    if os.path.exists(importance_path):
        with open(importance_path, 'r') as f:
            predictor.feature_importance_ = json.load(f)
    
    # Load training metrics if available
    metrics_path = os.path.join(model_dir, 'training_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            predictor.training_metrics_ = json.load(f)
    
    return predictor

def upload_to_supabase(local_path: str, bucket_name: str = 'model-registry') -> bool:
    """
    Upload files to Supabase Storage with automatic bucket creation

    Args:
        local_path: Path to local file or directory to upload
        bucket_name: Name of the Supabase storage bucket (default: 'model-registry')

    Returns:
        bool: True if upload was successful, False otherwise
    """
    try:
        from supabase import create_client
        import os
        from pathlib import Path
        from urllib.parse import urljoin
        import json

        # Initialize Supabase client
        supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        )

        # Create bucket if it doesn't exist
        try:
            supabase.storage.create_bucket(
                bucket_name,
                public=True,
                file_size_limit=1024 * 1024 * 100  # 100MB limit
            )
            logger.info(f"Created bucket: {bucket_name}")
        except Exception as e:
            if 'Bucket already exists' not in str(e):
                logger.error(f"Error creating bucket: {str(e)}")
                return False

        # Get list of files to upload
        if os.path.isdir(local_path):
            file_paths = [os.path.join(root, f) for root, _, files in os.walk(local_path) 
                         for f in files if not f.startswith('.')]  # Skip hidden files
        else:
            file_paths = [local_path]

        # Upload each file
        for file_path in file_paths:
            try:
                # Create relative path for storage (handle Windows paths)
                rel_path = str(Path(file_path).relative_to(Path(local_path).parent))
                storage_path = rel_path.replace('\\', '/')  # Convert Windows paths to forward slashes
                
                # Read file data
                with open(file_path, 'rb') as f:
                    file_data = f.read()

                # Determine content type
                content_type = "application/octet-stream"
                if file_path.endswith('.json'):
                    content_type = "application/json"
                elif file_path.endswith(('.jpg', '.jpeg', '.png')):
                    content_type = f"image/{file_path.split('.')[-1]}"

                # Upload to Supabase Storage
                supabase.storage.from_(bucket_name).upload(
                    path=storage_path,
                    file=file_data,
                    file_options={"content-type": content_type}
                )
                logger.info(f"Uploaded {storage_path} to {bucket_name}")
                
            except Exception as file_error:
                logger.error(f"Error uploading {file_path}: {str(file_error)}")
                continue

        return True
        
    except Exception as e:
        logger.error(f"Error in upload_to_supabase: {str(e)}", exc_info=True)
        return False
