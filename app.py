from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import shutil
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import uvicorn
from datetime import datetime
import logging
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the prediction router
from predict import router as prediction_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NeraDwell AI Prediction Service",
    description="API for uploading property data and generating predictions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the prediction router
app.include_router(prediction_router, prefix="/api", tags=["predictions"])

# Configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Response model for file upload
class UploadResponse(BaseModel):
    filename: str
    file_size: int
    file_type: str
    rows: int
    columns: List[str]
    preview: List[Dict[str, Any]]



@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Handle file uploads and return file metadata with a preview.
    """
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in [".csv", ".xlsx", ".xls"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a CSV or Excel file."
            )
        
        # Create a safe filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read the file for validation and preview
        try:
            if file_ext == ".csv":
                df = pd.read_csv(file_path)
            else:  # Excel files
                df = pd.read_excel(file_path)
            
            # Standardize column names (case-insensitive)
            df.columns = [col.strip().title() for col in df.columns]
            
            # Define required columns
            required_columns = [
                'Title', 'Price', 'Location', 'Bedrooms', 
                'Bathrooms', 'Toilets', 'Property Type'
            ]
            
            # Check for missing columns (case-insensitive)
            missing_columns = [
                col for col in required_columns 
                if col.lower() not in [c.lower() for c in df.columns]
            ]
            
            if missing_columns:
                # Clean up the uploaded file
                os.remove(file_path)
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required columns: {', '.join(missing_columns)}. "
                           f"Please ensure your file includes all required columns: "
                           f"{', '.join(required_columns)}"
                )
            
            # Select only the required columns and standardize their names
            df = df[required_columns].copy()
            
            # Convert preview rows to list of dicts
            preview_rows = df.head(100).replace({np.nan: None}).to_dict(orient='records')
            
            return {
                "filename": safe_filename,
                "file_size": os.path.getsize(file_path),
                "file_type": file.content_type,
                "rows": len(df),
                "columns": list(df.columns),
                "preview": preview_rows
            }
            
        except Exception as e:
            # Clean up the uploaded file if there's an error reading it
            if os.path.exists(file_path):
                os.remove(file_path)
            logger.error(f"Error processing file: {str(e)}")
            if not isinstance(e, HTTPException):
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing file: {str(e)}"
                )
            raise
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    finally:
        # Ensure the file is closed
        await file.close()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "nera-dwell-api"}

# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Welcome to NERA Dwell AI API",
        "documentation": "/docs",
        "endpoints": {
            "health": "/api/health (GET)",
            "upload": "/upload (POST)"
        },
        "status": "operational"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8002, reload=True)
