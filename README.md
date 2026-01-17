# Nera AI Prediction Service

A FastAPI-based prediction service for property data analysis and predictions.

## Features

- Upload property data in CSV or Excel format
- Generate predictions using pre-trained models
- RESTful API endpoints for easy integration
- Containerized with Docker for easy deployment
- Environment variable configuration
- Comprehensive logging

## Prerequisites

- Python 3.8+
- Docker (optional, for containerized deployment)
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Nera-ai/backend/prediction_service
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with the following variables:
   ```
   # Server configuration
   HOST=0.0.0.0
   PORT=8002
   
   # Model paths
   MODEL_PATH=./models/your_model.pkl
   ```

## Usage

### Running the Service

#### Development Mode
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8002
```

#### Production Mode
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
```

#### Using Docker
```bash
docker build -t nera-prediction-service .
docker run -p 8002:8002 nera-prediction-service
```

### API Endpoints

- `GET /`: Service information
- `GET /health`: Health check endpoint
- `POST /upload`: Upload property data file (CSV/Excel)
- `POST /predict`: Generate predictions for uploaded data

### Example API Request

```bash
curl -X POST "http://localhost:8002/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/property_data.csv"
```

## Project Structure

```
prediction_service/
├── app.py               # Main FastAPI application
├── predict.py           # Prediction logic and endpoints
├── setup_models.py      # Model loading and setup
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── .env                # Environment variables
├── models/             # Pre-trained models
├── uploads/            # Temporary file storage
└── predicted_data/     # Output predictions
```

## Development

### Setting Up Development Environment

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt  # If available
   ```

2. Configure pre-commit hooks (if available):
   ```bash
   pre-commit install
   ```

### Running Tests

```bash
pytest tests/
```

## Deployment

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t nera-prediction-service .
   ```

2. Run the container:
   ```bash
   docker run -d -p 8002:8002 --env-file .env nera-prediction-service
   ```

### Cloud Deployment

The service can be deployed to any cloud platform that supports containerized applications (e.g., AWS ECS, Google Cloud Run, Azure Container Instances).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Specify License]
