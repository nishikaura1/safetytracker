# Safety Tracker API

A FastAPI-based service that predicts safety scores for routes using machine learning, incorporating real-time data from various sources including weather, crime, and terrain data.

## Features

- Real-time safety score prediction using Random Forest Regressor
- Integration with LAPD crime data
- Weather forecasting integration
- Route optimization using OpenRouteService
- Automatic model retraining and versioning
- Cross-validation and model metrics
- Feature importance analysis

## Project Structure

```
safetytracker/
├── app.py                 # Main FastAPI application
├── config.py             # Configuration and constants
├── models/
│   └── safety_model.py   # ML model implementation
├── services/
│   └── external_apis.py  # External API integrations
├── utils/
│   └── helpers.py        # Helper functions
├── data/                 # Data storage directory
├── models/              # Model storage directory
└── requirements.txt     # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export ORS_API_KEY="your_openrouteservice_api_key"
export OPENWEATHERMAP_API_KEY="your_openweathermap_api_key"
```

## Running the Application

Start the FastAPI server:
```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Get Route Safety
```
GET /route-safety
```
Parameters:
- `home_lat`: Starting point latitude
- `home_lon`: Starting point longitude
- `school_lat`: Destination latitude
- `school_lon`: Destination longitude

### 2. Get Model Metrics
```
GET /model-metrics
```
Returns current model performance metrics including:
- Cross-validation scores
- Feature importance
- Mean and standard deviation of CV scores

## Model Details

The safety prediction model uses a Random Forest Regressor with the following features:
- Crime count
- Lighting conditions
- Past reports
- Weather conditions
- Terrain type
- Mud risk

The model automatically retrains when new data is available and maintains versioning for tracking improvements.

## Error Handling

The application includes comprehensive error handling and logging:
- All API calls are wrapped in try-except blocks
- Errors are logged to `safety_tracker.log`
- HTTP exceptions are raised with appropriate status codes
- Fallback prediction mechanism when model is unavailable

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 
