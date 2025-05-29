# Car Price Predictor

A machine learning model for predicting car prices using scikit-learn.

## Setup
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the predictor:
```bash
python car_price_predictor.py
```

## Features
- Multiple ML models comparison
- Interactive price prediction
- Data visualization
- Model evaluation metrics

## Dataset Features
- Present_Price: Showroom price (in lakhs)
- Driven_kms: Number of kilometers driven
- Fuel_Type: Petrol/Diesel/CNG
- Selling_type: Individual/Dealer
- Transmission: Manual/Automatic
- Owner: First/Second/Third owner
- Year: Year of purchase

## Model Details
The model automatically selects the best performing algorithm from:
- Random Forest Regressor
- Gradient Boosting Regressor
- Linear Regression

![image](https://github.com/user-attachments/assets/1a852f17-d474-4b62-8407-a765d17fadfc)

## Sample Usage
```python
sample_car = {
    'Present_Price': 5.59,
    'Driven_kms': 27000,
    'Year': 2014,
    'Owner': 0,
    'Fuel_Type': 'Petrol',
    'Selling_type': 'Individual',
    'Transmission': 'Manual'
}
```
