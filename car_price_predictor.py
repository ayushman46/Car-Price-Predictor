# Car Price Prediction Machine Learning Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class CarPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.is_trained = False
        
    def load_and_explore_data(self, file_path):
        """Load and explore the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(file_path)
        
        print("\n=== DATASET OVERVIEW ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nColumns: {list(self.df.columns)}")
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        
        print(f"\n=== DATA TYPES ===")
        print(self.df.dtypes)
        
        print(f"\n=== MISSING VALUES ===")
        print(self.df.isnull().sum())
        
        print(f"\n=== STATISTICAL SUMMARY ===")
        print(self.df.describe())
        
        return self.df
    
    def visualize_data(self):
        """Create visualizations to understand the data"""
        plt.figure(figsize=(15, 12))
        
        # Price distribution
        plt.subplot(2, 3, 1)
        plt.hist(self.df['Selling_Price'], bins=30, edgecolor='black', alpha=0.7)
        plt.title('Distribution of Selling Price')
        plt.xlabel('Selling Price (in lakhs)')
        plt.ylabel('Frequency')
        
        # Year vs Price
        plt.subplot(2, 3, 2)
        plt.scatter(self.df['Year'], self.df['Selling_Price'], alpha=0.6)
        plt.title('Year vs Selling Price')
        plt.xlabel('Year')
        plt.ylabel('Selling Price (in lakhs)')
        
        # Driven_kms vs Price
        plt.subplot(2, 3, 3)
        plt.scatter(self.df['Driven_kms'], self.df['Selling_Price'], alpha=0.6)
        plt.title('Kilometers Driven vs Selling Price')
        plt.xlabel('Kilometers Driven')
        plt.ylabel('Selling Price (in lakhs)')
        
        # Fuel Type distribution
        plt.subplot(2, 3, 4)
        self.df['Fuel_Type'].value_counts().plot(kind='bar')
        plt.title('Fuel Type Distribution')
        plt.xticks(rotation=45)
        
        # Transmission distribution
        plt.subplot(2, 3, 5)
        self.df['Transmission'].value_counts().plot(kind='bar')
        plt.title('Transmission Type Distribution')
        plt.xticks(rotation=45)
        
        # Present Price vs Selling Price
        plt.subplot(2, 3, 6)
        plt.scatter(self.df['Present_Price'], self.df['Selling_Price'], alpha=0.6)
        plt.title('Present Price vs Selling Price')
        plt.xlabel('Present Price (in lakhs)')
        plt.ylabel('Selling Price (in lakhs)')
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        print("\n=== PREPROCESSING DATA ===")
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Feature Engineering
        # Calculate car age
        current_year = 2024
        df_processed['Car_Age'] = current_year - df_processed['Year']
        
        # Calculate depreciation rate
        df_processed['Depreciation_Rate'] = (df_processed['Present_Price'] - df_processed['Selling_Price']) / df_processed['Present_Price']
        
        # Handle categorical variables
        categorical_columns = ['Fuel_Type', 'Selling_type', 'Transmission']
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        
        # Select features for training
        feature_columns = [
            'Present_Price', 'Driven_kms', 'Car_Age', 'Owner',
            'Fuel_Type_encoded', 'Selling_type_encoded', 'Transmission_encoded'
        ]
        
        # Prepare features and target
        X = df_processed[feature_columns]
        y = df_processed['Selling_Price']
        
        self.feature_names = feature_columns
        
        print(f"Features selected: {feature_columns}")
        print(f"Target: Selling_Price")
        
        return X, y
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        print("\n=== TRAINING MODELS ===")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store test data for evaluation
        self.X_test = X_test_scaled
        self.y_test = y_test
        
        # Define models to try
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_score = -np.inf
        best_model = None
        best_model_name = None
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'R2': r2,
                'model': model
            }
            
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
            
            # Select best model based on R² score
            if r2 > best_score:
                best_score = r2
                best_model = model
                best_model_name = name
        
        self.model = best_model
        self.best_model_name = best_model_name
        self.results = results
        self.is_trained = True
        
        print(f"\n=== BEST MODEL: {best_model_name} ===")
        print(f"R² Score: {best_score:.4f}")
        
        return results
    
    def evaluate_model(self):
        """Evaluate the best model"""
        if not self.is_trained:
            print("Model not trained yet!")
            return
        
        print(f"\n=== MODEL EVALUATION ({self.best_model_name}) ===")
        
        # Make predictions
        if self.best_model_name == 'Linear Regression':
            y_pred = self.model.predict(self.X_test)
        else:
            # For tree-based models, we need to use original features
            X_test_original = pd.DataFrame(self.scaler.inverse_transform(self.X_test), 
                                         columns=self.feature_names)
            y_pred = self.model.predict(X_test_original)
        
        # Print detailed metrics
        for name, metrics in self.results.items():
            if name == self.best_model_name:
                print(f"Mean Absolute Error: {metrics['MAE']:.4f} lakhs")
                print(f"Root Mean Square Error: {metrics['RMSE']:.4f} lakhs")
                print(f"R² Score: {metrics['R2']:.4f}")
                break
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            print(f"\n=== FEATURE IMPORTANCE ===")
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(importance_df)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
    
    def predict_price(self, car_details):
        """Predict price for a single car"""
        if not self.is_trained:
            print("Model not trained yet!")
            return None
        
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([car_details])
            
            # Feature engineering
            current_year = 2024
            input_df['Car_Age'] = current_year - input_df['Year']
            
            # Encode categorical variables
            for col in ['Fuel_Type', 'Selling_type', 'Transmission']:
                if col in input_df.columns:
                    input_df[col + '_encoded'] = self.label_encoders[col].transform(input_df[col])
            
            # Select features
            input_features = input_df[self.feature_names]
            
            # Make prediction
            if self.best_model_name == 'Linear Regression':
                input_scaled = self.scaler.transform(input_features)
                prediction = self.model.predict(input_scaled)[0]
            else:
                prediction = self.model.predict(input_features)[0]
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def save_model(self, filename='car_price_model.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            print("No trained model to save!")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'best_model_name': self.best_model_name
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='car_price_model.pkl'):
        """Load a saved model"""
        try:
            model_data = joblib.load(filename)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.best_model_name = model_data['best_model_name']
            self.is_trained = True
            
            print(f"Model loaded successfully ({self.best_model_name})")
            
        except Exception as e:
            print(f"Error loading model: {e}")

# Example usage and testing
def main():
    """Main function to demonstrate the model"""
    
    predictor = CarPricePredictor()
    
    # Use the exact filename as it appears in the directory
    try:
        df = predictor.load_and_explore_data('car data.csv')  # Correct filename with space
        if df is None:
            print("Error: Could not load dataset. Please check if 'car data.csv' exists in the current directory.")
            return None
        
        # Visualize data
        predictor.visualize_data()
        
        # Preprocess data
        X, y = predictor.preprocess_data()
        
        # Train models
        results = predictor.train_models(X, y)
        
        # Evaluate the best model
        predictor.evaluate_model()
        
        # Save the model
        predictor.save_model()
        
        # Example prediction
        print("\n=== EXAMPLE PREDICTION ===")
        sample_car = {
            'Present_Price': 5.59,  # Current market price in lakhs
            'Driven_kms': 27000,    # Kilometers driven
            'Year': 2014,           # Manufacturing year
            'Owner': 0,             # Number of previous owners
            'Fuel_Type': 'Petrol',  # Fuel type
            'Selling_type': 'Individual',  # Selling type
            'Transmission': 'Manual'  # Transmission type
        }
        
        predicted_price = predictor.predict_price(sample_car)
        if predicted_price:
            print(f"Predicted selling price: ₹{predicted_price:.2f} lakhs")
        
        return predictor
        
    except FileNotFoundError:
        print("Error: 'car_data.csv' file not found!")
        print("Please make sure the dataset file is in the same directory as this script.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Interactive prediction function
def interactive_prediction(predictor):
    """Interactive function to get predictions from user input"""
    if not predictor or not predictor.is_trained:
        print("Model not available or not trained!")
        return
    
    print("\n=== INTERACTIVE CAR PRICE PREDICTION ===")
    print("Enter the following details about the car:")
    
    try:
        present_price = float(input("Present Price (in lakhs): "))
        driven_kms = int(input("Kilometers Driven: "))
        year = int(input("Manufacturing Year: "))
        owner = int(input("Number of Previous Owners: "))
        
        print("\nFuel Type options: Petrol, Diesel, CNG")
        fuel_type = input("Fuel Type: ").title()
        
        print("\nSelling Type options: Individual, Dealer")
        selling_type = input("Selling Type: ").title()
        
        print("\nTransmission options: Manual, Automatic")
        transmission = input("Transmission: ").title()
        
        car_details = {
            'Present_Price': present_price,
            'Driven_kms': driven_kms,
            'Year': year,
            'Owner': owner,
            'Fuel_Type': fuel_type,
            'Selling_type': selling_type,
            'Transmission': transmission
        }
        
        predicted_price = predictor.predict_price(car_details)
        
        if predicted_price:
            print(f"\n🚗 Predicted Selling Price: ₹{predicted_price:.2f} lakhs")
            print(f"💰 Estimated Price Range: ₹{predicted_price*0.9:.2f} - ₹{predicted_price*1.1:.2f} lakhs")
        else:
            print("Error in prediction. Please check your inputs.")
            
    except ValueError:
        print("Invalid input! Please enter numeric values where required.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Train the model
    predictor = main()
    
    # Run interactive prediction
    if predictor:
        while True:
            choice = input("\nWould you like to predict a car price? (y/n): ").lower()
            if choice == 'y':
                interactive_prediction(predictor)
            else:
                break
        
        print("Thank you for using the Car Price Predictor!")