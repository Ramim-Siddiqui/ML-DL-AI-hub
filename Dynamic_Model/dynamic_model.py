import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import numpy as np
import datetime

def get_dataset_path():
    while True:
        dataset_path = input("Please enter the path for the dataset (CSV or XLSX format): ")
        if not (dataset_path.endswith('.csv') or dataset_path.endswith('.xlsx')):
            print("The dataset file should be in CSV or XLSX format. Please try again.")
        else:
            return dataset_path

def load_dataset(dataset_path):
    try:
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path, encoding='utf-8')
        else:
            df = pd.read_excel(dataset_path, engine='openpyxl')
        return df
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(dataset_path, encoding='latin-1')
            return df
        except Exception as e:
            print(f"Failed to load the dataset. Error: {e}")
            return None
    except Exception as e:
        print(f"Failed to load the dataset. Error: {e}")
        return None

def validate_dataset_columns(df):
    if df.shape[1] > 15:
        raise ValueError("The dataset should have 15 columns or fewer. Please check your dataset and try again.")
    return df

def get_target_column(df):
    while True:
        target_column = input("Please enter the column name you want to predict: ")
        if target_column not in df.columns:
            print(f"The column '{target_column}' is not in the dataset. Please try again.")
        else:
            return target_column

def get_columns_to_discard(df):
    while True:
        columns_to_discard = input("Please enter the columns to discard (comma-separated) or type 'No': ")
        if columns_to_discard.lower() == 'no':
            return []
        else:
            columns_to_discard = [col.strip() for col in columns_to_discard.split(',')]
            invalid_columns = [col for col in columns_to_discard if col not in df.columns]
            if invalid_columns:
                print(f"The columns '{', '.join(invalid_columns)}' are not in the dataset. Please try again.")
            else:
                return columns_to_discard

def get_date_column(df):
    while True:
        date_column = input("Please enter the column name that contains dates or type 'no' if there is no date column: ")
        if date_column.lower() == 'no':
            return None
        elif date_column not in df.columns:
            print(f"The column '{date_column}' is not in the dataset. Please try again.")
        else:
            return date_column

def extract_date_components(df, date_column):
    df[date_column] = df[date_column].astype(str)
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    if df[date_column].isnull().all():
        raise ValueError(f"The column '{date_column}' does not contain valid date values.")
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df = df.drop(columns=[date_column])
    return df

def label_encode_columns(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object', 'string']).columns:
        # Convert the column to string type to avoid mixed types
        df[column] = df[column].astype(str)
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    return df, label_encoders


def train_models(df, target_column):
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'Elastic Net': ElasticNet(),
        'Decision Tree Regression': DecisionTreeRegressor(),
        'Random Forest Regressor': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'XGBoost Regression': XGBRegressor(),
        'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        'K-nearest Neighbors (KNN)': KNeighborsRegressor(),
        'Extra Trees Regression': ExtraTreesRegressor(),
        'Bayesian Ridge Regression': BayesianRidge(),
        'ARIMA': sm.tsa.ARIMA(y_train, order=(5, 1, 0)),
        'LSTM': Sequential(),
        'ETS': sm.tsa.ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=12)
    }

    r2_scores = {}

    for name, model in models.items():
        try:
            if name == 'ARIMA':
                arima_model = model.fit()
                arima_pred = arima_model.forecast(steps=len(y_test))
                r2_scores[name] = r2_score(y_test, arima_pred)
            elif name == 'LSTM':
                X_train_lstm = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
                X_test_lstm = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

                model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
                model.add(LSTM(units=50, return_sequences=False))
                model.add(Dense(units=1))
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
                model.fit(X_train_lstm, y_train, epochs=10, batch_size=32)
                lstm_pred = model.predict(X_test_lstm)
                r2_scores[name] = r2_score(y_test, lstm_pred)
            elif name == 'ETS':
                ets_model = model.fit()
                ets_pred = ets_model.forecast(len(y_test))
                r2_scores[name] = r2_score(y_test, ets_pred)
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                r2_scores[name] = r2_score(y_test, predictions)
            print(f"{name} model R² score: {r2_scores[name]}")
        except Exception as e:
            print(f"{name} model cannot be applied: {e}")

    # Save R² scores to CSV
    r2_scores_df = pd.DataFrame(list(r2_scores.items()), columns=['Model', 'R2 Score'])
    r2_scores_df.to_csv('r2scores.csv', index=False)
    print("R² scores have been saved to 'r2scores.csv'.")
    
    best_model_name = max(r2_scores, key=r2_scores.get)
    best_model = models[best_model_name]
    return best_model_name, best_model, r2_scores, X.columns

def get_prediction_type():
    while True:
        prediction_type = input("How would you like to do predictions? (daily/monthly): ").strip().lower()
        if prediction_type in ['daily', 'monthly']:
            return prediction_type
        else:
            print("Invalid input. Please enter 'daily' or 'monthly'.")

def get_daily_prediction_params():
    while True:
        try:
            start_date = input("Enter the start date (YYYY-MM-DD): ")
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
            num_days = int(input("Enter the number of days for prediction (1-30): "))
            if num_days < 1 or num_days > 30:
                print("Number of days should be between 1 and 30. Please try again.")
            else:
                return start_date, num_days
        except ValueError:
            print("Invalid input. Please enter a valid date and number of days.")

def get_monthly_prediction_params():
    while True:
        try:
            start_month = input("Enter the start month (YYYY-MM): ")
            start_month = datetime.datetime.strptime(start_month, '%Y-%m')
            return start_month
        except ValueError:
            print("Invalid input. Please enter a valid month (YYYY-MM).")

def predict_with_best_model(df, best_model_name, best_model, target_column, prediction_type, feature_columns, label_encoders):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if prediction_type == 'daily':
        start_date, num_days = get_daily_prediction_params()
        prediction_dates = [start_date + datetime.timedelta(days=i) for i in range(num_days)]
        # Create a DataFrame with all unique values combinations and prediction dates
        unique_combinations = df[feature_columns].drop_duplicates()
        future_df = pd.DataFrame({
            'ds': np.repeat(prediction_dates, len(unique_combinations))
        })
    
        unique_combinations_repeated = pd.concat([unique_combinations] * len(prediction_dates), ignore_index=True)
        future_df = pd.concat([future_df, unique_combinations_repeated], axis=1)
    
        if 'ds' in future_df.columns:
            future_df = extract_date_components(future_df, 'ds')
        future_df = future_df[feature_columns]

        if best_model_name == 'ARIMA':
            arima_model = best_model.fit()
            predictions = arima_model.forecast(steps=len(future_df))
        elif best_model_name == 'LSTM':
            X_future = np.reshape(future_df.values, (future_df.shape[0], 1, future_df.shape[1]))
            predictions = best_model.predict(X_future)
        elif best_model_name == 'ETS':
            ets_model = best_model.fit()
            predictions = ets_model.forecast(steps=len(future_df))
        else:
            predictions = best_model.predict(future_df)

        results_df = pd.DataFrame({
            'Date': np.repeat(prediction_dates, len(unique_combinations))
        })
        results_df[feature_columns] = future_df[feature_columns]
        results_df['Predictions'] = predictions.flatten()

        # Inverse label encoding
        for col, le in label_encoders.items():
            if col in results_df.columns:
                results_df[col] = le.inverse_transform(results_df[col].astype(int))
        results_df = results_df.drop_duplicates()
        print("Predictions:")
        print(results_df)

        # Save predictions to CSV file
        results_df.to_csv('predictions.csv', index=False)
        print("Predictions have been saved to 'predictions.csv'.")
        
    elif prediction_type == 'monthly':
        start_month = get_monthly_prediction_params()
        prediction_dates = [start_month + datetime.timedelta(days=i*30) for i in range(13)]
        
        # Ask user if they want combinations or months
        choice = input("Do you want to do predictions for 'combinations' or 'months'? ").strip().lower()
        
        if choice == 'months':
            # Perform predictions
            future_df = pd.DataFrame({
                'ds': np.repeat(prediction_dates, len(df[feature_columns].drop_duplicates()))
            })
            unique_combinations = df[feature_columns].drop_duplicates()
            unique_combinations_repeated = pd.concat([unique_combinations] * len(prediction_dates), ignore_index=True)
            future_df = pd.concat([future_df, unique_combinations_repeated], axis=1)
            
            if 'ds' in future_df.columns:
                future_df = extract_date_components(future_df, 'ds')
            future_df = future_df[feature_columns]

            # Make predictions
            if best_model_name == 'ARIMA':
                arima_model = best_model.fit()
                predictions = arima_model.forecast(steps=len(future_df))
            elif best_model_name == 'LSTM':
                X_future = np.reshape(future_df.values, (future_df.shape[0], 1, future_df.shape[1]))
                predictions = best_model.predict(X_future)
            elif best_model_name == 'ETS':
                ets_model = best_model.fit()
                predictions = ets_model.forecast(steps=len(future_df))
            else:
                predictions = best_model.predict(future_df)

            results_df = pd.DataFrame({
                'Date': np.repeat(prediction_dates, len(df[feature_columns].drop_duplicates()))
            })
            results_df[feature_columns] = future_df[feature_columns]
            results_df['Predictions'] = predictions.flatten()

            # Inverse label encoding
            for col, le in label_encoders.items():
                if col in results_df.columns:
                    results_df[col] = le.inverse_transform(results_df[col].astype(int))
            
            results_df = results_df.drop_duplicates()
            results_df = results_df[['Date', 'Predictions']]
            results_df['Month'] = pd.to_datetime(results_df['Date']).dt.to_period('M')
            monthly_summary = results_df.groupby('Month').agg({'Predictions': 'mean'}).reset_index()
            
            print("Monthly Predictions:")
            print(monthly_summary)

            # Save predictions to CSV file
            monthly_summary.to_csv('monthly_predictions.csv', index=False)
            print("Monthly predictions have been saved to 'monthly_predictions.csv'.")
        
        elif choice == 'combinations':
            # Existing code for 'combinations' prediction
            future_df = pd.DataFrame({
                'ds': np.repeat(prediction_dates, len(df[feature_columns].drop_duplicates()))
            })
            unique_combinations = df[feature_columns].drop_duplicates()
            unique_combinations_repeated = pd.concat([unique_combinations] * len(prediction_dates), ignore_index=True)
            future_df = pd.concat([future_df, unique_combinations_repeated], axis=1)
            
            if 'ds' in future_df.columns:
                future_df = extract_date_components(future_df, 'ds')
            future_df = future_df[feature_columns]

            # Make predictions
            if best_model_name == 'ARIMA':
                arima_model = best_model.fit()
                predictions = arima_model.forecast(steps=len(future_df))
            elif best_model_name == 'LSTM':
                X_future = np.reshape(future_df.values, (future_df.shape[0], 1, future_df.shape[1]))
                predictions = best_model.predict(X_future)
            elif best_model_name == 'ETS':
                ets_model = best_model.fit()
                predictions = ets_model.forecast(steps=len(future_df))
            else:
                predictions = best_model.predict(future_df)

            results_df = pd.DataFrame({
                'Date': np.repeat(prediction_dates, len(df[feature_columns].drop_duplicates()))
            })
            results_df[feature_columns] = future_df[feature_columns]
            results_df['Predictions'] = predictions.flatten()

            # Inverse label encoding
            for col, le in label_encoders.items():
                if col in results_df.columns:
                    results_df[col] = le.inverse_transform(results_df[col].astype(int))
            results_df = results_df.drop_duplicates()
            print("Combinations Predictions:")
            print(results_df)

            # Save predictions to CSV file
            results_df.to_csv('predictions.csv', index=False)
            print("Predictions have been saved to 'predictions.csv'.")
    
    else:
        raise ValueError("Invalid prediction_type. Choose 'daily' or 'monthly'.")

def main():
    while True:
        try:
            # Step 1: Get dataset path
            dataset_path = get_dataset_path()
            
            # Step 2: Load dataset
            df = load_dataset(dataset_path)
            if df is None:
                continue
            else:
                try:
                    df = validate_dataset_columns(df)
                    print(f"Dataset loaded successfully with {df.shape[1]} columns.")
                except ValueError as e:
                    print(e)
                    continue
            
            # Step 3: Validate dataset columns
            df = validate_dataset_columns(df)
            
            # Step 4: Get target column
            target_column = get_target_column(df)
            
            # Check if target column needs label encoding
            if df[target_column].dtype == 'object':
                le = LabelEncoder()
                df[target_column] = le.fit_transform(df[target_column])
                print(f"Label encoded target column '{target_column}'.")
            
            # Step 5: Get date column
            date_column = get_date_column(df)
            
            # Step 6: Extract date components if date column is provided
            if date_column:
                df = extract_date_components(df, date_column)
            
            # Step 7: Ask the user if they want to drop columns
            columns_to_discard = get_columns_to_discard(df)
            df = df.drop(columns=columns_to_discard)
            
            # Step 10: Label encode string/object columns
            df, label_encoders = label_encode_columns(df)

            # Step 8: Display correlation of target column with all columns
            correlation_matrix = df.corr()
            correlation_with_target = correlation_matrix[target_column].sort_values(ascending=False)
            print("Correlation of all columns with the target column:")
            print(correlation_with_target)
            
            # Step 9: Display dataset and ask if user wants to drop any more columns
            print("Here is a preview of the dataset:")
            print(df.head())
            while True:
                continue_with_dataset = input("Do you want to drop any more columns? (yes/no): ").strip().lower()
                if continue_with_dataset == 'yes':
                    columns_to_discard = get_columns_to_discard(df)
                    df = df.drop(columns=columns_to_discard)
                    print("Updated dataset preview:")
                    print(df.head())
                elif continue_with_dataset == 'no':
                    break
                else:
                    print("Please answer 'yes' or 'no'.")
            
           # Drop rows with zero, negative, or NaN values in the target column
            df = df[df[target_column] >= 0].dropna(subset=[target_column])
            
            # Drop rows with 0, negative values, or NaN in any column
            df = df[(df >= 0).all(axis=1)].dropna()
            # # Drop rows with NaN values in the target column
            # df = df.dropna(subset=[target_column])

            # # Drop rows with NaN values in any column
            # df = df.dropna()


            # Step 11: Train models
            best_model_name, best_model, r2_scores, feature_columns = train_models(df, target_column)
            print(f"Best model: {best_model_name} with R² score: {r2_scores[best_model_name]}")
            
            # Step 12: Get prediction type
            prediction_type = get_prediction_type()
            
            # Step 13: Predict with best model
            predict_with_best_model(df, best_model_name, best_model, target_column, prediction_type, feature_columns, label_encoders)
            
            break
            
        except ValueError as e:
            print(f"Error: {e}. Please try again.")

if __name__ == "__main__":
    main()
