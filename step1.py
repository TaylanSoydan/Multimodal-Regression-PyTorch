import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.model_selection import train_test_split
from utils import *

file = "candidates_data.csv"

replace_5_2 = True
keep_bool = False
missing_threshold = 0.3

df = pd.read_csv(file)
# Dropping description as instructed
df = df.drop(columns="description",axis=1)
# Converting feature_1 and 2 to integers  
df["feature_1"] = df["feature_1"].str.split("_").str[1].astype(int)
df["feature_2"] = df["feature_2"].str.split("_").str[1].astype(int)

if replace_5_2:
    df = replace_special_values(df, epsilon=0.001, if_keep_bool=False)

# Drop missing features
print(f"Dropping columns with more than {missing_threshold*100}% missing values")
df = df.drop(columns=df.columns[df.isnull().mean() > missing_threshold])
column_classifications = classify_columns(df)
cat_cols = [k for k,v in column_classifications.items() if v == "Categorical"]
num_cols = [k for k,v in column_classifications.items() if (v == "Numerical") and (k!="target")]

df[cat_cols] = df[cat_cols].astype(int)

# Step 1: Prepare the data
X = df.drop(columns=['target'])
y = df['target']

# Split the data into train, validation, and test sets
# 70 / 15 / 15 split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42)


# Selected with RFE (see notebook)
selected_features = [0, 2, 3, 5, 6, 7, 8, 11, 12, 16, 17, 18, 19, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 37] #best_features
selected_cat_features = [df.columns[col] for col in selected_features if df.columns[col] in cat_cols]
X_train_val_selected = X_train_val.iloc[:, selected_features]
X_test_selected = X_test.iloc[:, selected_features]
print("Training CatBoostRegressor")
# RandomSearch did not improve the default model performance
best_model = CatBoostRegressor(random_seed=42, cat_features=selected_cat_features, verbose=False)
best_model.fit(X_train_val_selected, y_train_val)

# Evaluate on test set
y_pred = best_model.predict(X_test_selected)

# Clip predictions, training data suggests target can't be smaller than 1
y_pred = np.clip(y_pred, 1, np.inf)

# Calculate RMSE and R²
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nTest RMSE = {rmse:.4f}')
print(f'Test R² = {r2:.4f}')