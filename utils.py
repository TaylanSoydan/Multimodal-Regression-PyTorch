import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

def root_mean_squared_error(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

# Function to check if a column consists only of -5 and -2 (boolean sense)
def is_boolean_column_with_only_special_values(column, epsilon):
    """
    Check if a column contains only -5 and -2 (with epsilon margin).
    
    Parameters:
        column (pd.Series): The column to check.
        epsilon (float): Allowed margin for comparison.
    
    Returns:
        bool: True if the column contains only -5 and -2 within the epsilon margin.
    """
    unique_values = column.dropna().unique()
    return set(unique_values).issubset({-5, -2, -5 + epsilon, -5 - epsilon, -2 + epsilon, -2 - epsilon})

# Function to replace -5 and -2 with NaN, with an option to keep boolean-like columns
def replace_special_values(df, epsilon=0.001, if_keep_bool=True):
    """
    Replace -5 and -2 with NaN in the dataframe, with the option to keep boolean-like columns.
    
    Parameters:
        df (pd.DataFrame): Input dataframe.
        epsilon (float): Margin for floating-point comparison.
        if_keep_bool (bool): If True, keep columns with only -5 and -2 unchanged.
    
    Returns:
        pd.DataFrame: Dataframe with -5 and -2 replaced by NaN where applicable.
    """
    columns = list(df.columns)

    if "description" in columns:
        columns.remove("description")

    for col in columns:
        if is_boolean_column_with_only_special_values(df[col], epsilon):
            if not if_keep_bool:
                df[col] = df[col].mask((df[col] >= -5 - epsilon) & (df[col] <= -5 + epsilon), np.nan)
                df[col] = df[col].mask((df[col] >= -2 - epsilon) & (df[col] <= -2 + epsilon), np.nan)
        else:
            df[col] = df[col].mask((df[col] >= -5 - epsilon) & (df[col] <= -5 + epsilon), np.nan)
            df[col] = df[col].mask((df[col] >= -2 - epsilon) & (df[col] <= -2 + epsilon), np.nan)
    return df



def classify_columns(df):
    """
    Classify columns in the dataframe as either 'Numerical' or 'Categorical' based on 
    the number of unique values and whether the column contains integer-like values.
    
    The function uses the following logic:
    1. If the column has more than 50 unique values, it is classified as 'Numerical'.
    2. If the column contains numeric values and all of them are integer-like (e.g., 
       a float column where all values are whole numbers), it is classified as 'Categorical'.
    3. Otherwise, the column is classified as 'Numerical'.
    
    Parameters:
        df (pd.DataFrame): The input dataframe.
    
    Returns:
        dict: A dictionary where keys are column names and values are the classification 
              ('Numerical' or 'Categorical').
    """
    column_types = {}
    for col in df.columns:
        unique_values = df[col].nunique()
        # If unique values are more than 50, classify as numerical
        if unique_values > 50:
            column_types[col] = 'Numerical'
        # Check if the column contains integer-like floats (i.e., all values are integer-like)
        elif pd.api.types.is_numeric_dtype(df[col]) and all(df[col].dropna() == df[col].dropna().astype(int)):
            column_types[col] = 'Categorical'
        else:
            column_types[col] = 'Numerical'
    return column_types

def impute_data(df, num_cols, cat_cols):
    """
    Impute missing values in numerical and categorical columns using median for numerical
    columns and most frequent for categorical columns.
    
    Parameters:
        df (pd.DataFrame): The input dataframe with missing values.
        num_cols (list): List of numerical columns to impute.
        cat_cols (list): List of categorical columns to impute.
                          
    Returns:
        pd.DataFrame: The dataframe with imputed values.
    """
    num_cols.append("target") if "target" not in num_cols else None
    
    # Impute numerical columns using median
    num_imputer = SimpleImputer(strategy='median')
    df_num_imputed = pd.DataFrame(num_imputer.fit_transform(df[num_cols]), columns=num_cols)
    
    # Impute categorical columns using most frequent
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df_cat_imputed = pd.DataFrame(cat_imputer.fit_transform(df[cat_cols]), columns=cat_cols)
    
    # Combine numerical and categorical dataframes
    df_imputed = pd.concat([df_num_imputed, df_cat_imputed], axis=1)
    
    return df_imputed

def visualize_special_values(df, sample_size=100):
    """
    Visualizes the -5, -2, and missing values in the dataframe.
    - Red for -5
    - Orange for -2
    - Blue for missing values

    Args:
    df (DataFrame): The input dataframe.
    sample_size (int): Number of rows to sample for visualization. Default is 100.

    Returns:
    None: Displays the plot.
    """
    # Sample rows from the dataframe
    sampled_data = df.sample(sample_size)

    # Create masks for -5, -2, and missing values
    mask_neg_5 = (sampled_data == -5).astype(int)
    mask_neg_2 = (sampled_data == -2).astype(int)
    mask_missing = sampled_data.isna().astype(int)

    # Create a numeric matrix to represent the color codes
    numeric_color_matrix = np.zeros(sampled_data.shape)

    # Assign numeric codes: 1 for red (-5), 2 for orange (-2), 3 for blue (missing)
    numeric_color_matrix[mask_neg_5 == 1] = 1
    numeric_color_matrix[mask_neg_2 == 1] = 2
    numeric_color_matrix[mask_missing == 1] = 3

    # Define a colormap with white for valid values, red for -5, orange for -2, and blue for missing
    cmap = ListedColormap(["white", "red", "orange", "blue"])

    # Plot the matrix
    plt.figure(figsize=(12, 8))
    plt.imshow(numeric_color_matrix, aspect="auto", cmap=cmap, interpolation="none")

    # Add labels and title
    plt.title('Visualization of -5 (Red), -2 (Orange), and Missing (Blue) Values in Sampled Data')
    plt.xlabel('Columns')
    plt.ylabel('Rows')

    # Show plot
    plt.show()

def compute_grad_norm_and_weights(model, track_biases=False, track_weights=False):
    grad_norms = {}
    weight_values = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not track_biases and "bias" in name:
                continue  # Skip biases if track_biases is False
            
            grad_norm = round(param.grad.norm(2).item(), 4)  # L2 norm of gradients
            grad_norms[name] = grad_norm
            
            if track_weights:  # If we want to track weight values as well
                weight_values[name] = round(param.data.norm(2).item(), 4)  # L2 norm of weights

    return grad_norms, weight_values

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



