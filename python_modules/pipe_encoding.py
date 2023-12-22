import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler


def preprocessor(data: pd.DataFrame, numerical_strategy='mean', categorical_strategy='most_frequent',
                 scaling_method='standard'):
    """
    Preprocess a dataframe with mixed data types.

    :param data: Input dataframe. Expected to be a Pandas DataFrame with numerical and/or categorical columns.
    :param numerical_strategy: Strategy for imputing missing values in numerical columns.
            Options: 'mean', 'median', 'most_frequent', or a specific value.
    :param categorical_strategy:  Strategy for imputing missing values in categorical columns.
            Options: 'most_frequent', 'constant', or a specific value
    :param scaling_method: Method for scaling numerical columns.
            Options: 'standard', 'minmax', 'robust', or None (no scaling)

    :returns:  Preprocessed dataframe with all variables encoded.

    :raises ValueError: If an invalid strategy or method is provided.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a Pandas DataFrame.")

    numerical_columns = data.select_dtypes(include=['number']).columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns

    # Validate strategies and methods
    valid_num_strategies = {'mean', 'median', 'most_frequent'}
    valid_cat_strategies = {'most_frequent', 'constant'}
    valid_scaling_methods = {'standard', 'minmax', 'robust', None}

    if numerical_strategy not in valid_num_strategies and not isinstance(numerical_strategy, (int, float)):
        raise ValueError(f"Invalid numerical_strategy: {numerical_strategy}")
    if categorical_strategy not in valid_cat_strategies and not isinstance(categorical_strategy, str):
        raise ValueError(f"Invalid categorical_strategy: {categorical_strategy}")
    if scaling_method not in valid_scaling_methods:
        raise ValueError(f"Invalid scaling_method: {scaling_method}")

    # Define preprocessing steps for numerical columns
    numerical_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numerical_strategy))
    ])

    if scaling_method == 'standard':
        numerical_preprocessor.steps.append(('scaler', StandardScaler()))
    elif scaling_method == 'minmax':
        numerical_preprocessor.steps.append(('scaler', MinMaxScaler()))
    elif scaling_method == 'robust':
        numerical_preprocessor.steps.append(('scaler', RobustScaler()))

    # Define preprocessing steps for categorical columns
    categorical_preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_strategy)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine numerical and categorical preprocessing using ColumnTransformer
    pre_processor = ColumnTransformer(transformers=[
        ('numerical', numerical_preprocessor, numerical_columns),
        ('categorical', categorical_preprocessor, categorical_columns)
    ])

    # Fit and transform the preprocessor on the data
    output = pre_processor.fit_transform(data)

    # Generate new column names
    encoded_columns = (list(numerical_columns) +
                       list(pre_processor.named_transformers_['categorical']
                            .named_steps['onehot']
                            .get_feature_names_out(categorical_columns)))

    output_df = pd.DataFrame(output, columns=encoded_columns)

    return output_df


if __name__ == '__main__':
    pass
