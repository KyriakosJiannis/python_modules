import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def preprocessor(data: pd.DataFrame, numerical_strategy='mean', categorical_strategy='most_frequent',
                 scaling_method='standard', unknown_category='ignore', custom_constant=None):
    """
    Preprocess a dataframe with mixed data types (Numerical and Object/categorical).

    :param data: Input dataframe. Expected to be a Pandas DataFrame with numerical and/or categorical columns.
    :param numerical_strategy: Strategy for imputing missing values in numerical columns.
            Options: 'mean', 'median', 'most_frequent', or a specific value.
    :param categorical_strategy: Strategy for imputing missing values in categorical columns.
            Options: 'most_frequent', 'constant', or a specific value.
            If 'constant' is chosen, the 'custom_constant' parameter must be set.
    :param scaling_method: Method for scaling numerical columns.
            Options: 'standard', 'minmax', 'robust', or None (no scaling).
    :param unknown_category: Specifies how to handle unknown categories during one-hot encoding.
            Options: 'ignore', 'error'. If 'ignore', unknown categories will be ignored during
            transformation. If 'error', an error will be raised if unknown categories are encountered.
    :param custom_constant: The constant value used for imputing missing values in categorical columns
            when categorical_strategy is set to 'constant'. This value is used to fill in missing values
            in categorical columns. It must be specified if 'constant' is chosen as the categorical_strategy.
            This parameter is ignored if categorical_strategy is not 'constant'.

    :returns: Preprocessed dataframe with all variables encoded. .

    :raises ValueError: If an invalid strategy or method is provided, or if 'constant' is selected
                        for categorical_strategy without specifying 'custom_constant'.
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
        ('imputer', SimpleImputer(strategy=categorical_strategy, fill_value=custom_constant)),
        ('onehot', OneHotEncoder(handle_unknown=unknown_category))
    ])

    # Combine numerical and categorical preprocessing using ColumnTransformer
    pre_processor = ColumnTransformer(transformers=[
        ('numerical', numerical_preprocessor, numerical_columns),
        ('categorical', categorical_preprocessor, categorical_columns)
    ])

    # Fit and transform the preprocessor on the data
    output = pre_processor.fit_transform(data)

    # Generate new column names for the output DataFrame
    encoded_columns = (list(numerical_columns) +
                       list(pre_processor.named_transformers_['categorical']
                            .named_steps['onehot']
                            .get_feature_names_out(categorical_columns)))

    # Construct the output DataFrame with appropriate column names
    output_df = pd.DataFrame(output, columns=encoded_columns, index=data.index)

    return output_df


def simplified_fit_impute(train_data: pd.DataFrame, test_data: pd.DataFrame = None, skipCols: list = None):
    """

    Apply the imputation of missing data using the MICE (Multiple Imputation by Chained Equations) algorithm
    to both training and optionally to test data. This function is designed to simplify the process of
    imputing missing values in a dataset with numerical features only.

    :param train_data: The training DataFrame with missing values.
    :param test_data:  The test DataFrame with missing values. Default is None.
    :param skipCols: optional: List of column names to be excluded from imputation.
    :returns: (pd.DataFrame, pd.DataFrame): train and test imputed dataframe

    """
    train_data_copy = train_data.copy()
    original_train_index = train_data_copy.index  # Store the original index for the training data

    if test_data is not None:
        test_data_copy = test_data.copy()
        original_test_index = test_data_copy.index  # Store the original index for the test data

    # todo with skip columns

    # Remove specified columns from both datasets
    if skipCols is not None:
        train_data_copy.drop(columns=skipCols, inplace=True)
        if test_data is not None:
            test_data_copy.drop(columns=skipCols, inplace=True)

    # Initialize the imputer with specified maximum iterations and random state
    imputer = IterativeImputer(max_iter=10, random_state=0)

    # Fit the imputer on the training data and transform it
    imputer.fit(train_data_copy)
    imputed_train_data = imputer.transform(train_data_copy)
    imputed_train_df = pd.DataFrame(imputed_train_data, columns=train_data_copy.columns, index=original_train_index)

    imputed_test_df = None
    if test_data is not None:
        # Transform the test data using the fitted imputer
        imputed_test_data = imputer.transform(test_data_copy)
        imputed_test_df = pd.DataFrame(imputed_test_data, columns=test_data_copy.columns, index=original_test_index)

    return imputed_train_df, imputed_test_df


if __name__ == '__main__':
    # Example usage
    pd.set_option('display.max_columns', None)

    df = pd.DataFrame({
        'Age': [25, 30, 35, np.nan],
        'Salary': [50000, 60000, 65000, 58000],
        'Gender': ['Male', 'Female', np.nan, 'Male'],
        'Department': ['IT', 'HR', 'Finance', 'IT']
    })

    # Preprocess the DataFrame
    processed_df = preprocessor(
        data=df,
        numerical_strategy='median',
        categorical_strategy='constant',
        scaling_method='standard',
        unknown_category='ignore',
        custom_constant='Unknown'  #
    )
    df_encoded = pd.get_dummies(df, columns=['Gender', 'Department'], dummy_na=False)
    imputed_df = simplified_fit_impute(df_encoded)

    print("Original DataFrame:")
    print(df)
    print("\nProcessed DataFrame:")
    print(processed_df)
    print("\nImputed DataFrame:")
    print(imputed_df)
