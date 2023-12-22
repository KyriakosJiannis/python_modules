import pandas as pd
import numpy as np
from scipy import stats


def compare_distributions(data: pd.DataFrame, numerical_col: str, category_col: str):
    """
    Compare the distribution of a numerical variable across categories using KS test.

    Args:
        data: DataFrame containing the data.
        numerical_col: Name of the numerical variable column.
        category_col: Name of the categorical variable columns.

    Returns:
        DataFrame with category pairs and their corresponding p-values.
    """
    category_pairs = data[category_col].unique()
    results = []

    for i in range(len(category_pairs)):
        for j in range(i + 1, len(category_pairs)):
            category1 = category_pairs[i]
            category2 = category_pairs[j]

            data1 = data[data[category_col] == category1][numerical_col]
            data2 = data[data[category_col] == category2][numerical_col]

            _, p_value = stats.ks_2samp(data1, data2)
            formatted_p_value = f'{p_value:.3f}' if p_value > 0.001 else '0.000'
            results.append([category1, category2, formatted_p_value])

    result_df = pd.DataFrame(results, columns=['Var1', 'Var2', 'Results'])
    return result_df


def feature_analysis_with_tests(data, dependent_var):
    """
    Perform feature analysis including
    variance and Pearson correlation test
    and return the results as a DataFrame.

    Args:
        data:
        dependent_var:

    Returns:
        DataFrame: A DataFrame containing features with their correlations,
        variances, and Pearson correlation coefficients.
    """

    # Variance Analysis
    variances = data.var().reset_index()
    variances.columns = ['Feature', 'Variance']

    # Pearson Correlation Analysis
    pearson_results = []
    for feature in data.columns:
        if data[feature].dtype in [np.int64, np.float64] and feature != dependent_var:
            # Drop NaNs from the pair of features being analyzed
            valid_data = data[[feature, dependent_var]].dropna()
            # Only calculate if there's enough data remaining
            if len(valid_data) > 1:
                corr, _ = stats.pearsonr(valid_data[feature], valid_data[dependent_var])
                correlation_type = 'Positive' if corr > 0 else 'Negative'
                pearson_results.append({'Feature': feature, 'Pearson': corr, 'Correlation': correlation_type})

    pearson_df = pd.DataFrame(pearson_results)

    # Merging the variance and Pearson correlation results
    combined_df = pd.merge(variances, pearson_df, on='Feature')

    # Sorting within each correlation group
    combined_df = combined_df.sort_values(by=['Correlation', 'Pearson'], ascending=[True, False])

    return combined_df


if __name__ == "__main__":
    pass
