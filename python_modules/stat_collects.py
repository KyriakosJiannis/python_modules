import pandas as pd
from scipy import stats


def compare_distributions(data, numerical_col, category_col):
    """
    Compare the distribution of a numerical variable across categories using KS test.

    Parameters:
        - data: DataFrame containing the data.
        - numerical_col: Name of the numerical variable column.
        - category_col: Name of the categorical variable column.

    Returns:
        - DataFrame with category pairs and their corresponding p-values.
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


if __name__ == "__main__":
    pass
