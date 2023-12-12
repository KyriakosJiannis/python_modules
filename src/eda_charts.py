from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def dual_line_barplot(data: pd.DataFrame, target_var: str, feature_var: str, bins: int = None,
                      round_bins: int = None):
    """
    Create a dual chart (bar chart and line chart) for binary classification cases.
    The line chart represents the percentage of the target variable, and the bars represent a numerical feature.

    :param data: Imported DataFrame.
    :param target_var: Target variable.
    :param feature_var: Numerical feature.
    :param bins: Number of bins for numerical feature (default: 10).
    :param round_bins: Decimals for rounding.
    :return: Seaborn dual charts.
    """

    # Set the bin number, and create the grouped dataframe
    if pd.api.types.is_numeric_dtype(data[feature_var]):
        # Determine the number of bins based on the specified method or default to 10
        num_bins = bins if bins is not None and bins not in ["auto", "fd", "scott", "rice", "sturges", "doane",
                                                             "sqrt"] else 10

        # Create bins
        bins_group = pd.cut(data[feature_var], bins=num_bins, right=False)

        # Calculate counts and mean for each bin to be used in the chart
        grouped_data = data.groupby(bins_group)[[feature_var, target_var]].agg(
            {feature_var: 'count', target_var: 'mean'})
    else:
        # If the feature variable is not numerical, aggregate using the feature variable directly
        grouped_data = data.groupby(feature_var)[[feature_var, target_var]].agg(
            {feature_var: 'count', target_var: 'mean'})

    # Rename columns for clarity
    grouped_data.columns = [f'{feature_var} Counts', target_var]

    grouped_data.reset_index(inplace=True)  # Reset index

    # Convert feature variable to string for categorical display
    grouped_data[feature_var] = grouped_data[feature_var].astype(str)
    grouped_data[feature_var] = grouped_data[feature_var].str.replace(", ", "-")

    # Function to round numeric values within the bracketed string
    if pd.api.types.is_numeric_dtype(data[feature_var]) and round_bins is not None:
        def round_the_brackets(value, decimals):
            boundaries = [round(float(num), decimals) for num in value[1:-1].split('-')]
            formatted_boundaries = [int(num) if decimals == 0 else num for num in boundaries]
            return '[' + '-'.join(map(str, formatted_boundaries)) + ')'

        grouped_data[feature_var] = grouped_data[feature_var].apply(
            lambda x: round_the_brackets(x, decimals=round_bins))

    # Multiply the target variable by 100 for percentage display
    grouped_data[target_var] *= 100

    # Drop rows with missing values
    grouped_data.dropna(inplace=True)

    # Create dual chart
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot creation
    ax1.set_title(f'{feature_var} Distribution', fontsize=12)
    ax1.set_xlabel(f'{feature_var} Bins', fontsize=10)
    ax1.set_ylabel(f'{feature_var} Counts', fontsize=10)

    sns.barplot(x=feature_var, y=f'{feature_var} Counts', data=grouped_data, color='C0', ax=ax1)

    # Adjust tick sizes
    ax1.tick_params(axis='both', labelsize=8)

    # Specify that we want to share the same x-axis
    ax2 = ax1.twinx()

    # Line plot creation
    ax2.set_ylabel(f'{target_var} in %', fontsize=16)
    sns.lineplot(x=feature_var, y=target_var, data=grouped_data, sort=False, color="C1", lw=5, ax=ax2)

    # Adjust tick sizes
    ax2.tick_params(axis='y', labelsize=8)

    # Automatically adjust subplot parameters
    plt.tight_layout()

    # Show the plot
    plt.show()


def sub_hist_boxplot(data: pd.DataFrame, target_var: str, feature_var: str, stat="density", bins: int = None, ):
    """
    For classification cases, creates a subplot with a density chart and boxplot on target variable classes for a numerical feature.

    :param data: Pandas DataFrame.
    :param target_var: Target variable.
    :param feature_var: Numerical feature.
    :param bins:
    :param stat:
    :return: Subplot with Seaborn density and boxplot.
    """

    num_bins = bins if bins is not None and bins not in ["auto", "fd", "scott", "rice", "sturges", "doane",
                                                         "sqrt"] else 10

    # Create chart
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f'Distribution and Boxplot for {feature_var} by {target_var}', fontsize=16)
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    # Create the distribution plot
    ax1 = fig.add_subplot(2, 2, 1)
    sns.histplot(data=data, x=feature_var, stat=stat, hue=target_var, bins=num_bins, kde=True, common_norm=False,
                 element="step", ax=ax1)

    # Create the boxplot
    ax2 = fig.add_subplot(2, 2, 2)
    sns.boxplot(x=target_var, y=feature_var, data=data, palette='tab10', ax=ax2)

    # Automatically adjust subplot parameters
    plt.tight_layout()
    plt.show()


def tabular_plot(data: pd.DataFrame, target_var: str, feature_var: str):
    """
    Create a tabular chart for classification cases, showing the relationship
    between the target variable and an ordinal/categorical feature.

    :param data: The input DataFrame.
    :param target_var: The name of the target variable (categorical).
    :param feature_var: The name of the ordinal/categorical feature variable.

    :return: Displays a tabular chart using Seaborn catplot.
    """

    # Calculate percentages and reshape data
    cross_df = data.groupby(feature_var)[target_var].value_counts(normalize=True)
    cross_df = cross_df.mul(100).rename('Percent').reset_index()

    # Create the tabular chart using Seaborn catplot with a title
    chart = sns.catplot(x=feature_var, y='Percent', hue=target_var, kind='bar', data=cross_df)
    chart.fig.suptitle(f'Tabular Chart: {target_var} vs {feature_var}', y=1.02)

    # Set y-axis limit to ensure percentages are within the range [0, 100]
    chart.ax.set_ylim(0, 100)

    # Show the plot
    plt.show()


def distribution_by_group(data: pd.DataFrame, feature_var: str, by: str, stat="density", bins: int = None):
    """
    Create density plots for the classes of the 'by' variable for a feature.

    :param data: DataFrame containing the data.
    :param feature_var: The numerical feature variable for which density plots are to be created.
    :param by: The ordinal/categorical feature variable for grouping.
    :param bins:
    :param stat: seaborn stat parameter
    :return: Displays density plots using Seaborn .
    """

    num_bins = bins if bins is not None and bins not in ["auto", "fd", "scott", "rice", "sturges", "doane",
                                                         "sqrt"] else 10

    plt.figure(figsize=(12, 10))

    sns.histplot(data=data, x=feature_var, stat=stat, hue=by, common_norm=False, palette='tab10', bins=num_bins,
                 kde=True)

    plt.xlabel(feature_var, fontsize=12 * 0.8)
    plt.title(f"Distribution of {feature_var} by {by}", fontsize=12)

    plt.show()


def plot_distributions(data: pd.DataFrame, stat="count", common_norm=True, by=None, num_cols: int = None,
                       bins: int = None):
    """
    Count distributions for numerical, objects and categorical variables in a DataFrame.

    :param data: Pandas DataFrame.
    :param common_norm:
    :param stat:
    :param num_cols:
    :param bins:

    :param by:
    """
    num_bins = bins if bins is not None and bins not in ["auto", "fd", "scott", "rice", "sturges", "doane",
                                                         "sqrt"] else 10

    # Number of rows and columns for subplots
    if num_cols is None:
        num_cols = len(data.columns)
    num_rows = (num_cols - 1) // 2 + 1

    # Create subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))
    fig.suptitle('Distributions of Variables', fontsize=16)

    # Flatten the axes array for ease of indexing
    axes = axes.flatten()

    # Iterate through each column
    for i, col in enumerate(data.columns):
        data_type = data[col].dtype

        # Plot distributions for numerical variables
        if pd.api.types.is_numeric_dtype(data_type):
            if by is None:
                sns.histplot(data[col], kde=True, stat=stat, bins=num_bins, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
            else:
                if col != by:
                    sns.histplot(data, x=col, kde=True, stat=stat, hue=by, bins=num_bins, palette='tab10',
                                 common_norm=common_norm, ax=axes[i], )
                    axes[i].set_title(f'Distribution of {col}')

        # Plot counts for object (categorical) variables
        elif pd.api.types.is_categorical_dtype(data_type) or pd.api.types.is_object_dtype(data_type):
            if by is None:
                sns.countplot(data[col], stat=stat, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
            else:
                if col != by:
                    sns.countplot(data, x=col, stat=stat, hue=by, palette='tab10', ax=axes[i], )
                    axes[i].set_title(f'Distribution of {col}')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def descriptive_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a comprehensive descriptive DataFrame from a given DataFrame.
    includes data types, counts, NaN counts, unique values, top values, frequencies,
    statistical measures, quantiles, kurtosis, skew, and range for numerical variables.

    :param data: The input DataFrame.
    :returns: The comprehensive descriptive DataFrame w
    """
    # Initial data type, count, and null value calculations
    count_df = pd.DataFrame({
        'Dtype': data.dtypes,
        'Count': data.shape[0],
        'Non-Null Count': data.notnull().sum(),
        'NaN Count': data.isna().sum()
    })

    # Descriptive statistics
    describe_df = data.describe(include='all').T

    # Round numeric statistics to 3 decimals
    numeric_stats = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
    for col in numeric_stats:
        if col in describe_df.columns:
            describe_df[col] = describe_df[col].apply(lambda x: round(x, 2) if pd.notnull(x) else x)

    # Calculating additional statistics for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    describe_df.loc[numeric_cols, '5%'] = data[numeric_cols].quantile(0.05).round(3)
    describe_df.loc[numeric_cols, '95%'] = data[numeric_cols].quantile(0.95).round(3)
    describe_df.loc[numeric_cols, 'Range'] = (data[numeric_cols].max() - data[numeric_cols].min()).round(3)
    describe_df.loc[numeric_cols, 'Kurtosis'] = data[numeric_cols].kurtosis().round(3)
    describe_df.loc[numeric_cols, 'Skew'] = data[numeric_cols].skew().round(3)

    # Merging all DataFrames into one
    final_df = pd.concat([count_df, describe_df], axis=1)

    # Adjusting the order of columns
    final_columns = ['Dtype', 'Count', 'NaN Count', 'unique', 'top', 'freq',
                     'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'Range', 'Kurtosis', 'Skew']
    final_df = final_df[final_columns]

    return final_df


if __name__ == "__main__":
    pass
