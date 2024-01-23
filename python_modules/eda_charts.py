import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns


def dual_line_barplot(data: pd.DataFrame, target_var: str, feature_var: str, bins: int = None,
                      round_bins: int = None) -> Figure:
    """
    Generates a dual-axis chart combining a bar chart and a line chart, specifically designed
    for visualizing the distribution and impact of a feature variable on a binary target variable
    in classification problems.
    The target variable should be binary (0,1).
    The bar chart displays the count of observations for each bin or category of the feature variable, while the line
    chart shows the percentage of one class of the target variable within each bin or category.

    :param data: The DataFrame containing the dataset to be visualized.
    :param target_var: The name of the binary target variable.  Must contain only 0 and 1
    :param feature_var: The name of the feature variable to analyze. Can be numeric or object.
    :param bins: Optional. The number of bins to divide the numeric feature variable into. If `None`,
                defaults to 10. Ignored for categorical/object variables.
    :param round_bins: Optional. The number of decimal places to round the edges of the bins to.
                Only applicable if `bins` is used for a numeric feature variable.
    :return: dual matplotlib figure.
    """

    if not all(data[target_var].isin([0, 1])):
        raise ValueError("Target and feature variables must contain only 0 and 1.")

    if bins is not None and not isinstance(bins, int):
        raise TypeError("Bins should be an integer or None.")

    if round_bins is not None and not isinstance(round_bins, int):
        raise TypeError("Round bins should be an integer or None.")

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

    # Rename columns
    grouped_data.columns = [f'{feature_var} Counts', target_var]
    grouped_data.reset_index(inplace=True)

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

    # Multiply by 100 for percentage display
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

    # Create a secondary y-axis chart
    ax2 = ax1.twinx()

    # Line plot creation
    ax2.set_ylabel(f'{target_var} in %', fontsize=16)
    sns.lineplot(x=feature_var, y=target_var, data=grouped_data, sort=False, color="C1", lw=5, ax=ax2)

    # Adjust tick sizes
    ax2.tick_params(axis='y', labelsize=8)

    # tight and modifying
    fig.tight_layout()
    plt.close(fig)
    return fig


def dual_line_bar_boxplot(data: pd.DataFrame, target_var: str, feature_var: str, chart="boxplot",
                          bins: int = None) -> Figure:
    """
    Generates a dual-axis chart combining a bar chart, boxplot or violin or boxenplot and a line chart, specifically
    designed for visualizing the distribution and impact of a feature variable on target variable on
    regression problems.

    The bar chart displays the count of observations for each bin or category of the feature variable, while the line
    chart shows the mean target variable on feature bands and boxplot/violet or boxenplot discribe its distribution

    :param data: The DataFrame containing the dataset to be visualized.
    :param target_var: The name of target variable.  Must be numerical
    :param feature_var: The name of the feature variable to analyze. Can be numeric or object.
    :param chart: boxplot/violin/boxenplot
    :param bins: Number of bins (default: 10).
    :return: Seaborn dual chart.
    """

    if bins is not None and not isinstance(bins, int):
        raise TypeError("Bins should be an integer or None.")

    # Set the bin number, and create the grouped dataframe
    if pd.api.types.is_numeric_dtype(data[feature_var]):
        # Determine the number of bins based on the specified method or default to 10
        num_bins = bins if bins is not None and bins not in ["auto", "fd", "scott", "rice", "sturges", "doane",
                                                             "sqrt"] else 10

        # Create bins
        bins_group = pd.cut(data[feature_var], bins=num_bins, right=False)

        # for boxplot only rawdata
        # todo merge codes
        bx_temp = data.copy()
        bx_temp[feature_var] = pd.cut(bx_temp[feature_var], bins=num_bins, right=False)
        bx_temp[feature_var] = bx_temp[feature_var].astype('category')
        bx_temp[feature_var] = bx_temp[feature_var].cat.rename_categories(bins_group)
        # Convert feature variable to string for categorical display
        bx_temp[feature_var] = bx_temp[feature_var].astype(str)
        bx_temp[feature_var] = bx_temp[feature_var].str.replace(", ", "-")

        # Calculate counts and mean for each bin to be used in the chart
        grouped_data = data.groupby(bins_group)[[feature_var, target_var]].agg(
            {feature_var: 'count', target_var: 'mean'})

    else:
        # If the feature variable is not numerical, aggregate using the feature variable directly
        # todo merge codes bx is temp
        bx_temp = data.copy()

        grouped_data = data.groupby(feature_var)[[feature_var, target_var]].agg(
            {feature_var: 'count', target_var: 'mean'})

    # Rename columns for clarity
    grouped_data.columns = [f'{feature_var} Counts', target_var]

    grouped_data.reset_index(inplace=True)  # Reset index

    # Convert feature variable to string for categorical display
    # todo
    grouped_data[feature_var] = grouped_data[feature_var].astype(str)
    grouped_data[feature_var] = grouped_data[feature_var].str.replace(", ", "-")

    # Create dual chart
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot creation
    ax1.set_title(f'{feature_var} Distribution with Mean {target_var}', fontsize=12, color='black')
    ax1.set_xlabel(f'{feature_var} Bins', fontsize=10, color='black')
    ax1.set_ylabel(f'{feature_var} Counts', fontsize=10, color='black')

    sns.barplot(x=feature_var, y=f'{feature_var} Counts', data=grouped_data, color='C0', ax=ax1)  # alpha=0.6

    # Adjust tick sizes
    ax1.tick_params(axis='both', labelsize=8)

    # Create a secondary y-axis for the mean line plot and boxplots
    ax2 = ax1.twinx()
    # Set y-axis limits
    min_target = data[target_var].min()
    max_target = data[target_var].max()
    ax2.set_ylim(min_target, max_target)

    # Create boxplots for each category
    # todo data for numerical
    if chart == 'boxplot':
        sns.boxplot(x=feature_var, y=target_var, data=bx_temp, ax=ax2, color='grey', fliersize=3)
    elif chart == 'violin':
        sns.violinplot(x=feature_var, y=target_var, data=bx_temp, ax=ax2, color='grey',
                       inner_kws=dict(box_width=15, whis_width=2, color=".8"))
    elif chart == 'boxenplot':
        sns.boxenplot(x=feature_var, y=target_var, data=bx_temp, ax=ax2, color='grey',
                      k_depth="trustworthy", trust_alpha=0.01)

    # Line plot for the mean of the continuous variable with color set to C1
    sns.lineplot(x=feature_var, y=target_var, data=grouped_data, lw=2, sort=False, ax=ax2, color='C1')

    # Set the labels for the secondary y-axis in black
    ax2.set_ylabel(f'Mean {target_var}', color='black')
    ax2.tick_params(axis='y', colors='black')

    # Change the color of the labels on the x-axis to black
    for label in ax1.get_xticklabels():
        label.set_color('black')

    # tight and modifying
    fig.tight_layout()
    plt.close(fig)
    return fig


def sub_hist_boxplot(data: pd.DataFrame, target_var: str, feature_var: str, stat="density",
                     bins: int = None) -> Figure:
    """
    Generates subplot with a density chart and boxplot for classification on numerical features.

    :param data: Pandas DataFrame containing the dataset.
    :param target_var: Name of the target variable (should be numeric).
    :param feature_var: Name of the numerical feature to analyze.
    :param stat: Optional. Type of statistic to compute for the density plot (default is "density").
    :param bins: Optional. Number of bins for the density plot (default is None).
    :return: Subplot with Seaborn density and boxplot.
    """

    if not (pd.api.types.is_numeric_dtype(data[target_var]) and pd.api.types.is_numeric_dtype(data[feature_var])):
        raise TypeError(f"Both {target_var} and {feature_var} should be numeric columns.")

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

    # Tighten the layout and close the figure
    fig.tight_layout()
    plt.close(fig)
    return fig


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

    chart.fig.tight_layout()
    plt.close(chart.fig)
    return chart.fig


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

    plt.tight_layout()
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
        elif isinstance(data_type, CategoricalDtype) or pd.api.types.is_object_dtype(data_type):
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
    Create a descriptive DataFrame from a given DataFrame.
    Includes data types, counts, NaN counts, unique values, top values, frequencies,
    statistical measures, quantiles, kurtosis, skew, and range for numerical variables.

    :param data: The input DataFrame.
    :returns: The comprehensive descriptive DataFrame.
    """

    # Initial data type, count, and null value calculations
    summary_df = pd.DataFrame(
        {
            'Dtype': data.dtypes,
            'Count': data.count(),
            'Non-Null Count': data.notnull().sum(),
            'NaN Count': data.isna().sum(),
            'Unique': data.apply(lambda x: x.nunique(), result_type='reduce'),
            'Top': data.apply(lambda x: x.value_counts().idxmax()),
            'Freq': data.apply(lambda x: x.value_counts().max())
        }
    )

    # Rounding the 'Top' value for numerical columns to 3 decimal places
    summary_df['Top'] = summary_df.apply(
        lambda row: round(row['Top'], 3) if (row['Dtype'] in ['float64', 'int64']) else row['Top'], axis=1)

    # Descriptive statistics for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    numeric_desc_df = data[numeric_cols].describe().T
    numeric_desc_df['5%'] = data[numeric_cols].quantile(0.05)
    numeric_desc_df['95%'] = data[numeric_cols].quantile(0.95)
    numeric_desc_df['Range'] = data[numeric_cols].max() - data[numeric_cols].min()
    numeric_desc_df['Kurtosis'] = data[numeric_cols].kurtosis()
    numeric_desc_df['Skew'] = data[numeric_cols].skew()

    # Combine the descriptive dataframes
    combined_df = summary_df.join(numeric_desc_df, how='left', rsuffix='_num')

    # Select and reorder columns, ensuring that we don't include the extra columns generated from join
    final_columns = ['Dtype', 'Count', 'NaN Count', 'Unique', 'Top', 'Freq',
                     'mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'Range', 'Kurtosis', 'Skew']
    final_df = combined_df[final_columns]

    # Round
    round_cols = ['mean', 'std', 'min', '5%', '25%', '50%', '75%', '95%', 'max', 'Range', 'Kurtosis', 'Skew']
    final_df.loc[:, round_cols] = final_df[round_cols].round(3)

    return final_df


if __name__ == "__main__":
    pass
