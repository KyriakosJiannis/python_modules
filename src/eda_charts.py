"""This is a module for EDA proposes"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def classification_dual_barchart(data, target_var, feature_var, bins=None):
    """
    for Binary classification cases create a dual chart barchart and line charts.
    With the line to refers % target across and bars to numerical feature

    :param data: pandas dataframe
    :param target_var: target
    :param feature_var: numerical feature
    :param bins: number of bins
    :return: seaborn dual charts
    """
    # check if variable is numerical to set the bin number in line with matplotlib functionality,default 10 bins
    # https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html
    # Set the bin for numerical,
    # and aggregate dataframe to be used in a dual seaborn chart

    if pd.api.types.is_numeric_dtype(data[feature_var]):
        if bins is None:
            bins_num = 10
        elif bins in ["auto", "fd", "scott", "rice", "rice", "sturges", "doane", "sqrt"]:
            bins_num = len(np.histogram_bin_edges(data[feature_var].values, bins))
        else:
            bins_num = bins

        # Create the bands
        bins_group = pd.cut(data[feature_var], bins=bins_num, right=False)

        # Calculate for each band the counts and mean to be in the chart
        group = data.groupby(bins_group)[feature_var, target_var].agg({feature_var: 'count', target_var: 'mean'})
    else:
        group = data.groupby(feature_var)[feature_var, target_var].agg({feature_var: 'count', target_var: 'mean'})

    group.columns = [feature_var + " Counts", target_var]  # rename the features counts
    group.reset_index(inplace=True)  # reset index
    group[feature_var] = group[feature_var].astype(str)  # set bands as string
    group[target_var] = group[target_var] * 100  # multiply for percent
    group.dropna(inplace=True)

    # Create dual chart
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # bar plot creation
    ax1.set_title(feature_var + ' Distribution', fontsize=12)
    ax1.set_xlabel(feature_var, fontsize=10)
    ax1.set_ylabel(target_var, fontsize=10)

    ax1 = sns.barplot(x=feature_var, y=feature_var + " Counts", data=group, color='C0')

    ax1.tick_params(axis='y', labelsize=8)
    ax1.tick_params(axis='x', labelsize=8)

    # specify we want to share the same x-axis
    ax2 = ax1.twinx()

    # line plot creation
    ax2.set_ylabel(target_var + '%', fontsize=12)
    ax2 = sns.lineplot(x=feature_var, y=target_var, data=group, sort=False, color="C1", lw=5)

    ax2.tick_params(axis='y', labelsize=8)
    plt.show()


def classification_destiny_boxplot(data, target_var, feature_var):
    """
    For classification cases creates a subplot with a destiny chart
    nd boxplot on target variable classes for a numerical feature

    :param data: pandas dataframe
    :param target_var: target
    :param feature_var: numerical feature
    :return: subplot with seaborn destiny and boxplot
    """

    # Create chart
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    labels = np.sort(data.target.unique())

    # plot creation
    ax = fig.add_subplot(2, 2, 1)
    for ix in labels:
        sns.distplot(data[feature_var][data[target_var] == labels[ix]],  bins=10, hist_kws={'alpha': 0.6}, ax=ax)

    ax = fig.add_subplot(2, 2, 2)
    sns.boxplot(x=target_var, y=feature_var, data=data, ax=ax)

    plt.show()


def classification_tabular_chart(data, target_var, feature_var):
    """
    For classification cases creates the tabular chart between target and ordinal /categorical feature

    :param data: pandas dataframe
    :param target_var: target
    :param feature_var: ordinal/categorical  feature
    :return: tabular chart using seaborn catplot
    """
    cross_df = data.groupby(feature_var)[target_var].value_counts(normalize=True)
    cross_df = cross_df.mul(100).rename('Percent').reset_index()

    g = sns.catplot(x=feature_var, y='Percent', hue=target_var, kind='bar', data=cross_df)
    g.ax.set_ylim(0, 100)
    plt.show()


def distribution_by_group(data, feature_var, by):
    """
    Create destiny charts for the classes of by variable

    :param data: pandas dataframe
    :param feature_var: num feature
    :param by: ordinal/categorical  feature
    :return:
    """

    fig = plt.figure(figsize=(12, 10))

    labels = np.sort(data[by].unique()).tolist()

    for ix in labels:
        sns.distplot(data[feature_var][data[by] == ix], rug=False, norm_hist=False, hist_kws={'alpha': 0.6})

    fig.legend(labels=labels,
               title=by,
               bbox_to_anchor=(1.01, 1),
               loc='upper left')

    plt.ylabel('Density', fontsize=12 * 0.8)
    plt.xlabel(feature_var, fontsize=12 * 0.8)
    plt.title("Distribution of " + feature_var + " by " + by, fontsize=12)
    plt.show()
