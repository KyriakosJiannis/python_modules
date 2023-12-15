import matplotlib

matplotlib.use('TkAgg', force=True)
from matplotlib import pyplot as plt

print("Switched to:", matplotlib.get_backend())
import pandas as pd
import sys

sys.path.append('../')
from python_modules.eda_charts import dual_line_barplot, sub_hist_boxplot, tabular_plot, distribution_by_group, \
    plot_distributions, descriptive_dataframe, dual_line_bar_boxplot

# import
df = pd.read_csv('https://filedn.com/lK8J7mCaIwsQFcheqaDLG5z/data/heart.csv')
df.head()

cp_mapping = {
    0: 'typical angina',
    1: 'atypical angina',
    2: 'non-anginal pain',
    3: 'asymptomatic'
}

df['cp'] = df['cp'].map(cp_mapping)

# debug
dff = descriptive_dataframe(df)
plot_distributions(df)
plot_distributions(df, by='target')
dual_line_barplot(df, target_var='target', feature_var='trestbps', bins=8, round_bins=0)
sub_hist_boxplot(df, target_var='target', feature_var='trestbps', stat="percent")
tabular_plot(df, target_var='target', feature_var='cp')
dual_line_bar_boxplot(df, target_var="chol", feature_var='cp', chart="boxenplot")
distribution_by_group(df, "chol", by="slope")
