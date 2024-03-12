# python_modules

Contains EDA, data processing and ML modules

## Table of Contents


## Installation

```bash
pip install git+https://github.com/KyriakosJiannis/python_modules.git@main
```

## Usage

EDA modules examples at [eda charts.ipynb](./tests/eda charts.ipynb)
```python
from python_modules.eda_charts import dual_line_barplot, sub_hist_boxplot, tabular_plot, distribution_by_group, plot_distributions, descriptive_dataframe, dual_line_bar_boxplot
```
Flaml custom metrics
```python
from python_modules.flaml_custom_metrics import flaml_recall
```
Quick ML sklearn learn encoding pipe
```python
from python_modules.pipe_encoding import preprocessor, simplified_fit_impute
```
Time encoder to sin - con 
```python
from python_modules.time_encoder import TimeEncoder
```

Collection of stat tests
```python
from python_modules.stat_collects import compare_distributions, feature_analysis_with_tests 
```


## License
completely free and open-source and licensed under the MIT license.
