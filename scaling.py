from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame, concat
from matplotlib.pyplot import subplots, show, figure, savefig




register_matplotlib_converters()
file = 'algae'
filename = 'data/algae.csv'
data = read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)


variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']

df_nr = data[numeric_vars]
df_sb = data[symbolic_vars]
df_bool = data[boolean_vars]


# Standard scaler with z-score transformation

transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_zscore = concat([tmp, df_sb,  df_bool], axis=1)
norm_data_zscore.to_csv(f'data/{file}_scaled_zscore.csv', index=False)

# Standard scaler with MinMax  

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_minmax = concat([tmp, df_sb,  df_bool], axis=1)
norm_data_minmax.to_csv(f'data/{file}_scaled_minmax.csv', index=False)
print(norm_data_minmax.describe())



fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('Z-score normalization')
norm_data_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 2])
savefig(f'images/scaled.png')

norm_data_zscore.to_csv('data/algae_scaled_zscore.csv', index=False)


