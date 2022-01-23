from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart, get_variable_types



register_matplotlib_converters()
filename = 'data/NYC_collisions_tabular.csv'
file = "NYC_collisions"
data = read_csv(filename, index_col='CRASH_DATE', na_values='', parse_dates=True, infer_datetime_format=True)

register_matplotlib_converters()
filename = 'data/air_quality_tabular.csv'
file = "air_quality"
data = read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)


print(data.shape)


figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig(f'images/{file}records_variables.png')

print(data.dtypes)

cat_vars = data.select_dtypes(include='object')
data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
print(data.dtypes)

variable_types = get_variable_types(data)
print(variable_types)
counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4,2))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
savefig(f'images/{file}variable_types.png')


mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure(figsize=(12,12))
bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values', rotation=True)
savefig(f'images/{file}mv.png')
