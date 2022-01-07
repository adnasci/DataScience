from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig
from ds_charts import bar_chart
from sklearn.impute import SimpleImputer
from pandas import concat, DataFrame
from ds_charts import get_variable_types
from numpy import nan
from pandas import DataFrame, concat
from ds_charts import get_variable_types
from sklearn.preprocessing import OneHotEncoder
from numpy import number

register_matplotlib_converters()
"""
folderName='images'
file = 'algae'
filename = 'data/algae.csv'
data = read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)
"""
"""
file = 'collisions'
filename = 'data/NYC_collisions_tabular.csv'
data = read_csv(filename, index_col='PERSON_INJURY', na_values='', parse_dates=True, infer_datetime_format=True)
folderName = 'Lab2Collisions'
"""

file = 'air'
filename = 'data/air_quality_tabular.csv'
data = read_csv(filename, index_col='ALARM', na_values='', parse_dates=True, infer_datetime_format=True)
folderName = 'Lab2Air'

mv = {}
figure(figsize=(8,10))
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)
savefig(f'{folderName}/{file}_missing_values.png')


threshold = data.shape[0] * 0.90
missings = [c for c in mv.keys() if mv[c]>threshold]
df = data.drop(columns=missings, inplace=False)
df.to_csv(f'{folderName}/{file}_drop_columns_mv.csv', index=False)
print('Dropped variables', missings)

threshold = data.shape[1] * 0.50
df = data.dropna(thresh=threshold, inplace=False)
df.to_csv(f'{folderName}/{file}_drop_records_mv.csv', index=False)
print(df.shape)


tmp_nr, tmp_sb, tmp_bool = None, None, None
variables = get_variable_types(data)
numeric_vars = variables['Numeric']
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']

if len(numeric_vars) > 0:
    imp = SimpleImputer(strategy='constant', fill_value=0, missing_values=nan, copy=True)
    tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
if len(symbolic_vars) > 0:
    imp = SimpleImputer(strategy='constant', fill_value='NA', missing_values=nan, copy=True)
    tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
if len(binary_vars) > 0:
    imp = SimpleImputer(strategy='constant', fill_value=False, missing_values=nan, copy=True)
    tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
df.to_csv(f'{folderName}/{file}_mv_constant.csv', index=False)
#print(df.describe(include='all'))


tmp_nr, tmp_sb, tmp_bool = None, None, None
variables = get_variable_types(data)
numeric_vars = variables['Numeric']
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']

tmp_nr, tmp_sb, tmp_bool = None, None, None
if len(numeric_vars) > 0:
    imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
    tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
if len(symbolic_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
if len(binary_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
df.to_csv(f'{folderName}/{file}_mv_most_frequent.csv', index=False)
#print(df.describe(include='all'))


data.dropna(inplace=True)

def dummify(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    X = df[vars_to_dummify]
    encoder.fit(X)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(X)
    dummy = DataFrame(trans_X, columns=new_vars, index=X.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)

    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df

print(data.dtypes)
cat_vars = data.select_dtypes(include='object')
data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
print(data.dtypes)

variables = get_variable_types(data)
symbolic_vars = variables['Symbolic']

df = dummify(data, symbolic_vars)
df.to_csv(f'{folderName}/{file}_dummified.csv', index=False)

print(df.describe(include=[bool]))