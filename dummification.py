from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from pandas import DataFrame, concat
from ds_charts import get_variable_types
from sklearn.preprocessing import OneHotEncoder
from numpy import number


register_matplotlib_converters()
file = 'algae'
filename = 'data/algae.csv'
data = read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)

# Drop out all records with missing values
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

variables = get_variable_types(data)
symbolic_vars = variables['Symbolic']
df = dummify(data, symbolic_vars)
df.to_csv(f'data/{file}_dummified.csv', index=False)

print(df.describe(include=[bool]))