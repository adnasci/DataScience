from pandas import read_csv, concat, DataFrame
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, subplots, title
from ds_charts import bar_chart, get_variable_types, choose_grid, HEIGHT, multiple_bar_chart
from ds_charts import histogram_with_distributions, compute_known_distributions
from seaborn import distplot, heatmap
from sklearn.impute import SimpleImputer
from numpy import nan, number
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler




##################################################################
#                      Duvidas

# O granularity study com target variable person injury nao mostra o numero d records com killed
# Symbolic data sparsity -> é suposto a data e o crash time serem symbolic??
# Dummified demora muito tempo a correr
##################################################################



register_matplotlib_converters()

filename = 'data/algae.csv'
data = read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)

filename = 'data/NYC_collisions_tabular.csv'
file = 'NYC_collisions'
data = read_csv(filename, index_col='CRASH_DATE', na_values='', parse_dates=True, infer_datetime_format=True)
#folderName = 'Lab1Collisions'
folderName = 'Lab2Collisions'

#print(data.dtypes)
cat_vars = data.select_dtypes(include='object')
data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
#print(data.dtypes)    

##################################################################
#                      Lab 1
##################################################################
#                      Data Dimensionality
##################################################################

def records_variables():
    figure(figsize=(4,2))
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
    bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
    savefig(f'{folderName}/records_variables.png')

################################################################## 

def variable_types():
    variable_types = get_variable_types(data)
    counts = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])
    figure(figsize=(4,2))
    bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
    savefig(f'{folderName}/variable_types.png')

##################################################################

def MissingValues():
    mv = {}
    for var in data:
        nr = data[var].isna().sum()
        if nr > 0:
            mv[var] = nr

    figure(figsize=(8,16))
    bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
                xlabel='variables', ylabel='nr missing values', rotation=True)
    savefig(f'{folderName}/MissingValues.png')

##################################################################
#                      Data Distribution
##################################################################

def single_boxplots():
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Boxplot for %s'%numeric_vars[n])
        axs[i, j].boxplot(data[numeric_vars[n]].dropna().values)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'{folderName}/single_boxplots.png')

##################################################################

def outliers():
    NR_STDEV: int = 2
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    outliers_iqr = []
    outliers_stdev = []
    summary5 = data.describe(include='number')

    for var in numeric_vars:
        iqr = 1.5 * (summary5[var]['75%'] - summary5[var]['25%'])
        outliers_iqr += [
            data[data[var] > summary5[var]['75%']  + iqr].count()[var] +
            data[data[var] < summary5[var]['25%']  - iqr].count()[var]]
        std = NR_STDEV * summary5[var]['std']
        outliers_stdev += [
            data[data[var] > summary5[var]['mean'] + std].count()[var] +
            data[data[var] < summary5[var]['mean'] - std].count()[var]]

    outliers = {'iqr': outliers_iqr, 'stdev': outliers_stdev}
    figure(figsize=(12, HEIGHT))
    multiple_bar_chart(numeric_vars, outliers, title='Nr of outliers per variable', xlabel='variables', ylabel='nr outliers', percentage=False)
    savefig(f'{folderName}/outliers.png')

##################################################################

def single_histograms_numeric():
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram for %s'%numeric_vars[n])
        axs[i, j].set_xlabel(numeric_vars[n])
        axs[i, j].set_ylabel("nr records")
        axs[i, j].hist(data[numeric_vars[n]].dropna().values, 'auto')
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'{folderName}/single_histograms_numeric.png')

##################################################################

def histograms_trend_numeric():
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        axs[i, j].set_title('Histogram with trend for %s'%numeric_vars[n])
        distplot(data[numeric_vars[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'{folderName}/histograms_trend_numeric.png')

##################################################################

def histogram_numeric_distribution():
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')
    
    rows, cols = choose_grid(len(numeric_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(numeric_vars)):
        histogram_with_distributions(axs[i, j], data[numeric_vars[n]].dropna(), numeric_vars[n])
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'{folderName}/histogram_numeric_distribution.png')

##################################################################

def histograms_symbolic():
    symbolic_vars = get_variable_types(data)['Symbolic']
    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')

    rows, cols = choose_grid(len(symbolic_vars))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(symbolic_vars)):
        counts = data[symbolic_vars[n]].value_counts()
        bar_chart(counts.index.to_list(), counts.values, ax=axs[i, j], title='Histogram for %s'%symbolic_vars[n], xlabel=symbolic_vars[n], ylabel='nr records', percentage=False)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'{folderName}/histograms_symbolic.png')

##################################################################
#                      Data Granularity
##################################################################
def granularity_single():
    variables = get_variable_types(data)['Numeric']
    if [] == variables:
        raise ValueError('There are no numeric variables.')

    rows, cols = choose_grid(len(variables))
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    i, j = 0, 0
    for n in range(len(variables)):
        axs[i, j].set_title('Histogram for %s'%variables[n])
        axs[i, j].set_xlabel(variables[n])
        axs[i, j].set_ylabel('nr records')
        axs[i, j].hist(data[variables[n]].values, bins=100)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'{folderName}/granularity_single.png')

##################################################################

def granularity_study_variable():
    #data = read_csv(filename)
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
    
    variable = 'PERSON_INJURY'
    #variable = 'ALARM'
    bins = (10, 100, 1000, 10000)
    fig, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
    for j in range(len(bins)):
        axs[j].set_title('Histogram for %s %d bins'%(variable, bins[j]))
        axs[j].set_xlabel(variable)
        axs[j].set_ylabel('Nr records')
        axs[j].hist(data[variable].values, bins=bins[j])
    savefig(f'{folderName}/granularity_study_{variable}.png')

##################################################################


def granularity_study():
    data = read_csv(filename)
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
    
    variables = get_variable_types(data)['Numeric']
    if [] == variables:
        raise ValueError('There are no numeric variables.')

    rows = len(variables)
    bins = (10, 100, 1000)
    cols = len(bins)
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
            axs[i, j].set_xlabel(variables[i])
            axs[i, j].set_ylabel('Nr records')
            axs[i, j].hist(data[variables[i]].values, bins=bins[j])
    savefig(f'{folderName}/granularity_study.png')


##################################################################
#                      Data Sparsity
##################################################################


def sparsity_study_numeric():
    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(numeric_vars)):
        var1 = numeric_vars[i]
        for j in range(i+1, len(numeric_vars)):
            var2 = numeric_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    savefig(f'{folderName}/sparsity_study_numeric.png')

######################################################################

def sparsity_study_symbolic():
    symbolic_vars = get_variable_types(data)['Symbolic']
    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')

    rows, cols = len(symbolic_vars)-1, len(symbolic_vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT, rows*HEIGHT), squeeze=False)
    for i in range(len(symbolic_vars)):
        var1 = symbolic_vars[i]
        for j in range(i+1, len(symbolic_vars)):
            var2 = symbolic_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    savefig(f'{folderName}/sparsity_study_symbolic.png')

######################################################################

def correlation_analysis():
    corr_mtx = abs(data.corr())
    fig = figure(figsize=[18,18])
    heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    title('Correlation analysis')
    savefig(f'{folderName}/correlation_analysis.png')


##################################################################
#                      Lab 2
##################################################################

mv = {}
figure(figsize=(12,16))
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr
bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable', xlabel='variables', ylabel='nr missing values', rotation=True)
savefig(f'{folderName}/{file}_missing_values.png')


#                      Dropping Missing Values
##################################################################


# When a column has a significant number of missing values
# Discard the entire column
threshold = data.shape[0] * 0.90
missings = [c for c in mv.keys() if mv[c]>threshold]
df = data.drop(columns=missings, inplace=False)
df.to_csv(f'data/{file}_drop_columns_mv.csv', index=False)
#print('Dropped variables', missings)

# Presence of single records that have a majority of variables without values. 
# In this case, we prefer to discard the records instead of dropping all columns

threshold = data.shape[1] * 0.50
df = data.dropna(thresh=threshold, inplace=False)
df.to_csv(f'data/{file}_drop_records_mv.csv', index=False)
#print(df.shape)

#                      Filling Missing Values
##################################################################


# Filling with a constant value
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
df.to_csv(f'data/{file}_mv_constant.csv', index=False)
#print(df.describe(include='all'))

# Usually apply the mean and mode instead
# Don't forget to save the resulting data to a datafile, 
#   to be used for training models and discovering other kinds of information.
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
df.to_csv(f'data/{file}_mv_most_frequent.csv', index=False)
#print(df.describe(include='all'))

#                      Data Preparation
##################################################################

# Dummification

# Drop out all records with missing values
#data.dropna(inplace=True)

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
# Demora muito tempo a correr
#df = dummify(data, symbolic_vars)
#df.to_csv(f'data/{file}_dummified.csv', index=False) 
#print(df.describe(include=[bool]))


# Scaling

variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']

df_nr = data[numeric_vars]
df_sb = data[symbolic_vars]
df_bool = data[boolean_vars]

# Z-Score transformation
transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_zscore = concat([tmp, df_sb,  df_bool], axis=1)
norm_data_zscore.to_csv(f'data/{file}_scaled_zscore.csv', index=False)

#MinMaxScaler
transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_minmax = concat([tmp, df_sb,  df_bool], axis=1)
norm_data_minmax.to_csv(f'data/{file}_scaled_minmax.csv', index=False)
#print(norm_data_minmax.describe())

fig, axs = subplots(1, 3, figsize=(20,10),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('Z-score normalization')
norm_data_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 2])
savefig(f'{folderName}/SingleBoxplot_Scaling.png')


##################################################################
#                      Lab 3
##################################################################

#                      Classification
##################################################################

