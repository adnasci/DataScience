from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show, title
from ds_charts import bar_chart, get_variable_types, multiple_line_chart, choose_grid, HEIGHT, multiple_bar_chart
from ds_charts import compute_known_distributions, histogram_with_distributions
from matplotlib.pyplot import savefig, show, subplots, figure, Axes
from seaborn import distplot, heatmap
from numpy import log
from pandas import Series
from scipy.stats import norm, expon, lognorm


"""
register_matplotlib_converters()
filename = 'data/NYC_collisions_tabular.csv'
data = read_csv(filename, index_col='PERSON_INJURY', na_values='', parse_dates=True, infer_datetime_format=True)
folderName = 'Lab1Collisions'
"""
register_matplotlib_converters()
filename = 'data/air_quality_tabular.csv'
data = read_csv(filename, index_col='ALARM', na_values='', parse_dates=True, infer_datetime_format=True)
folderName = 'Lab1Air'

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

def granularity_single():
    data = read_csv(filename)
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

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

def granularity_study_variable():
    data = read_csv(filename)
    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
    
    #variable = 'PERSON_INJURY'
    variable = 'ALARM'
    bins = (10, 100, 1000, 10000)
    fig, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
    for j in range(len(bins)):
        axs[j].set_title('Histogram for %s %d bins'%(variable, bins[j]))
        axs[j].set_xlabel(variable)
        axs[j].set_ylabel('Nr records')
        axs[j].hist(data[variable].values, bins=bins[j])
    savefig(f'{folderName}/granularity_study_{variable}.png')

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
######################################################################

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
    cat_vars = data.select_dtypes(include='float')
    data[cat_vars.columns] = data.select_dtypes(['float']).apply(lambda x: x.astype('category'))

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

######################################################################

def DataDimensionality():
    records_variables()
    variable_types()
    MissingValues()

def DataDistribution():
    single_boxplots()
    outliers()
    single_histograms_numeric()
    histograms_trend_numeric()
    #histogram_numeric_distribution() #demora muito tempo a correr
    #histograms_symbolic() # demora muito tempo

def DataGranularity():
    granularity_single()
    granularity_study_variable()
    granularity_study()

def DataSparsity():
    sparsity_study_numeric()
    #sparsity_study_symbolic() # nao funciona
    correlation_analysis()
