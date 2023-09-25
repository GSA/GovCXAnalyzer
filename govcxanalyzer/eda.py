import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
from IPython import display
from ydata_profiling import ProfileReport

## specifically for OMB A-11 - nullity + processing

def get_profile(df, use_omb_a11=True, idcols =  ['ID', 'UUID'], 
    additional_cols = ['CreatedAt', "Page", "Referrer", "User Agent"]):
    
    if use_omb_a11:
        return ProfileReport(df[[col for col in list(df.columns) if col not in idcols + additional_cols]])
    else:
        return ProfileReport(df)

        

# Utility functions
def count_unique_values(df, cat_col_list):
    cat_df = df[cat_col_list]
    val_df = pd.DataFrame({
        'columns': cat_df.columns, 
        'cardinality': cat_df.nunique()
    })
    return val_df

def check_null_values(df):
    null_df = pd.DataFrame({
        'columns': df.columns, 
        'percent_null': df.isnull().sum() * 100 / len(df), 
        'percent_zero': df.isin([0]).sum() * 100 / len(df)
    })
    return null_df 


def create_cardinality_feature(df):
    num_rows = len(df)
    random_code_list = np.arange(100, 1000, 1)
    return np.random.choice(random_code_list, num_rows)


stats_dict = {
    'sum': 'TotalCount',
    'average': 'Avg',
    'mean': 'Avg',
    'median': 'Median',

}


def get_stat(df, label, aggfunc):
    name_ = stats_dict.get(aggfunc)
    return pd.pivot_table(data=df, values=label, aggfunc=aggfunc).rename(
        columns={label: label.split('N_')[-1].title() + name_}).round(1)


def NullUnique(df):
    dic = defaultdict(list)
    for col in df.columns:
        dic['Feature'].append(col)
        dic['NumUnique'].append(len(df[col].unique()))
        dic['NumNull'].append(df[col].isnull().sum())
        dic['%Null'].append(round(df[col].isnull().sum()/df.shape[0] * 100,2))
    return pd.DataFrame(dict(dic)).sort_values(['%Null'],ascending=False).head(18).style.background_gradient()


def data_range(x):
    return max(x) - min(x)

def de_mean(x):
    """translate x by subtractig its mean so the result has mean 0"""
    x_bar = mean(x)
    
    return [x_i - x_bar for x_i in x]


def variance(x):
    """assumes x has at least two elemeents"""
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

def standard_deviation(x):
    return math.sqrt(variance(x))

def interquartile_range(x):
    return quantile(x, 0.75) - quantile(x, 0.25)

### correlation

def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y))/ (n -1)

def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0 # no variation, correlation is 0


def create_table_for_plots(df, col, top_n=10, explode=True):
    if explode:
    
        return df[col].explode().value_counts().sort_values(ascending=False)[:top_n]
    else:
        return df[col].value_counts().sort_values(ascending=False)[:top_n]


def visualize_cat_distributions(df, c):
    # visualize categorical distributions
    df[c].value_counts().plot(kind='bar')
    plt.show()
    plt.close()

## exploratory data analysis and visualizations for various data types 
def make_swarm_plot(df, x="attributes_q2_point_scale", y="agency", hue="channel"):
    sns.set_theme(style="ticks", palette="pastel")

    ax = sns.swarmplot(data=df, x=x, y=y, hue=hue)
    ax.set(ylabel="")
    plt.show()
    return ax


def make_boxplot(df, x="service provider", y="attributes_q2_point_scale",hue="channel"):
    g = sns.boxplot(data=df, x=x, y=y, hue=hue)

    sns.despine(offset=1, trim=True)
    plt.xticks(rotation=90)

    plt.show()
    return g
    

def make_cat_plot(df, category_code,title, topk=15, color='paleturquoise'):
    
    if df[category_code].dropna().unique() < topk:
        n = len(df[category_code].dropna().unique())
    else: 
        n = topk
        
    df2 = df.copy()
    height = df2[category_code].value_counts()[:n].tolist()
    bars =  df2[category_code].value_counts()[:n].index.tolist()
    y_pos = np.arange(len(bars))
    a = plt.bar(y_pos, height , width=0.7 ,color= ['c']+[color]*14)
    plt.xticks(y_pos, bars)
    plt.xticks(rotation=90)
    plt.title(f"Top {str(n)} {title}", fontdict=None)
    plt.show()
    return a
    
    
def plot_likert_group_col(df, group_col, likert_col):
    cdf = df[[group_col, likert_col]].dropna()

    g = sns.catplot(
        data=cdf, x=group_col, y=likert_col,
        capsize=.2, palette="YlGnBu_d", errorbar="se",
        kind="point", height=6, aspect=.75,
    )
    g.despine(left=True)


    plt.xticks(rotation = 'vertical')

    degrees = 90  # Adjust according to one's preferences/needs
    plt.xticks(rotation=degrees)
    plt.show()
    return g


def create_catplot(df, xcat, ylikert):
    # Draw a pointplot to show pulse as a function of categorical factors
    g  = sns.catplot(
    data=df, x=xcat, y=ylikert,
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75,)
    g.despine(left=True)

    return g

def return_likert_boxenplot(df, likertcol):

    return sns.boxenplot(
   df[likertcol]
   )


def return_distplot_quant(df,  quantcol):
    return sns.distplot(
    df[quantcol])


# Utility functions
def count_unique_values(df, cat_col_list):
    cat_df = df[cat_col_list]
    val_df = pd.DataFrame({
        'columns': cat_df.columns, 
        'cardinality': cat_df.nunique()
    })
    return val_df

def check_null_values(df):
    null_df = pd.DataFrame({
        'columns': df.columns, 
        'percent_null': df.isnull().sum() * 100 / len(df), 
        'percent_zero': df.isin([0]).sum() * 100 / len(df)
    })
    return null_df 


def create_cardinality_feature(df):
    num_rows = len(df)
    random_code_list = np.arange(100, 1000, 1)
    return np.random.choice(random_code_list, num_rows)


def make_segment_likert_table(df, usergroupcol, likertcol):
    """takes a dataframe, usergroup, and specific target col of interest"""
    table =  df.groupby(usergroupcol)[likertcol].describe()

    table = table.reset_index()
    display.display(table)

    
    return table


stats_dict = {
    'sum': 'TotalCount',
    'average': 'Avg',
    'mean': 'Avg',
    'median': 'Median',

}


def get_stat(df, label, aggfunc):
    name_ = stats_dict.get(aggfunc)
    return pd.pivot_table(data=df, values=label, aggfunc=aggfunc).rename(
        columns={label: label.split('N_')[-1].title() + name_}).round(1)


def NullUnique(df):
    dic = defaultdict(list)
    for col in df.columns:
        dic['Feature'].append(col)
        dic['NumUnique'].append(len(df[col].unique()))
        dic['NumNull'].append(df[col].isnull().sum())
        dic['%Null'].append(round(df[col].isnull().sum()/df.shape[0] * 100,2))
    return pd.DataFrame(dict(dic)).sort_values(['%Null'],ascending=False).head(18).style.background_gradient()


def data_range(x):
    return max(x) - min(x)

def de_mean(x):
    """translate x by subtractig its mean so the result has mean 0"""
    x_bar = mean(x)
    
    return [x_i - x_bar for x_i in x]


def variance(x):
    """assumes x has at least two elemeents"""
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

def standard_deviation(x):
    return math.sqrt(variance(x))

def interquartile_range(x):
    return quantile(x, 0.75) - quantile(x, 0.25)

### correlation

def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y))/ (n -1)

def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0 # no variation, correlation is 0


def make_year_pct_pos(df, yearcol, cxdriver):
    """plot for creating percent positive by yr"""
    dfyear = pd.crosstab(df[yearcol], df[cxdriver]).reset_index()
    dfyear[yearcol] = dfyear[yearcol].map(np.int)
    dfyear[f'PCT_POSITIVE'] = dfyear['yes'] / (dfyear['yes'] + dfyear['no'])
    dfyear['PCT_NONPOSITIVE'] = dfyear['no'] / (dfyear['yes'] + dfyear['no'])
    dfpctgranted =pd.melt(dfyear, id_vars=[yearcol], value_vars=['PCT_POSITIVE', 'PCT_NONPOSITIVE'])
    sns.set_style("white")
    ax = sns.barplot(x=yearcol, y="value", hue=cxdriver, data=dfpctgranted)
    plt.show()
    return ax

