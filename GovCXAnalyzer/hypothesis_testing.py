import scipy.stats as ss
import scikit_posthocs as sp


def get_kruskal(df, col, target_col):
    """this function takes in a dataframe, the column for user groups, and the target col
    Returns: {'test': 'Kruskal',
    'N': 1267,
    'pvalue': 0.0017765384251897183,
    'statistic': 19.183534371601905,
    'likert_col': 'I am satisfied with the information I received from CareerOneStop.',
    'group_col': 'group',
    'data':           1         2         3         4         5         6
    1  1.000000  1.000000  0.001676  0.757479  1.000000  1.000000
    2  1.000000  1.000000  0.138366  1.000000  1.000000  1.000000
    3  0.001676  0.138366  1.000000  1.000000  0.521213  0.173964
    4  0.757479  1.000000  1.000000  1.000000  1.000000  0.521213
    5  1.000000  1.000000  0.521213  1.000000  1.000000  1.000000
    6  1.000000  1.000000  0.173964  0.521213  1.000000  1.000000}
    """
    cdf = df[[col, target_col]].dropna()
    
    data = [cdf.loc[ids, target_col].values for ids in cdf.groupby(col).groups.values()]
    
    
    rd = {}
    
    print('Kruskal')
    print('col:', col)
    print('target col: ', target_col)
    
    var = cdf.groupby(col)[target_col].apply(list).values
    
    N = cdf.shape[0]
    k = ss.kruskal(*list(var))
    
    print(k)
    print(f"N={N}")
    
    rd['test'] = 'Kruskal'
    rd['N']= cdf.shape[0]
    rd['pvalue'] = k.pvalue
    if k.pvalue < 0.05:
        print('++')
        
    
    rd['statistic'] = k.statistic
    rd['likert_col'] = target_col
    rd['group_col'] = col
    
    print('\n')
    
    rd['data'] =  sp.posthoc_conover(data, val_col=target_col, group_col=col, p_adjust = 'holm')
    
    return rd


def get_spearman(cdf, col, target_col):

    
    print('Spearman')
    print('col:', col)
    print('target col: ', target_col)
    
    var = cdf.groupby(col)[target_col].apply(list).values
    
    N = cdf.shape[0]
    k = ss.spearmanr(*list(var))
    rd = {}
    print(k)
    print(f"N={N}")
    
    rd['test'] = 'Spearman'
    rd['N']= cdf.shape[0]
    rd['pvalue'] = k.pvalue
    if k.pvalue < 0.05:
        print('++')
        
    
    rd['statistic'] = k.correlation
    rd['likert_col'] = target_col
    rd['group_col'] = col
    return rd



    



    