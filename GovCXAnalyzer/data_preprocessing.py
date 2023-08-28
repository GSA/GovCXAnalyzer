import pandas as pd
import re
import string
import seaborn as sns
from collections import Counter
from .utils import *
from nltk.corpus import stopwords
import user_agents
import string
from nltk.corpus import stopwords
from urllib.parse import urlparse, unquote, parse_qs
from nltk.tokenize import TweetTokenizer
stop_words = set(stopwords.words("english"))
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from ekphrasis.classes.spellcorrect import SpellCorrector
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

def camel_case_split(s):

    # use map to add an underscore before each uppercase letter
    modified_string = list(map(lambda x: '_' + x if x.isupper() else x, s))
    # join the modified string and split it at the underscores
    split_string = ''.join(modified_string).split('_')
    # remove any empty strings from the list
    split_string = list(filter(lambda x: x != '', split_string))
    return split_string

def load_data(filepath):
    # Example: Load the data (replace with actual loading code if necessary)
    df = pd.read_csv(filepath)
    
    return df

def read_json(path):
    
    with open(path, 'r') as fin:
        
        dat = json.load(fin)
    
    return dat

def read_jsonl(path):
    return pd.read_json(path, lines=True)


def preprocess_columns(df):
    """Preprocesses the columns of a dataframe."""
    df = df.dropna(axis=1, how='all')
    for col in df.columns:
        if len(df[col].unique()) == 1:
            df.drop(col,inplace=True,axis=1)
    return df

def make_numeric_subset(df):
    """Filter and return only numeric columns from a dataframe."""
    numeric_df = df.select_dtypes(include=['int', 'float'])
    print(numeric_df.columns)
    return numeric_df

def make_categorical_subset(df):
    """Filter and return only numeric columns from a dataframe."""
    cat_df = df.select_dtypes(include=['object'])
    print(cat_df.columns)
    return cat_df


def convert_boolean_vals(df, colinterest, val_map={'yes': 1, 'no': 0}):
    df[f"{colinterest}_bool"] = df[colinterest].map(lambda x: val_map)


def cln(i, extent=1):
    if isinstance(i, str) and i != "":
        if extent == 1:
            return re.sub(r"\s\s+", " ", i)
        elif extent == 2:
            return re.sub(r"\s+", "", i)
    else:
        return i


def make_categorical(series):
    q = np.quantile(series, 0.25)
    return series.map(lambda x: 'no' if x < q else 'yes')
    
def convert_likert_cols(df, likert_cols, agree_scale=False):
    agree2likert = {'Strongly Agree': 5,'Agree': 4, 'Neutral': 3,'Disagree':2, 'Strongly Disagree': 1,}
    
    
    inv_map = make_inv_map(agree2likert)
    dfnew = df.copy()
    if agree_scale:
        for col in likert_cols:
            
            dfnew[col] = dfnew[col].map(lambda x: inv_map.get(x, x))
    else:
        
    
    
        for col in likert_cols:
            dfnew[col] = df[col].map(lambda x: agree2likert.get(x, x))
    return dfnew


def make_date_touchpoints_df(df, datecol="Created At"):
    
    df['CreatedAt'] = df[datecol].map(pd.to_datetime)
    
    df['year'] = df['CreatedAt'].map(lambda dt: dt.year)
    df['date'] = df['CreatedAt'].map(lambda x: x.strftime("%b-%Y"))
    
    return df


def make_user_agent_info(user_agent):
    """pass user_agent object to retrieve"""
    d = {}
   
    d['os'] = user_agent.os.family#
    d['browser'] = user_agent.browser.family
    d['device'] = user_agent.device.family
    d['is_mobile'] = user_agent.is_mobile # returns True
    d['is_pc'] =user_agent.is_pc # returns False
   
    return d


def make_user_agent_df_cols(df, user_agent_col="User Agent"):
    user_agent_df = df['User Agent'].map(user_agents.parse).map(make_user_agent_info).apply(pd.Series)
    user_agent_cols = list(user_agent_df.columns)
    return user_agent_df

def make_user_agent_info(user_agent):
    """pass user_agent object to retrieve"""
    d = {}
   
    d['os'] = user_agent.os.family#
    d['browser'] = user_agent.browser.family
    d['device'] = user_agent.device.family
    d['is_mobile'] = user_agent.is_mobile # returns True
    d['is_pc'] =user_agent.is_pc # returns False
   
    return d



def init_textprocesser():
    text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
    )   
    return text_processor


def return_preprocess_sents(sentences,text_processor):
    l = []
    for s in sentences:
        l.append(" ".join(text_processor.pre_process_doc(s)))
    return l


def init_spellcorrector():

    sp = SpellCorrector(corpus="english")
    return sp


def preprocess_text(text, lower=True):
    """takes a tqeet text and tokenizes it"""


    tok = TweetTokenizer()

    tokens = tok.tokenize(text)
    if lower is True:
        return ' '.join([token.lower() for token in tokens])
    else:
        return ' '.join(tokens)
 

def normalize_dataframe(df, columns):
    """
    uses min max scaling
    :param df: pandas dataframe object
    :param columns: list of nominal features
    :return: scaled dataframe with column values processed with min max scaling
    """

    result = df.copy()
    for feature_name in columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def return_no_str(val):

    if str(val).lower().strip() in ["none","nan", "n/a"]:
        return None
    else: 
        return val
    
def remove_digits(ini_string):
    return re.sub(r'\d+', ' ', ini_string)

def remove_white_spaces(ini_string):
    return re.sub(r'\s+', ' ', ini_string)


def lexical_diversity(my_text_data):
    """
    input is list of text data
    output gives diversity_score
    """
    word_count = len(my_text_data)
    vocab_size = len(set(my_text_data))
    diversity_score = word_count / vocab_size
    return diversity_score


def scale(data_matrix):
    """returns the means and standard deviations of each columns"""
    num_rows, num_cols = shape(data_matrix)
    means = [mean(get_column(data_matrix, j)) for j in range(num_cols)]
    stdevs = [standard_deviation(get_column(data_matrix, j)) for j in range(num_cols)]
    return means, stdevs

def rescale(data_matrix):
    """rescales the input data so that each column has a mean 0 and atd of 1 leaves alone columns with no deviation"""
    means, stdevs = scale(data_matrix)
    def rescaled(i, j):
        if stdevs[j] > 0:
            return (data_matrix[i][j] - mean[j]/stdevs[j])
        else:
            return data_matrix[i][j]
    
    num_rows, num_cols = shape(data_matrix)
    return make_matrix(num_rows, num_cols, rescaled)

def wavg(group, avg_name, weight_name):
    """ http://stackoverflow.com/questions/10951341/pandas-dataframe-aggregate-function-using-multiple-columns
    In rare instance, we may not have weights, so just return mean
    """
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return d.mean()

def get_weights_from_terms(d, average=False):
    """want to add heigher weright for words more similar to aut"""
    
    if not average:
        return np.sum(list(d.values()))
    
    else:
        n = len(t)
        return np.sum(list(d.values())) / n
    

def get_dict_withind_interest(d_, kw):
    return_dict = {}
    dww = d_.get('w').lower()
    if kw.lower() in dww.lower():
        return_dict[t] = kw
    
    t = d_.get('t')
    if strip_punctuation(kw.lower()) in strip_punctuation(t.lower()):
        return_dict[t] = kw

    
        
        