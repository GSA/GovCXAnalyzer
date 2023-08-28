from __future__ import absolute_import
import string
import unicodedata
import re
import numpy as np
from collections import defaultdict
import pandas as pd
import datetime
import ujson

import warnings
 
def warning_function():
    warnings.warn("deprecated", DeprecationWarning)
 
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    warning_function() 


def to_string(s):
    """Converts input to a string."""
    try:
        return str(s)
    except:
        return s.encode('utf-8')

def unicodeToAscii(s):
    """Converts unicode to ASCII."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def create_cardinality_feature(df):
    """Creates a cardinality feature for a dataframe."""
    num_rows = len(df)
    random_code_list = np.arange(100, 1000, 1)
    return np.random.choice(random_code_list, num_rows)

def print_current_date():
    todaysdate = datetime.datetime.today().strftime('%Y-%m-%d %H:%M')
    print(f"{todaysdate}")

def get_date_timestamp():
    todaysdate = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    return todaysdate

toeval = lambda x: eval(x) if isinstance(x, str) and x[0]=='[' else x
tolist = lambda x: x if isinstance(x, list) else (x if pd.isnull(x) else [x])

def term_in_val(val, term):
    if term in val:
        return True
    else:
        return False

def items_present_test(input_list, clist):
    return any(x in input_list for x in clist)


def chunks(l, n):
    """Yield n number of sequential chunks from l."""
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield l[si:si + (d + 1 if i < r else d)]


flatten = lambda l: [item for sublist in l for item in sublist]

def replace_none(val, returnval = '--'):
    if val == None:
        return returnval
    
    else:
        return val
    

def matches(input1, input2):
    return (input1 == input2)

def intersection_two_lists(lst1, lst2):
    return set(lst1).intersection(lst2)


def cln(i, extent=1):
    """
    String white space 'cleaner'.
    :param i:
    :param extent: 1 --> all white space reduced to length 1; 2 --> removal of all white space.
    :return:
    """

    if isinstance(i, str) and i != "":
        if extent == 1:
            return re.sub(r"\s\s+", " ", i)
        elif extent == 2:
            return re.sub(r"\s+", "", i)
    else:
        return i

transtable = str.maketrans(dict.fromkeys(string.punctuation))


def strip_punctuation(input_string):
    """cleans string by stripping punctuation """
    return input_string.translate(transtable)

def get_match(text, rex):
    if isinstance(rex, (list, tuple, set)):
        rex = '(' + '|'.join(rex) + ')'
    result = re.findall(rex, text)
    return result

def partial_match(input_str, looking_for):
    """
    :param input_str:
    :param looking_for:
    :return:
    """
    if isinstance(input_str, str) and isinstance(looking_for, str):
        return cln(looking_for.lower(), 1) in cln(input_str.lower(), 1)
    else:
        return False


def return_no_str(val):

    if str(val).lower().strip() in ["none","nan", "n/a"]:
        return None
    else: 
        return val
    

def listify(l): return l if isinstance(l, (list, tuple, set)) else [l]

    
def flatten_json(y):
    """
    this flattens_json obj
    adds begining key to beginning
    """
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
            out[name[:-1]] = x

    flatten(y)
    return out


def returnNonList(val):
    if not isinstance(val, list):
        return val
    if isinstance(val, list):
        if len(val) == 1:
            return val[0]
        else:
            return '|'.join(val)

    
def gather_categorical_col(df, n=10):
    df = df.applymap(returnNonList)
    categorical_col = []
    for column in df.columns:
        if df[column].dtype == object and len(df[column].unique()) <= n:
            categorical_col.append(column)

            print(f"{column} : {df[column].unique()}")
            print("====================================")
    return categorical_col


def make_inv_map(my_map):
    inv_map = {v: k for k, v in my_map.items()}
    return inv_map


def write_jsonl(file_path, lines):
    """
    wrts a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))


def make_generator(file_name, sep=","):
    """this function uses a generator"""
    qlines = (line for line in open(file_name))
    list_line = (s.rstrip().split(sep) for s in lines)
    cols = next(list_line)
    df_dicts = (dict(zip(cols, data)) for data in list_line)
    return df_dicts