from __future__ import absolute_import
from nltk.tokenize import TweetTokenizer
tok = TweetTokenizer()

def preprocess_text(text, lower=True):
    """takes a tqeet text and tokenizes it"""
    tokens = tok.tokenize(text)
    if lower is True:
        return ' '.join([token.lower() for token in tokens])
    else:
        return ' '.join(tokens)


def listify(l): return l if isinstance(l, (list, tuple, set)) else [l]

def flatten_list(l): return [item for sublist in l for item in sublist]


def get_top_cat(doc):
    """takes a spacy doc object and returns the category with highest score"""
    cats = doc.cats
    max_score = max(cats.values())
    max_cats = [k for k, v in cats.items() if v == max_score]
    max_cat = max_cats[0]
    return (max_cat, max_score)


def matches(input1, input2):
    return (input1 == input2)

