# Standard Library Imports
import pandas as pd
import glob
import string
import seaborn as sns
import datetime
import warnings
import requests
from IPython import display
from collections import Counter, defaultdict
import unicodedata
import re
import numpy as np
 # load stop words
import en_core_web_md
import PyLDAVis

# load spacy nlp model
web = en_core_web_md.load()

# NLP Imports
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk import FreqDist
import nltk
from nrclex import NRCLex
 
ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
NON_ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENGLISH_STOPWORDS
ENGLISH_STOPWORDS.update([' ' ,'s',"/", "'s","also", "e.g.",])
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}
 
def get_language(text):
    words = set(nltk.wordpunct_tokenize(text.lower()))
    return max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key = lambda x: x[1])[0]

def is_english(text):
    text = text.lower()
    words = set(nltk.wordpunct_tokenize(text))
    return len(words & ENGLISH_STOPWORDS) > len(words & NON_ENGLISH_STOPWORDS)

def get_nrc_emotions(tokens):
    l = []
    for i in range(len(tokens)):

        emotion = NRCLex(tokens[i])
        
   # Classify emotion
        l.append({'token': tokens[i], 'emotions': emotion.top_emotions})
    
    return l




# # Iterate through list
# for i in range(len(text)):

#    # call by object creation
#    emotion = NRCLex(text[i])

#    # Classify emotion
#    print('\n', text[i], ': ', emotion.top_emotions) 


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, MiniBatchNMF, LatentDirichletAllocation

# n_samples = 2000
# n_features = 1000
# n_components = 10
# n_top_words = 20
# batch_size = 128
# init = "nndsvda"


# def plot_top_words(model, feature_names, n_top_words=20, title):
#     fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
#     axes = axes.flatten()
    
#     dl = []
    
    
#     for topic_idx, topic in enumerate(model.components_):
#         d = {}
        
#         top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
#         top_features = [feature_names[i] for i in top_features_ind]
#         weights = topic[top_features_ind]
#         d['label'] =f"Topic {topic_idx +1}"
#         d['weights'] = weights
#         d['top_features'] =top_features
#         dl.append(d)
#         ax = axes[topic_idx]
#         ax.barh(top_features, weights, height=0.7)
#         ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
#         ax.invert_yaxis()
#         ax.tick_params(axis="both", which="major", labelsize=20)
#         for i in "top right left".split():
#             ax.spines[i].set_visible(False)
#         fig.suptitle(title, fontsize=40)

#     plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
#     plt.show()
    
#     return pd.DataFrame(dl)


    


def loading_dataset(ltexts):
    print("Loading dataset...")
    t0 = time()

    data_samples = ltexts
    print("done in %0.3fs." % (time() - t0))
    
def get_top_cat(doc):
    """takes a spacy doc object and returns the category with highest score"""
    cats = doc.cats
    max_score = max(cats.values())
    max_cats = [k for k, v in cats.items() if v == max_score]
    max_cat = max_cats[0]
    return (max_cat, max_score)

def pipe_ents(entities, label,textcol='text', entitycol='label'):
    """entities is a list"""
    return ' | '.join(list(set([ent[textcol] for ent in entities if ent[entitycol] == label])))



    

import re

import numpy as np
from nltk import sent_tokenize, word_tokenize

from nltk.cluster.util import cosine_distance

MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)


def normalize_whitespace(text):
    """
    Translates multiple whitespace into single space character.
    If there is at least one new line character chunk is replaced
    by single LF (Unix new line) character.
    """
    return MULTIPLE_WHITESPACE_PATTERN.sub(_replace_whitespace, text)


def _replace_whitespace(match):
    text = match.group()

    if "\n" in text or "\r" in text:
        return "\n"
    else:
        return " "


def is_blank(string):
    """
    Returns `True` if string contains only white-space characters
    or is empty. Otherwise `False` is returned.
    """
    return not string or string.isspace()


def get_symmetric_matrix(matrix):
    """
    Get Symmetric matrix
    :param matrix:
    :return: matrix
    """
    return matrix + matrix.T - np.diag(matrix.diagonal())


def core_cosine_similarity(vector1, vector2):
    """
    measure cosine similarity between two vectors
    :param vector1:
    :param vector2:
    :return: 0 < cosine similarity value < 1
    """
    return 1 - cosine_distance(vector1, vector2)



class TextRank4Sentences():
    def __init__(self):
        self.damping = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 100  # iteration steps
        self.text_str = None
        self.sentences = None
        self.pr_vector = None

    def _sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return core_cosine_similarity(vector1, vector2)

    def _build_similarity_matrix(self, sentences, stopwords=None):
        # create an empty similarity matrix
        sm = np.zeros([len(sentences), len(sentences)])

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                sm[idx1][idx2] = self._sentence_similarity(sentences[idx1], sentences[idx2], stopwords=stopwords)

        # Get Symmeric matrix
        sm = get_symmetric_matrix(sm)

        # Normalize matrix by column
        norm = np.sum(sm, axis=0)
        sm_norm = np.divide(sm, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return sm_norm

    def _run_page_rank(self, similarity_matrix):

        pr_vector = np.array([1] * len(similarity_matrix))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr_vector = (1 - self.damping) + self.damping * np.matmul(similarity_matrix, pr_vector)
            if abs(previous_pr - sum(pr_vector)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr_vector)

        return pr_vector

    def _get_sentence(self, index):

        try:
            return self.sentences[index]
        except IndexError:
            return ""

    def get_top_sentences(self, number=5):

        top_sentences = []

        if self.pr_vector is not None:

            sorted_pr = np.argsort(self.pr_vector)
            sorted_pr = list(sorted_pr)
            sorted_pr.reverse()

            index = 0
            for epoch in range(number):
                sent = self.sentences[sorted_pr[index]]
                sent = normalize_whitespace(sent)
                top_sentences.append(sent)
                index += 1

        return top_sentences

    def analyze(self, text, stop_words=None):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)

        tokenized_sentences = [word_tokenize(sent) for sent in self.sentences]

        similarity_matrix = self._build_similarity_matrix(tokenized_sentences, stop_words)

        self.pr_vector = self._run_page_rank(similarity_matrix)


def save_pyldavis(vis_data, outpath):
    PyLDAVis.save_html(vis_data, outpath)

    
from spacy import displacy

# def make_text
# # for all statements when I ranked them/ 
# text_str = '\n'.join([i +': ' + z for i, z in zip(
#     textlist1, textlist2
#                                            )]
#                     )
# tr4sh = TextRank4Sentences()
# tr4sh.analyze(text_str)
# t5 = tr4sh.get_top_sentences(7)


# for i,t in enumerate(t5[1:]):
    
#     displacy.render(web(t + '\n\n'), 'ent')

import nltk
 
ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))
NON_ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words()) - ENGLISH_STOPWORDS
 
STOPWORDS_DICT = {lang: set(nltk.corpus.stopwords.words(lang)) for lang in nltk.corpus.stopwords.fileids()}
 
def get_language(text):
    words = set(nltk.wordpunct_tokenize(text.lower()))
    return max(((lang, len(words & stopwords)) for lang, stopwords in STOPWORDS_DICT.items()), key = lambda x: x[1])[0]

def is_english(text):
    text = text.lower()
    words = set(nltk.wordpunct_tokenize(text))
    return len(words & ENGLISH_STOPWORDS) > len(words & NON_ENGLISH_STOPWORDS)
