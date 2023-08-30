
import re
import pandas as pd
from nltk.tokenize import TweetTokenizer, sent_tokenize, word_tokenize
from gensim import corpora
from gensim.models import LdaModel
import unicodedata
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.stem import PorterStemmer

stop_words = stopwords
import warnings
warnings.filterwarnings('ignore')

def to_string(s):
    try:
        return str(s)
    except:
        #Change the encoding type if needed
        return s.encode('utf-8')

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        )

def normalizeString(s):
    tknzr = TweetTokenizer()
    s = " ".join(tknzr.tokenize(s))
    s = re.sub(r"\d", "d", s)
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?]+)", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?-]+", r" ", s)
    s = to_string(s)

    return s


def preProc(text):
    """
    text is a string, for example: "Please keep humira refrigeraterd.
    """

    text2 = normalizeString(text)

    tokens = [word for sent in sent_tokenize(text2) for word in
          word_tokenize(sent)]

    tokens = [word.lower() for word in tokens]

    stopwds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwds]

    tokens = [word for word in tokens if len(word) >= 3]

    stemmer = PorterStemmer()
    try:
        tokens = [stemmer.stem(word) for word in tokens]

    except:
        tokens = tokens

    tagged_corpus = pos_tag(tokens)

    Noun_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    Verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

    lemmatizer = WordNetLemmatizer()


    def pratLemmatiz(token, tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token, 'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token, 'v')
        else:
            return lemmatizer.lemmatize(token, 'n')


    pre_proc_text = " ".join([pratLemmatiz(token, tag) for token, tag in tagged_corpus])

    return pre_proc_text


def bigram_preprocess(tokens, deacc=True, lowercase=True, errors='ignore',
    stemmer=None, stopwords=None):
    """
    Convert a document into a list of tokens.
    Split text into sentences and sentences into bigrams.
    the bigrams returned are the tokens
    """
    bigrams = []

    if len(tokens) >1:
        for i in range(0,len(tokens)-1):
            yield tokens[i] + '_' + tokens[i+1]


def create_lda_from_df_textcol(df, textcol):
                               
    
    documents = df[textcol].dropna().apply(lambda x: preProc(str(x))).tolist()
    
    texts = [word_tokenize(doc)  for doc in documents]
    texts_lower = [[word.lower() for word in text] for text in texts]
    unigrams = [[word for word in text if not word.isdigit() and word not in stop_words and len(word) > 1] for text in texts_lower]

    bigrams = [[bigram for bigram in bigram_preprocess(text)] for text in unigrams]

    dictionary_bigrams = corpora.Dictionary(bigrams, prune_at=2000)
    dictionary_bigrams.save_as_text('gensim_dict_bigrams.txt')

    dictionary_unigrams = corpora.Dictionary(unigrams, prune_at=2000)
    dictionary_unigrams.save_as_text('gensim_dict_unigrams.txt')

    corpus = [dictionary_bigrams.doc2bow(text) for text in bigrams]
    lda = LdaModel(corpus, num_topics=15, id2word=dictionary_bigrams)

    return lda
