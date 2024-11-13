import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
import re
import pandas as pd
import numpy as np 
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from wordcloud import WordCloud
from sklearn.cluster import KMeans, SpectralClustering
from functools import partial
import requests

nltk.download("stopwords")
pd.options.mode.copy_on_write = True

def clean_text(s, stop_words = stopwords.words('english')):
    # remove punctuation and set string to lower case
    s = re.sub(r'[^\w\s]', ' ', str(s)).lower()
    s = s.replace('_', '')

    # remove numbers (digits)
    s = re.sub(r'\d+', ' ', s)

    # remove stopwords except word no. remove .com .gov
    stop_words += ['com', 'gov', 'https']
    stop_words = [w for w in stop_words if w != 'no']
    s = ' '.join([w for w in s.split() if w not in stop_words  ])
    return s
    
# spectral clustering algorithm from Shi and Malik (2000)
class SpecClustering:

    def __init__(self, algo='sklearn', min_doc_tfidf=50, max_doc_tfidf=0.25, min_ngram=1, max_ngram=3, stop_words=None, 
                 addl_stopwords= None, remove_stopwords=None, num_clusters=None, num_docs_summarize=10, 
                 ai_url = None, ai_cluster_descriptions=False, seednum=321):
        
        self.min_doc_tfidf = min_doc_tfidf
        self.max_doc_tfidf = max_doc_tfidf
        if stop_words is None:
            nltk.download("stopwords")
            stop_words = set(stopwords.words("english"))
        if addl_stopwords is not None:
            stop_words = stop_words.union(set(addl_stopwords))
        if remove_stopwords is not None:
            stop_words.discard(set(remove_stopwords))
        self.stopwords = list(stop_words)
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.num_clusters = num_clusters
        self.algo = algo
        self.num_docs_summarize = num_docs_summarize
        self.seednum = seednum
        self.ai_cluster_descriptions = ai_cluster_descriptions
        self.ai_url = ai_url

    def fit(self, X, y=None, **kwargs):
        self.X = X
        self.create_design_mx(**kwargs)

    def fit_predict(self, X, y=None, **kwargs):
        self.X = X
        self.create_design_mx(**kwargs)
        self.create_similarity_matrix(self.Xtrans)
        self.eigendecomposition(self.csim_mx)
        if self.num_clusters is None:
            self.find_num_clusters_eigengap()
        self.spclus_algorithm()
        self.summarize_clusters()
    
    def create_similarity_matrix(self, Xtrans):
        csim_mx = cosine_similarity(Xtrans)
        self.csim_mx = csim_mx
        
    def create_design_mx(self, **kwargs):
        X = self.X
        l = []
        coln = X.columns
    
        for c in coln:
            features_temp_pd = self.create_features_from_series(X[c], **kwargs)
            features_temp_pd.columns = [c + '_' + s for s in features_temp_pd.columns]
            l.append(features_temp_pd) 
    
        design_mx_pd = pd.concat(l, axis=1, ignore_index=False)
        self.Xtrans = design_mx_pd

    def create_features_from_series(self, x):
        min_doc_count = self.min_doc_tfidf
        max_doc_count = self.max_doc_tfidf
        min_ngram = self.min_ngram
        max_ngram = self.max_ngram
        
        clean_custom = partial(clean_text, stop_words=self.stopwords)
        x_cleaned = x.map(clean_custom)
        tfidf = TfidfVectorizer(min_df=min_doc_count, max_df=max_doc_count, norm='l2', ngram_range=(min_ngram, max_ngram))
        tfidf.fit_transform(x_cleaned)
        feature_names = list(tfidf.get_feature_names_out())
    
        l_name = []
        l_vals = []
        for feat in feature_names:
            x_bernoulli_temp = np.log1p(x_cleaned.str.count(feat))
            new_feature_name = re.sub(r"\s", "_", feat)
            l_name.append(new_feature_name)
            l_vals.append(x_bernoulli_temp)
    
        out_pd = pd.concat(l_vals, axis=1, ignore_index=True)
        out_pd.columns = l_name
    
        return out_pd

    def eigendecomposition(self, A):
        algo = self.algo
        # generalized eigendecomposition on similarity matrix A
        D = np.diag(np.sum(A, axis=1))
        if algo == 'shi':
            # shi malik computer generalized eigenvectors
            L = laplacian(A, normed=False)
            lbda, U = eigh(L, b=D)
        else:
            # ng, jordan, weiss compute normalized laplacian
            L = laplacian(A, normed=True)
            # U returned is normalized
            lbda, U = eigh(L)

        self.L = L
        self.D = D
        self.lbda = lbda
        self.U = U        

    def sort_eigenvalues(self, lbda, U):
        idx = np.argsort(lbda)
        lbda = lbda[idx]
        U = U[:, idx]
        return lbda, U

    def find_num_clusters_eigengap(self):
        # compute laplacian unnormalized matrix from csim. Note that this is equivalent to normalized Lrw
        U = self.U
        lbda = self.lbda
        # sort by eigenvalues ascending
        lbda, U = self.sort_eigenvalues(lbda, U)
        
        self.num_clusters_eigengap = np.argmax(np.diff(lbda[1:])) +2
        if self.num_clusters is None:
            self.num_clusters = self.num_clusters_eigengap

    def spclus_algorithm(self):
        algo = self.algo
        U = self.U
        lbda = self.lbda
        lbda, U = self.sort_eigenvalues(lbda, U)
        seednum = self.seednum
        
        num_clusters = self.num_clusters
        if algo == 'sklearn':
            csim_mx = self.csim_mx
            clus = SpectralClustering(affinity='precomputed',
                                     n_clusters=num_clusters,
                                     random_state=seednum)
            clus.fit(csim_mx)
            self.k = clus.labels_ + 1
        else:
            km = KMeans(n_clusters=num_clusters, random_state=seednum)
            Usub = U[:, :num_clusters]
            self.k =  km.fit_predict(Usub) + 1

    def summarize_clusters(self):
        """
        requires API endpoint to summarize clusters
        """

        if not self.ai_cluster_descriptions:
            return

        if not self.ai_url:
            return
        
        url = self.ai_url

        headers = {
            "Content-Type": "application/x-text"
        }
        if self.X is None:
            return
        X = self.X
        X['k'] = self.k
        num_docs_summarize = self.num_docs_summarize
        
        clus_nums = np.unique(self.k)
        summ_dict = {}
        for k in clus_nums:
            temp = X.loc[X.k == k]
            q5_series = temp['q5'].sample(num_docs_summarize, random_state=1)
            #data = temp['q5'][:num_docs_summarize].str.cat(sep = " .")
            data = q5_series.str.cat(sep = " .")
            response = requests.post(url, headers=headers, data=data)
            summ_dict[k] = response.json()['summary_text']

        self.summ_dict = summ_dict
    

    def plot(self, ptype='word', num_eigenvalues=40):
        if ptype == 'word':
            X = self.X
            num_clusters = self.num_clusters
            clean_custom = partial(clean_text, stop_words=self.stopwords)
            X['combined_strings'] = X.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            X['combined_strings'] = X['combined_strings'].map(clean_custom)
            X['k'] = self.k
            
            _, axes = plt.subplots(num_clusters // 2, num_clusters // 5 + 1, figsize = (5 * (num_clusters // 2), num_clusters // 2 * 4))
            clus = 'k'
            num_records_pd = X[clus].value_counts().reset_index()
                    
            for ax, cluster_id in zip(axes.flatten(), np.sort(X[clus].unique())):
                text = " ".join(X[X[clus] == cluster_id]['combined_strings'])
                num = num_records_pd[num_records_pd[clus] == cluster_id]['count'].values[0]
                wordcloud = WordCloud(width=800, height=400, background_color='black', colormap="Set3").generate(text)
                ax.imshow(wordcloud, interpolation='bilinear')
            
                ax.set_title(f"Word Cloud for Cluster {cluster_id}.\nNum Scripts = {num}", fontsize=30)
                ax.axis("off")
            plt.tight_layout()
        elif ptype == 'eigenvalues':
            plt.figure(figsize=(14, 8))
            U = self.U
            lbda = self.lbda
            plt.plot(range(num_eigenvalues), lbda[:num_eigenvalues], marker='o')
            plt.title(f'Eigengap heuristic: number of clusters {self.num_clusters_eigengap}')
        else:
            raise ValueError("ptype must be word or eigenvalues")
        plt.show()
