
from .utils import get_date_timestamp, return_no_str, flatten
import pandas as pd
import torch
from IPython import display
import jsonlines
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoTokenizer, pipeline

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


def initialize_emotion_classifier(topk=3):
    """default returns the top 3 emotions"""
    
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', top_k=topk)
    return classifier


def pred_emotion_classifier(classifier,
                             sample_text="I love using transformers. The best part is wide range of support and its easy to use",
                             verbose=False):

    prediction = classifier(sample_text)
    if verbose:
        print(prediction)
    return prediction



def get_text_emotion_preds(classifier, dfcomments, textcol):
    ts = get_date_timestamp()

    catpredlist = []
    dfcomments=dfcomments[dfcomments[textcol].notna()]
    N = dfcomments.shape[0]
    print(N)

    outpath=f"../{str(ts)}-{str(N)}-{str(textcol).lower().split(' ')[0]}-output-emotion-preds.jsonl"
    catpredlist = []
    with jsonlines.open(outpath, "w") as fout:
        
        for d in dfcomments.to_dict("records"):
            newd = d.copy()
            text = d[textcol]

            try: 
                if len(text) < 10: 
                    emot = classifier("NONE",)
                else:
                    emot = classifier(text,)
            except:
                pass
            
            newd['emot_topk'] = emot
            fout.write(newd)
            catpredlist.append(newd)

    return catpredlist


def turn_catpredlist_to_df(catpredlist):

    commentdfwithpreds = pd.DataFrame(catpredlist)
    
    commentdfwithpreds["emotion_top"] =commentdfwithpreds[
        'emot_topk'
    ].map(lambda x:  flatten(x)[0]if len(x) == 1  else x[0], na_action='ignore').map(lambda x: x.get('label', x))

    commentdfwithpreds["emotion_score"] =  commentdfwithpreds[
        'emot_topk'
    ].map(lambda x:  flatten(x)[0]if len(x) == 1  else x[0], na_action='ignore').map(lambda x: x.get('score', x))
    
    
    return commentdfwithpreds
    

def make_dfonlycomments(df, textcol="Anything else you\'d like to share with us?"):
    df[textcol] = df[textcol].map(return_no_str)
    print(df.shape[0])

    dfonlycomments = df[df[textcol].notna()]
    print(dfonlycomments.shape[0])
    return dfonlycomments



def get_label_val(vl):
    nv = {}
    for v in vl:
        for k, vv in v.items():
            nv[k["label"]] = vv["score"]
            
    return nv


def get_emotion_top_stats(dflistpred):
    N = dflistpred.shape[0]


    dcounts = dflistpred["emotion_top"].value_counts().reset_index().rename(columns={"emotion_top": "counts", "index": "emotion_top",
                   })
    display.display(dcounts)
    return dcounts, N



def create_group_counts_distribution(dflistpred, groupcol):
    
    dflistpred[groupcol] = dflistpred[groupcol].fillna("N/A")

    countsdf = dflistpred[groupcol].value_counts().to_frame().reset_index().rename(columns={"count": "# Responses"
                                                                                            })
    N = countsdf["# Responses"].sum()
    countsdf["% Responses"] = np.round(100*countsdf["# Responses"]/ N)

    return countsdf


def get_emotions_by_groupcol(dflistpred, groupcol):
    
    dflistpred[groupcol] = dflistpred[groupcol].fillna("Did not respond")
    
    countsemotionbygroup = pd.crosstab(
        dflistpred["emotion_top"], dflistpred[groupcol]
        ).reset_index()
    return countsemotionbygroup


negemotions = ["sadness", "anger", "fear"]
posemotions =["joy", "love"]

def filtercommments(dflistpred,emotionlist=posemotions):
    emotion_filtered_comments= (
    dflistpred[dflistpred["emotion_top"].isin(emotionlist)].sort_values(["emotion_score"], ascending=[False])
)
    return emotion_filtered_comments





