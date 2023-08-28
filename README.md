# GovCXAnalyzer
python based code repository for analyzing customer experience data, notably the A-11 survey data that HISPs collect


### Future
- ydataprofiling
- warning: likely break rest of the environment
- but for the time being, adding as an example in the notebook, but not adding to the notebook

## SM4H - Team **RxSpace** :star: !

## Table of Contents
* [HISP CX Background Details](#HISP-CX-Background-details)
* [Team Members](#team) :sparkles: :sparkles: :email:
* [Our Approach](#our-approach) :bookmark:
* [Requirements](#requirements)
* [Repo Layout](#repo-layout)
* [Text Corpora](#text-corpora) :books: 
* [Embeddings](#embeddings)
* [Sentiment Analysis](#snorkel)
* [Model Training](#model-training)
* [Evaluation](#evaluation) :chart_with_upwards_trend:
* [References](#references)
* [Tags](#tags) 
* [Future Work](#future-work) :crystal_ball:	


## HISP CX Background Details
*This repository contains code for tackling Task 4 of the SMM2020*

The Social Media Mining for Health Applications (#SMM4H) Shared Task involves natural language processing (NLP) challenges of using social media data for health research, including informal, colloquial expressions and misspellings of clinical concepts, noise, data sparsity, ambiguity, and multilingual posts. For each of the five tasks below, participating teams will be provided with a set of annotated tweets for developing systems, followed by a three-day window during which they will run their systems on unlabeled test data and upload the predictions of their systems to CodaLab. Informlsation about registration, data access, paper submissions, and presentations can be found below.
<br>

*Task 4: Automatic characterization of chatter related to prescription medication abuse in tweets* <br>

This new, multi-class classification task involves distinguishing, among tweets that mention at least one prescription opioid, benzodiazepine, atypical anti-psychotic, central nervous system stimulant or GABA analogue, tweets that report potential abuse/misuse (annotated as “A”) from those that report non-abuse/-misuse consumption (annotated as “C”), merely mention the medication (annotated as “M”), or are unrelated (annotated as “U”)3. <br>

#### Timeline
* Training data available: January 15, 2020 (may be sooner for some tasks) <br>
* Test data available: April 2, 2020 <br>
System predictions for test data due: April 5, 2020 (23:59 CodaLab server time) <br>
* System description paper submission deadline: May 5, 2020 <br>
* Notification of acceptance of system description papers: June 10, 2020 <br>
* Camera-ready papers due: June 30, 2020 <br>
* Workshop: September 13, 2020 <br>
* All deadlines, except for system predictions (see above), are 23:59 UTC (“anywhere on Earth”). <br>


## Team
### Team members
* Isabel Metzger - isabel.metzger@gsa.gov <br>
* Ashleigh Sanders - ashleigh.sanders@gsa.gov <br>

## Our Approach
* *Our approach can be broken up into 3 main sections: preprocessing, model architectures, and Ensemble*
* Pre-processing:
    *tokenization + using pre-trained embeddings/ creating our own pre-trained word representations*
    * Word Embeddings:
        * Glove (Pennington et al., 2014) , Word2Vec (Mikolov et al., 2013), fasText (Bojanowski et al., 2016):
            * params:
                * dim: 50, 100, 200, 300

                    
        *  Language Model: Elmo (Perters et al., 2018), Bert , sciBert:
            *     params: default
    * Model Architectures:
        * fasttext baseline
        * allennlp scibert text classifier
        * cnn text classifiers
     
        * train multiple models based on different training-set/val-set, different embeddings, different features, and even totally different architectures
    * we also train with different data-splits
    * *for all splits not using the originally provided train and dev set, we stratify by class e.g., 
        * Data split 1:
            * *utilizing split provided from SMM4H*
            * Train: orig train.csv (N = 10,537)
        * Dev: orig validation.csv (N =2,636)
    * Data split 2:
        * using an 70% | 30% split
        * Train:
        * Dev: 
     * Data split 3:
        * using a holdout from the dev set for 15%
        * Train: 65%
        * Dev:  20%
        * Hold-out:  15%, 
       *     *Hold-out is used to tune the thresholds*
  * Ensemble
  * Voting:
    * Models trined on different splits with weights according to dev set
    * validation metric fine tuned for includes overall_f1 and abuse_f1
    * fixed threshold = 0.5
    * fine-tune threshold according to the hold-out set for unfixed thresshold
    * weight models according to best class f1 on validation


## Requirements
* Important packages/frameworks utilized include [spacy](https://github.com/explosion/spaCy), [fastText](https://github.com/facebookresearch/fastText), [ekphrasis](https://github.com/cbaziotis/ekphrasis), [allennlp](https://github.com/allenai/allennlp), [PyTorch](https://github.com/pytorch/pytorch), [snorkel](https://github.com/snorkel-team/snorkel/)
* 
```bash
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar
tar -xvf scibert_scivocab_uncased.tar
```
* Exact requirements can be found in the requirements.txt file
* For specific processed done in jupyter notebooks, please find the packages listed in the beginning cells of each notebook


## Repo Layout
```
* notebooks - jupyter notebooks including notebooks that contain important steps including structued data, A-11 sopecific data preprocessing, text preprocessing, preprocessing for our huggingface models, snorkel labeling fxns and evaluation/exploratory analysis, and our baseline fasttext model (preprocessing, training, and saving)
* govcxanalyzer - our python code repo with our dataset loaders, functional scripts, predictors, and models
* data_configs - example data configs
* preds - directory with predictions
* data - directory with input data and output data made from preprocessing functions
* docs - more documentation (md and html files)
* saved-models - directory where saved models are
*notebooks - notebook examplesfor spacy and for compiling fasttext library
```

## Text Corpora
   

## Configuration and Model Training

 ```bash

```
```json 

 ```


## Methods
### Unsupervised Learning
* Analogy & similarity

### Sentiment Analysis
### Emotion Text classification

```

```
Out of the box with fasttext.train_supervised(tweets.train)
```bash


```

## Future Work
* Efficiently incorporating more sources:  
* Incorporating 

## Tags
* NLP, FederalCX, open-source, emotion and sentiment analysis, survey analysis