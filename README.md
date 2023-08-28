# GovCXAnalyzer
python based code repository for analyzing customer experience data, notably the A-11 survey data that HISPs collect

## HISP CX Analysis - Team **GovCXAnalyzer** :star: !

## Table of Contents
* [HISP CX Background Details](#HISP-CX-Background-details)
* [Project Members](#team) :sparkles: :sparkles: :email:
* [Our Approach](#our-approach) :bookmark:
* [Requirements](#requirements)
* [Repo Layout](#repo-layout)
* [Datasets](#text-corpora) :books: 
* [Unsupervised Learning](#Unsupervised Learning)
* [Sentiment Analysis](#snorkel)
* [Text Classification](#model-training)
* [Visualzation & Statistical Analysis Methods](#evaluation) :chart_with_upwards_trend:
* [References](#references)
* [Tags](#tags) 
* [Future Work](#future-work) :crystal_ball:	


## HISP CX Background Details
*This is the code repository for*

*add details - general description*



### Project Team
* Isabel (Izzy) Metzger - isabel.metzger@gsa.gov <br>
* Ashleigh Sanders - ashleigh.sanders@gsa.gov <br>

## TO DO: add thank you to testers
* 

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
## need to add additonal info


## Requirements
* Important packages/frameworks utilized include [sklearn](), [ydata-profiling](), [spacy](https://github.com/explosion/spaCy), [ekphrasis](https://github.com/cbaziotis/ekphrasis), [allennlp](https://github.com/allenai/allennlp), [PyTorch](https://github.com/pytorch/pytorch)


## Create your environment either through conda or python venv
```bash
## for conda
conda create -n cxenv python=3.9
# install requirements
pip install -r requirements.txt
python -m spacy download en_core_web_md
```
```bash
pip install virtualenv
cd GovCXAnalyzer
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
```
* Exact requirements can be found in the requirements.txt file

## Repo Layout
```
* notebooks - jupyter notebooks including notebooks that contain important steps including structued data, A-11 sopecific data preprocessing, text preprocessing, preprocessing for our huggingface models, snorkel labeling fxns and evaluation/exploratory analysis, and our baseline fasttext model (preprocessing, training, and saving)
* govcxanalyzer - our python code repo with our dataset loaders, functional scripts, predictors, and models
* constants - example data configs - change to fit for your data
* data - directory with input data and output data made from preprocessing functions
*  outputs - directory where saved models/ visualizations/ processed data tables / sentiment predictions are located
*notebooks - notebook examples for applying exploratory data visualization, survey data visualization, 
```

## TO DO: Datasets
   

# TO DO
## Methods
### Unsupervised Learning
* Analogy & similarity

### Sentiment Analysis
### Emotion Text classification

```bash

```

```bash
## add examples for running python non-notebook

```

## Future Work
* Efficiently incorporating more sources:  
* ..
## Tags
* NLP, FederalCX, open-source, emotion and sentiment analysis, survey analysis