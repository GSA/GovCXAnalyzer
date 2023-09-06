# GovCXAnalyzer
python based code repository for analyzing customer experience data, notably the A-11 survey data that HISPs collect

## HISP CX Analysis - Team **GovCXAnalyzer** :star: !

## Table of Contents
* [HISP CX Background Details](#HISP-CX-Background-Details)
* [Project Team](#project-team) :sparkles: :sparkles: :email:
* [Our Approach](#our-approach) :bookmark:
* [Requirements](#requirements)
* [Repo Layout](#repo-layout)
* [Datasets](#text-corpora) :books: 
* [Unsupervised Learning](#unsupervised-learning)
* [Sentiment Analysis](#snorkel)
* [Text Classification](#model-training)
* [Visualization & Statistical Analysis Methods](#evaluation) :chart_with_upwards_trend:
* [References](#references)
* [Tags](#tags) 
* [Future Work](#future-work) :crystal_ball:	


## HISP CX Background Details
*This is the code repository for analyzing survey data, in particular the A-11 data. This code repository equips users with a suite of statistical methods for analyzing survey data, including descriptive statistics, hypothesis testing, and data visualization tools, in addition to natural language processing and text analysis tools for thematic analysis and sentiment analysis.


### Project Team
* Isabel (Izzy) Metzger - isabel.metzger@gsa.gov <br>
* Ashleigh Sanders - ashleigh.sanders@gsa.gov <br>

## TO DO: add thank you to testers
* What insights can we glean about HISP performance and public sentiment from the CX response data that GSA collects? 
* What insights can we uncover about the response-level touchpoints data responses? 
* What statistical analyses are most meaningful for understanding the CX response data? How can these statistical functions be best captured in a reusable tool kit?
*  

Does the available CX data offer a more meaningful performance metric for HISPs? What about if we link it with financial data?
How could A-11 be redesigned to better capture meaningful CX data from the public?


## Our Approach
* 
## need to add additonal info


## Requirements
* Important packages/frameworks utilized include [sklearn](), [ydata-profiling](), [spacy](https://github.com/explosion/spaCy), [ekphrasis](https://github.com/cbaziotis/ekphrasis), [allennlp](https://github.com/allenai/allennlp), [PyTorch](https://github.com/pytorch/pytorch)


## Create your environment either through conda or python venv
```bash
## change directory to this repository - make sure you are in this repository
cd GovCXAnalyzer
## for conda
conda create -n cxenv python=3.9
# conda activate
conda activate cxenv
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
* constants - example data configs - change to fit for your data, keep all you
* data - directory with input data and output data made from preprocessing functions
*  outputs - directory where saved models/ visualizations/ processed data tables / sentiment predictions are located
*notebooks - notebook examples for applying exploratory data visualization, survey data visualization, 
```

## Datasets
## Unsupervised Learning
* Analogy & similarity

### Topic Modeling
### Text similarity Ranking
## Sentiment Analysis
### Lexicon-Based
### Aspect Level Sentiment Extraction
### Emotion Text classification


## Future Work
* Efficiently incorporating more sources:  
* ..
## Tags
* NLP, FederalCX, open-source, emotion and sentiment analysis, survey analysis