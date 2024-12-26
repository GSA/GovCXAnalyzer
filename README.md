# GovCXAnalyzer
python based code repository for analyzing customer experience data, notably the solicited customer feedback data such as A-11 survey data that HISPs collect, and also unsolicited customer feedback data such as emails, chatbot dialogues, and more.

## HISP CX Analysis - Team **GovCXAnalyzer** :star: !

## Table of Contents
* [HISP CX Background Details](#HISP-CX-Background-Details)
* [Project Team](#project-team) :sparkles: :sparkles: :email:
* [Requirements](#requirements)
* [Repo Layout](#repo-layout)
* [Tags](#tags) 

## HISP CX Background Details
*This is the code repository for analyzing survey data, in particular the A-11 data, as well as provides tools to enable users to systematically process and analyze these diverse data streams  unsolicited customer experience (CX) data such as emails, chats, and search queries. This code repository equips users with a suite of statistical methods for analyzing survey data, including descriptive statistics, hypothesis testing, and data visualization tools, in addition to natural language processing and text analysis tools for thematic analysis and sentiment analysis. We also provide our text classifier, built for categorizing emails by customer user groups as defined by CareerOneStop. The code also supports calculating metrics such as help request rates and normalized resource utilization rates. 

### Project Team
* Isabel (Izzy) Metzger - isabel.metzger@gsa.gov <br>
* Kai Cobb - kai.cobb@gsa.gov <br>
* Ashleigh Sanders - ashleigh.sanders@gsa.gov <br>
### Contributors
* Samual Thomas - samual.thomas@gsa.gov <br>

## We aimed to explore
* What insights can we glean about HISP performance and public sentiment from the CX response data that GSA collects? 
* What insights can we uncover about the response-level touchpoints data responses? 
* What statistical analyses are most meaningful for understanding the CX response data? How can these statistical functions be best captured in a reusable tool kit?



## Requirements
* Important packages/frameworks utilized include [sklearn](https://scikit-learn.org/stable/), [ydata-profiling](https://docs.profiling.ydata.ai/latest/), [spacy](https://github.com/explosion/spaCy), [ekphrasis](https://github.com/cbaziotis/ekphrasis), [allennlp](https://github.com/allenai/allennlp), [PyTorch](https://github.com/pytorch/pytorch), [BERTopic](https://maartengr.github.io/BERTopic/getting_started/quickstart/quickstart.html#fine-tune-topic-representations), [causalimpact](https://pypi.org/project/causalimpact/), [statsmodels](https://www.statsmodels.org/stable/index.html), [tslearn](https://tslearn.readthedocs.io/en/stable/)


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
*notebooks - notebook examples for applying exploratory data visualization, survey data visualization
```

### In addition to exploratory data analysis and statistical tests, additional methods this code repository has cover unsupervised and supervised machine learning techniques such as
* Topic Modeling
* Text similarity Ranking
* Sentiment Analysis (both Lexicon-Based and emotion text classification)
We also provide methods such as: 
* Dynamic Time Warping (DTW) to cluster conversations by their sentiment patterns

## Tags
* NLP, FederalCX, open-source, emotion and sentiment analysis, survey analysis
