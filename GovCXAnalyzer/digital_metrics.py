# digital performance measures
    
import re
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import pandas as pd
from urllib.parse import urlparse
import spacy
from collections import defaultdict

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_lg")

# Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

def analyze_service_specific_metrics(feedback_texts):
    # Dictionary to hold sentiment scores for each service
    service_sentiments = defaultdict(list)

    for text in feedback_texts:
        # Perform NER to identify services mentioned in the text
        doc = nlp(text)
        services = [ent.text for ent in doc.ents if ent.label_ in ('ORG', 'PRODUCT', 'SERVICE')]

        # Calculate sentiment score for the text
        sentiment_score = sia.polarity_scores(text)['compound']

        # Assign sentiment score to identified services
        for service in services:
            service_sentiments[service].append(sentiment_score)

    # Calculate average sentiment for each service
    average_sentiments = {service: sum(scores) / len(scores) for service, scores in service_sentiments.items() if scores}

    return average_sentiments


## nltk version for service specific metrics
def extract_services_and_sentiment(texts):
    sia = SentimentIntensityAnalyzer()
    services_sentiment = {}

    for text in texts:
        sentiment = sia.polarity_scores(text)
        entities = extract_named_entities(text)
        for entity in entities:
            services_sentiment.setdefault(entity, []).append(sentiment)

    return services_sentiment

# generic Named Entity Recognition to identify services - not spacy
def extract_named_entities(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    return [chunk[0] for chunk in chunked if isinstance(chunk, Tree) and chunk.label() == 'NE']

def calculate_normalized_utilization(resource_traffic, total_traffic):
    """
    Calculate the normalized utilization rate for resources.

    :param resource_traffic: Dictionary with resource names as keys and their traffic numbers as values.
    :param total_traffic: Total traffic number for the same period.
    :return: Dictionary with resource names as keys and their normalized utilization rates as values.
    """
    if total_traffic == 0:
        return {resource: 0 for resource in resource_traffic}  # Avoid division by zero

    return {resource: (visits / total_traffic) for resource, visits in resource_traffic.items()}
 
 
 # Extend this list with more keywords/phrases that indicate a request for help
keywords = [
        'please assist', 'having trouble', 'need help', 'support request',
        'help needed', 'assistance required', 'can’t figure out', 'stuck with',
        'issue with', 'problem with', 'trouble with', 'unable to', 'difficulty with',
        'can you help', 'help me with', 'question about', 'inquiry about', 'struggling with',
        'challenged by', 'facing an issue', 'how do i', 'how to', 'there’s a problem',
        'it’s not working', 'doesn’t work', 'can’t access', 'need information on',
        'seeking guidance', 'require assistance', 'technical support', 'customer support',
        'service desk', 'help desk', 'not functioning', 'error with', 'failure with',
        'malfunctioning', 'defective', 'not able to', 'how can i', 'assist me with',
        'guidance on', 'advice on', 'help with', 'troubleshoot', 'fixing', 'resolving'
    ]

def calculate_help_request_rate(texts, keywords=keywords):
    def preprocess_text(text):
        return re.sub(r'\W', ' ', text.lower())

    # Count occurrences of help-related keywords/phrases
    count = 0
    for text in texts:
        clean_text = preprocess_text(text)
        if any(keyword in clean_text for keyword in keywords):
            count += 1

    return count / len(texts) if texts else 0

# Dictionary of keywords for identifying help requests 
example_keywords_forhelp = {
# General Help Requests
"Direct Requests for Help" : ["need help", "assistance required", "can you help", "seeking help", "please assist", "support needed", "help needed", "requesting assistance"],
"Questions Indicating Need for Help": [ "how can I", "what should I do", "can someone explain", "I don't understand", "I'm struggling with", "I'm not sure how to", "could you guide"],
"Descriptive Terms of Difficulty": ["having trouble", "facing an issue", "problem with", "difficulties with", "challenge in", "stuck on", "trouble with", "can't figure out"],
## Technical Help Requests
"Technical Issues": [ "error", "issue", "bug", "problem", "technical support", "not working", "malfunction", "crash", "glitch", "troubleshoot"],
"Software/Hardware Terms": ["software", "application", "app", "system", "device", "hardware", "tool", "program"],

}

# Function to check for help request keywords in a list of messages
def identify_help_requests(messages, keyword_dict):
    flagged_messages = []
    for message in messages:
        for category in keyword_dict:
            for keyword in keyword_dict[category]:
                if re.search(r"\b" + re.escape(keyword) + r"\b", message, re.IGNORECASE):
                    flagged_messages.append(message)
                    break  # Break to avoid flagging the same message multiple times
    return flagged_messages


def get_url_parsed(url_str):
    """parses url into a dictionary"""
    d = {}
    if isinstance(url_str, str):
    
    
        u = urlparse(url_str)
      
        d['netloc'] = u.netloc
        d['path'] = u.path
        d['query'] = u.query
        d['params'] = u.params
        d['fragment'] = u.fragment
        return d
    else:
        return None
    


    