#!/usr/bin/env python
# coding: utf8
"""
Trains a baseline script for convolutional neural network text classifier using out of the box spacy the TextCategorizer component.

be sure to run prepare-for-spacy-and-pytorch.py using a directory with the original train.csv and validation.csv file to an output dir (e./g., "data-spacy-pytorch-jsonl" is used in this case
Usage: python train_spacy_textcategorizer.py <input_dir> <labels>
e.g., python train_spacy_textcategorizer.py -i "data-spacy-pytorch-jsonl" -l "UNEMPLOYED,STUDENT..etc"

* Training: https://spacy.io/usage/training

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
from utilz import listify, matches, get_top_cat
import spacy
from spacy.util import minibatch, compounding
import jsonlines




@plac.annotations(
    input_dir=("Input directory with data folder for train and validation", "option", "i", Path),
    labels=("String with labels that we are predicting for", "option", "l", str),
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int),
    init_tok2vec=("Pretrained tok2vec weights", "option", "t2v", Path),
)
def main(input_dir="data-jsonl", labels="", model=None, output_dir=None, n_iter=20, n_texts=10537, init_tok2vec=None):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.load("en_core_md_lg")
        nlp.add_pipe(nlp.create_pipe("sentencizer"))
        print("Created en core md lg base model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    labels = labels.split(',')
    labels = listify(labels)
    # add label to text classifier
    for label in labels:
        print(f'adding label {label}')
        textcat.add_label(label)

    print("Loading data...")
    (train_texts, train_cats), (dev_texts, dev_cats) = load_data(input_dir, labels=labels)
    train_texts = train_texts[:n_texts]
    train_cats = train_cats[:n_texts]
    print(
        "Using {} examples ({} training, {} evaluation)".format(
            n_texts, len(train_texts), len(dev_texts)
        )
    )
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))
    print(f'sample of training data:\n {train_data[:10]}')

    dev_data = list(zip(dev_texts,
                          [{'cats': cats} for cats in dev_cats]))
    print(f'sample dev:\n {dev_data[:10]}')
    # get names of other pipes to disable them during training
    pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        if init_tok2vec is not None:
            with init_tok2vec.open("rb") as file_:
                textcat.model.tok2vec.from_bytes(file_.read())
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(4.0, 64.0, 1.001)
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print(
                "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                    losses["textcat"],
                    scores["textcat_p"],
                    scores["textcat_r"],
                    scores["textcat_f"],
                )
            )

    # test the trained model
    print("testing trained model")
    example_dev_texts = dev_data[:10]
    for objs in example_dev_texts:
        test_text = objs[0]
        test_cats = objs[1]
        doc = nlp(test_text)
        predicted_cat = get_top_cat(doc)
        true_class = [k for k, v in test_cats.items() if v == True]
        true_class = true_class[0]
        print(f'TEXT:{test_text}\tPRED CLASS:{predicted_cat}\tTRUE CLASS:{true_class}')
    
    test_text = ""
    doc = nlp(test_text)
    print(test_text, doc.cats)

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)




def load_data(input_dir, labels):
    labels = listify(labels)
    training_path = Path(input_dir) / 'train.jsonl'
    # Partition off part of the train data for evaluation
    train_texts = []
    train_cats = []
    with jsonlines.open(training_path) as reader:
        for obj in reader:
            cats = {}
            text = obj['text']
            train_texts.append(text)
            label = obj['label']
            for k in labels:
                cats[k] = matches(k, label)

            train_texts.append(text)
            train_cats.append(cats)

    validation_path = Path(input_dir) / 'validation.jsonl'
    dev_texts = []
    dev_cats = []
    with jsonlines.open(validation_path) as reader:
        for obj in reader:
            labs = {}
            text = obj['text']
            lab = obj['label']
            for k in labels:
                labs[k] = matches(k, lab)
            dev_texts.append(text)
            dev_cats.append(labs)


           # val_data.append(obj)
    return (train_texts, train_cats), (dev_texts, dev_cats)

def get_tuples_data(train_texts, train_cats, dev_texts, dev_cats):
    train_data = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))
    print(train_data[0])
    random.shuffle(train_data)
    print(train_data[0])
    dev_data = list(zip(dev_texts,
                          [{'cats': cats} for cats in dev_cats]))
    
    return train_data, dev_data



def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == 'UNEMPLOYED':
                continue
            if label == 'STUDENT':
                continue
            if label == 'EMPLOYER':
                continue
            if label == 'COUNSELOR':
                continue
            if label == 'SPAM':
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


if __name__ == "__main__":
    plac.call(main)