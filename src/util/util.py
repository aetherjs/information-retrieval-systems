from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tabulate
import csv
import contractions
import re


OUTPUT_PATH = "../results/"
DATASET_PATH = "../dataset"


def clean_text(text):
    text=re.sub('\w*\d\w*','', text)
    text=re.sub('[^a-z]',' ',text)
    return text


def preprocess_document(raw_data, use_stopwords=True):
    """
    A function that takes raw document string as an input
    and returns preprocessed list of tokens
    """

    # Set English stopwords dictionary
    stopwords_set = set(stopwords.words('english'))

    # Use NLTK lemmatizer, and opt for lemmatizing instead of stemming
    lemmatizer = WordNetLemmatizer()

    # Use RegEx Tokenizer to also remove punctuation and digits
    tokenizer = RegexpTokenizer(r'[a-z]+|\d+')
    raw_tokens = tokenizer.tokenize(raw_data)

    clean_tokens = []

    if use_stopwords:
        for token in raw_tokens:
            if token not in stopwords_set:
                clean_token = contractions.fix(token)
                clean_token = lemmatizer.lemmatize(clean_token, pos="a")
                clean_tokens.append(clean_token)
    else:
        for token in raw_tokens:
            clean_token = contractions.fix(token)
            clean_token = lemmatizer.lemmatize(clean_token, pos="a")
            clean_tokens.append(clean_token)

    return clean_tokens


def flush_to_file(ranking, filename):
    ranked = []
    rank = 1
    for entry in ranking:
        ls = list(entry)
        ls.insert(3, rank)
        rank += 1
        ranked.append(ls)

    table = tabulate.tabulate(ranked, tablefmt="plain")
    with open(OUTPUT_PATH + filename, "a") as f:
        f.write(table + "\n")






def get_vocab(corpus):
    """
    Takes a corpus (list of documents as raw strings) and returns the set of unique tokens
    contained in this corpus (vocabulary)
    """
    vocab = set()
    for document in corpus:
        vocab.update(preprocess_document(document))
    return vocab


def read_candidates_file():
    """
    Reads the candidate_passages_top1000 tsv file and returns a list of lists
    of type [QID, PID, query, passage]
    """
    data = list()
    with open(DATASET_PATH + "/candidate_passages_top1000.tsv") as tsv:
        for line in csv.reader(tsv, delimiter="\t"):
            data.append(line)
    return data


def read_test_queries():
    """
    Reads the test-queries.tsv file and returns a list of QIDs
    """
    queries = list()
    with open(DATASET_PATH + "/test-queries.tsv") as tsv:
        for line in csv.reader(tsv, delimiter="\t"):
            queries.append((line[0], line[1]))
    return queries


