from src.util.util import get_vocab
from src.util.util import preprocess_document
import numpy as np
from collections import Counter


def tf(document):
    """
    Takes a document (tokenized, as a list of tokens) and a word and returns the
    number of occurrences of that token in the document
    """
    token_frequencies = Counter(document)
    return token_frequencies


def idf(token, corpus, index):
    # Calculate inverse document frequency for a token and corpus
    total_docs = len(corpus)
    if token in index.index:
        relevant_docs = len(index.index[token])
    else:
        relevant_docs = 0
    value = np.log(total_docs / (relevant_docs + 1) + 1)
    return value


def get_tf_idf(document, index):
    # Preprocess the document
    processed_doc = preprocess_document(document)

    # Extract corpus - a list of all participating documents
    corpus = []
    for doc in index.parsed_passages:
        corpus.append(doc[1])

    # Get corpus vocabulary and covert it to a map {word, index}
    vocab = get_vocab(corpus)
    vocab_map = {w: i for i, w in enumerate(vocab)}

    n_tokens = len(processed_doc)

    tf_idf_vector = np.zeros(len(vocab_map))

    tf_counter = tf(processed_doc)

    for token in processed_doc:
        i = vocab_map[token]
        tf_idf_vector[i] = tf_counter[token] / n_tokens
        tf_idf_vector[i] *= idf(token, corpus, index)

    return np.column_stack((np.array(list(vocab)), tf_idf_vector))


def cosine_similarity(a, b):
    similarity = np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))
    return similarity
