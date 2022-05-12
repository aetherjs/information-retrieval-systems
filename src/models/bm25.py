from src.indexing.inverse_index import InverseIndex
from _collections import defaultdict
from src.util.util import read_test_queries, read_candidates_file, preprocess_document, get_vocab, flush_to_file
from src.util.tf_idf import tf
import numpy as np

# Constants
K_1 = 1.2
K_2 = 10
B = 0.75


def get_avg_doc_length(corpus):
    sum_length = 0
    for document in corpus:
        processed = preprocess_document(document)
        sum_length += len(processed)
    return sum_length / len(corpus)


def calculate_capital_k(dl, avdl):
    return K_1 * ((1 - B) + B * dl / avdl)


def get_bm25_score(query, document, corpus, index):
    document = preprocess_document(document)
    query = preprocess_document(query)

    dl = len(query)
    avdl = get_avg_doc_length(corpus)
    N = len(corpus)
    K = calculate_capital_k(dl, avdl)

    score = 0
    f_counter = tf(document)
    for token in query:
        # Compute BM25 formula parameters
        n = len(index.index[token])
        f = f_counter[token]
        idf = np.log((N - n + 0.5)/(n + 0.5) + 1)
        d = len(document)
        score += idf * (f * (K_1 + 1)) / (f + K_1 * (1 - B + B * (d/avdl)))

    return score


def search_with_bm25():
    query_list = read_test_queries()
    candidates = read_candidates_file()
    for counter, query in enumerate(query_list):
        print("BM25 Query #", counter + 1)

        # Create an inverted index for a query
        index = InverseIndex(candidates)
        index.add_query(query[0])

        # A list which will keep track of document similarity values
        query_scores = dict()

        relevant_documents = index.parsed_passages

        corpus = []
        for doc in relevant_documents:
            corpus.append(doc[1])

        count = 1
        for docid, passage in relevant_documents:
            count += 1
            print("Doc #", count)
            score = get_bm25_score(query[1], passage, corpus, index)
            if docid in query_scores:
                query_scores[docid] += score
            else:
                query_scores[docid] = score

        ranking = []

        for key, value in query_scores.items():
            temp = (query[0], "A1", key, value, "BM25")
            ranking.append(temp)

        ranking.sort(key=lambda entry: entry[3], reverse=True)
        flush_to_file(ranking[:100], "BM25.txt")


search_with_bm25()




