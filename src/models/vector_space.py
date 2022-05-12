from src.util.util import read_test_queries
from src.util.util import read_candidates_file
from src.util.util import flush_to_file
from src.indexing.inverse_index import InverseIndex
from src.util.tf_idf import get_tf_idf  # Results were wrong ish - what went wrong?
from src.util.tf_idf import cosine_similarity
import numpy as np


def search_with_vs():
    query_list = read_test_queries()
    candidates = read_candidates_file()
    for counter, query in enumerate(query_list):
        print("VS Query #", counter + 1)

        # Create an inverted index for a query
        index = InverseIndex(candidates)
        index.add_query(query[0])

        # Extract relevant documents for a query
        documents = index.parsed_passages

        # A list which will keep track of document similarity values
        ranking = []

        # Compute TF-IDF representation of a query
        query_vector = get_tf_idf(query[1], index)

        for n, document in enumerate(documents):
            document_vector = get_tf_idf(document[1], index)
            query_scores = np.array(query_vector[:, 1], dtype=float)
            doc_scores = np.array(document_vector[:, 1], dtype=float)
            similarity = cosine_similarity(query_scores, doc_scores)
            ranking.append((query[0], "A1", document[0], similarity, "VS"))

        ranking.sort(key=lambda entry: entry[3], reverse=True)
        flush_to_file(ranking[:100], "VS.txt")


