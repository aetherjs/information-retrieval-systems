from src.util.util import read_candidates_file
from src.util.util import preprocess_document
from src.util.util import read_test_queries


class InverseIndex:
    def __init__(self, candidates=None):
        self.index = dict()
        self.qid = None
        self.candidates = candidates
        self.parsed_passages = set()
        self.indexed_queries = set()

    def _extract_relevant_passages(self):
        """
        Get PIDs associated with the given QID
        """
        passages = []
        filtered_entries = []
        for candidate in self.candidates:
            if candidate[0] == self.qid:
                filtered_entries.append(candidate)
        for entry in filtered_entries:
            passages.append((entry[1], entry[3]))
        return passages

    def add_query(self, query_id):
        """
        Sets a new qid and augments the index
        """
        self.qid = query_id
        self._build()
        self.indexed_queries.add(query_id)

    def _build(self):
        """
        Construct the inverted index for a current qid
        """
        print("Initiating Index construction for query", self.qid)
        passages = self._extract_relevant_passages()
        for parsed_passage in self.parsed_passages:
            if parsed_passage in passages:
                passages.remove(parsed_passage)

        for passage in passages:
            pid = passage[0]
            tokens = preprocess_document(passage[1], use_stopwords=True)
            for token in tokens:
                if token in self.index:
                    if pid in self.index[token]:
                        self.index[token][pid] = self.index[token][pid] + 1
                    else:
                        self.index[token][pid] = 1
                else:
                    self.index[token] = dict()
                    self.index[token][pid] = 1
            self.parsed_passages.add(passage)
        print("Indexing complete for query", self.qid)


def get_entire_reverse_index():
    data = read_candidates_file()
    queries = read_test_queries()
    index = InverseIndex(candidates=data)
    for query in queries:
        index.add_query(query[0])
    return index


def get_index_for_query(qid):
    data = read_candidates_file()
    index = InverseIndex(candidates=data)
    index.add_query(qid)
    return index
