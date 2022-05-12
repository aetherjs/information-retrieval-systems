from src.models.vector_space import search_with_vs
from src.models.bm25 import search_with_bm25
from src.textstatistics.text_statistics import plot_zipf

if __name__ == '__main__':
    plot_zipf()
    search_with_vs()
    search_with_bm25()
