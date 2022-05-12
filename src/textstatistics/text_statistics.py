from src.util.util import preprocess_document
from collections import Counter
import matplotlib.pyplot as plt
import statistics


DATASET_PATH = "../dataset"


def get_frequencies(terms, top_n=None):
    """
    A function that takes a list of preprocessed tokens and returns the list of tuples,
    where each tuple is a token and its corresponding frequency. If top_n argument is omitted,
    returns all terms and their frequencies, otherwise returns top_n most common terms
    """

    token_frequencies = Counter(terms)
    return token_frequencies.most_common(top_n)


def plot_zipf():
    """
    Takes the list of tuples of type (token, # of occurrences) carries out the necessary calculations,
    and plots
    """
    collection = open(DATASET_PATH + "passage_collection_new.txt")
    tokens = preprocess_document(collection)
    frequencies = get_frequencies(tokens)

    actual_probabilities = []
    zipf_probabilities = []
    ranks = []

    # Get the frequency of the most popular token and the and the # of all (non-unique) tokens
    highest_frequency = frequencies[0][1]
    sum_freq = sum(pair[1] for pair in frequencies)

    # Calculate the probability, ranks, and zipfian probability for each unique token
    for index, pair in enumerate(frequencies, start=1):
        actual_frequency = pair[1]
        actual_probability = actual_frequency / sum_freq * 100
        actual_probabilities.append(actual_probability)
        zipf_probability = (highest_frequency / index) / sum_freq * 100
        zipf_probabilities.append(zipf_probability)
        ranks.append(index)

    # Plot the data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(ranks, zipf_probabilities, s=10, c='r', marker='x', label='Ideal Zipfian Probabilities')
    ax.scatter(ranks, actual_probabilities, s=4, c='b', label='Dataset Terms Probabilities')
    plt.xlabel("Rank")
    plt.ylabel("Probability of Term Occurrence, %")
    plt.legend(loc='upper right')
    plt.show()

    # Calculate and return the Zipfs Law parameter for the dataset
    params = [a * b for a, b in zip(ranks, actual_probabilities)]
    return statistics.mean(params)


data = open('../../dataset/passage_collection_new.txt', 'r').read()
processed_data = preprocess_document(data, use_stopwords=True)
freq = get_frequencies(processed_data, 1000)
zipf_param = plot_zipf(freq)
print(zipf_param)
