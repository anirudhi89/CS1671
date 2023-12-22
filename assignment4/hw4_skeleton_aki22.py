import os
import subprocess
import csv
import re
import random
import numpy as np


def read_in_shakespeare():
    """Reads in the Shakespeare dataset processesit into a list of tuples.
       Also reads in the vocab and play name lists from files.

    Each tuple consists of
    tuple[0]: The name of the play
    tuple[1] A line from the play as a list of tokenized words.

    Returns:
      tuples: A list of tuples in the above format.
      document_names: A list of the plays present in the corpus.
      vocab: A list of all tokens in the vocabulary.
    """

    tuples = []

    with open("will_play_text.csv") as f:
        csv_reader = csv.reader(f, delimiter=";")
        for row in csv_reader:
            play_name = row[1]
            line = row[5]
            line_tokens = re.sub(r"[^a-zA-Z0-9\s]", " ", line).split()
            line_tokens = [token.lower() for token in line_tokens]

            tuples.append((play_name, line_tokens))

    with open("vocab.txt") as f:
        vocab = [line.strip() for line in f]

    with open("play_names.txt") as f:
        document_names = [line.strip() for line in f]

    return tuples, document_names, vocab


def get_row_vector(matrix, row_id):
    """A convenience function to get a particular row vector from a numpy matrix

    Inputs:
      matrix: a 2-dimensional numpy array
      row_id: an integer row_index for the desired row vector

    Returns:
      1-dimensional numpy array of the row vector
    """
    return matrix[row_id, :]


def get_column_vector(matrix, col_id):
    """A convenience function to get a particular column vector from a numpy matrix

    Inputs:
      matrix: a 2-dimensional numpy array
      col_id: an integer col_index for the desired row vector

    Returns:
      1-dimensional numpy array of the column vector
    """
    return matrix[:, col_id]


def create_term_document_matrix(line_tuples, document_names, vocab):
    """Returns a numpy array containing the term document matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      document_names: A list of the document names
      vocab: A list of the tokens in the vocabulary

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let m = len(vocab) and n = len(document_names).

    Returns:
      td_matrix: A mxn numpy array where the number of rows is the number of words
          and each column corresponds to a document. A_ij contains the
          frequency with which word i occurs in document j.
    """
    # YOUR CODE HERE
    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    docname_to_id = dict(zip(document_names, range(0, len(document_names))))

    td_matrix = np.zeros((len(vocab), len(document_names)))

    for doc_name, words in line_tuples:
        for word in words:
            td_matrix[vocab_to_id[word], docname_to_id[doc_name]] += 1

    return td_matrix



def create_term_context_matrix(line_tuples, vocab, context_window_size=1):
    """Returns a numpy array containing the term context matrix for the input lines.

    Inputs:
      line_tuples: A list of tuples, containing the name of the document and
      a tokenized line from that document.
      vocab: A list of the tokens in the vocabulary

    # NOTE: THIS DOCSTRING WAS UPDATED ON JAN 24, 12:39 PM.

    Let n = len(vocab).

    Returns:
      tc_matrix: A nxn numpy array where A_ij contains the frequency with which
          word j was found within context_window_size to the left or right of
          word i in any sentence in the tuples.
    """

    # YOUR CODE HERE
    vocab_to_id = dict(zip(vocab, range(0, len(vocab))))
    
    m = len(vocab)
    tc_matrix = np.zeros([m, m])

    for line_tuple in line_tuples:
        line = line_tuple[1]
        for idx, word in enumerate(line):
            word_id = vocab_to_id[word]
            for context_idx in range(1, context_window_size + 1):
                prev_idx = idx - context_idx
                if prev_idx >= 0:
                    prev_word = line[prev_idx]
                    prev_word_id = vocab_to_id[prev_word]
                    tc_matrix[word_id, prev_word_id] += 1
                next_idx = idx + context_idx
                if next_idx < len(line):
                    next_word = line[next_idx]
                    next_word_id = vocab_to_id[next_word]
                    tc_matrix[word_id, next_word_id] += 1
            tc_matrix[word_id, word_id] = 0 #self-co-oocurrence is not counted (See writeup)
    return tc_matrix



def create_tf_idf_matrix(term_document_matrix):
    """Given the term document matrix, output a tf-idf weighted version.

    See section 6.5 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_document_matrix: Numpy array where each column represents a document
      and each row, the frequency of a word in that document.

    Returns:
      A numpy array with the same dimension as term_document_matrix, where
      A_ij is weighted by the inverse document frequency of document h.
    """
    # YOUR CODE HERE
    num_documents = np.size(term_document_matrix, axis=1)
    tf_matrix = np.log10(term_document_matrix + 1) + 1
    document_frequency = np.sum(np.heaviside(term_document_matrix, 0), axis=1)
    idf_matrix = np.log10(num_documents / (document_frequency + 1)) 
    tf_idf_matrix = (tf_matrix.T * idf_matrix).T

    return tf_idf_matrix
	
def create_ppmi_matrix(term_context_matrix):
    """Given the term context matrix, output a ppmi weighted version.

    See section 6.6 in the textbook.

    Hint: Use numpy matrix and vector operations to speed up implementation.

    Input:
      term_context_matrix: Numpy array where each cell represents whether the 
	  word in the row appears within a window of the word in the column.

    Returns:
      A numpy array with the same dimension as term_context_matrix, where
      A_ij is weighted using PPMI.
    """

    # YOUR CODE HERE
    row_sums = np.sum(term_context_matrix, axis=1)
    col_sums = np.sum(term_context_matrix, axis=0)
    p_i_star = row_sums / (np.sum(term_context_matrix) + 1e-6) # 1e6 StackOverflow: avoid division by zero
    p_j_star = col_sums / (np.sum(term_context_matrix) + 1e-6)

    ppmi_matrix = np.maximum(
        np.log2((term_context_matrix + 1e-6) / (np.outer(p_i_star, p_j_star) + 1e-6)),
        0
    )

    return ppmi_matrix


def compute_cosine_similarity(vector1, vector2):
    """Computes the cosine similarity of the two input vectors.

    Inputs:
      vector1: A nx1 numpy array
      vector2: A nx1 numpy array

    Hint: Use numpy matrix and vector operations to speed up implementation.


    Returns:
      A scalar similarity value.
    """
    # YOUR CODE HERE
    numerator = np.dot(vector1, vector2)
    denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    if denominator != 0:
        return numerator / denominator
    else:
        return 0


def rank_words(target_word_index, matrix):
    """Ranks the similarity of all of the words to the target word using compute_cosine_similarity.

    Inputs:
      target_word_index: The index of the word we want to compare all others against.
      matrix: Numpy matrix where the ith row represents a vector embedding of the ith word.

    Returns:
      A length-n list of integer word indices, ordered by decreasing similarity to the
      target word indexed by word_index
      A length-n list of similarity scores, ordered by decreasing similarity to the
      target word indexed by word_index
    """
    # YOUR CODE HERE
    target_vector = matrix[target_word_index]

    similarities = [compute_cosine_similarity(target_vector, matrix[i]) for i in range(len(matrix))]

    ranked_indices = np.argsort(similarities)[::-1]  
    ranked_scores = [similarities[i] for i in ranked_indices]

    return ranked_indices, ranked_scores
	

if __name__ == "__main__":
    tuples, document_names, vocab = read_in_shakespeare()

    print("Computing term document matrix...")
    td_matrix = create_term_document_matrix(tuples, document_names, vocab)

    print("Computing tf-idf matrix...")
    tf_idf_matrix = create_tf_idf_matrix(td_matrix)


    print("Computing term context matrix...")
    tc_matrix = create_term_context_matrix(tuples, vocab, context_window_size=2)

    print("Computing PPMI matrix...")
    ppmi_matrix = create_ppmi_matrix(tc_matrix)
    # print("done")

    # random_idx = random.randint(0, len(document_names) - 1)

    word = "juliet"
    vocab_to_index = dict(zip(vocab, range(0, len(vocab))))

    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on term-document frequency matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], td_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))

    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on term-context frequency matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], tc_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))


    print(
        '\nThe 10 most similar words to "%s" using cosine-similarity on tf-idf matrix are:'
        % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], tf_idf_matrix)
    for idx in range(0,10):
        word_id = ranks[idx]
        print("%d: %s; %s" %(idx+1, vocab[word_id], scores[idx]))
    
    print(
    '\nThe 10 most similar words to "%s" using cosine-similarity on PPMI matrix are:'
    % (word)
    )
    ranks, scores = rank_words(vocab_to_index[word], ppmi_matrix)
    for idx in range(0, 10):
        word_idx = ranks[idx]
        similar_word = vocab[word_idx]
        print("%d: %s; %s" % (idx + 1, similar_word, scores[idx]))
