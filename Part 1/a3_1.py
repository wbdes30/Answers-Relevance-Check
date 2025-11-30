import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import pandas as pd

# Task 1 (5 marks)
def topN_pos(csv_file_path, N):
    """
    Example:
    >>> topN_pos('train.csv', 3)
    output would look like [(noun1, 22), (noun2, 10), ...]
    """
    # Load the data
    data = pd.read_csv(csv_file_path)
    
    # Extract unique questions
    data.drop_duplicates('qtext', inplace=True)
    questions = data['qtext'].to_list()
    questions = " ".join(questions)
    
    # Find topN_pos
    sents = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(questions)]
    tagged_sents = nltk.pos_tag_sents(sents, tagset="universal")
    tagged_sents_N = []
    for s in tagged_sents:
        for w in s:
            if w[1] == 'NOUN':
                tagged_sents_N.append(w[0])

    counter = collections.Counter(tagged_sents_N)
    return counter.most_common(N)

# Task 2 (5 marks)
def topN_2grams(csv_file_path, N):
    """
    Example:
    >>> topN_2grams('train.csv', 3)
    output would look like [('what', 'is', 0.4113), ('how', 'many', 0.2139), ....], [('I', 'feel', 0.1264), ('pain', 'in', 0.2132), ...]
    """
    # Load the data
    data = pd.read_csv(csv_file_path)
    
    # Extract unique questions
    data.drop_duplicates('qtext', inplace=True)
    questions = data['qtext'].to_list()
    questions = " ".join(questions)
    
    # Find topN_2grams
    sents = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(questions)]
    bigrams = []
    for s in sents:
        for g in nltk.bigrams(s):
            bigrams.append(g)
    bigrams_counter = collections.Counter(bigrams)
    topN = bigrams_counter.most_common(N)
    non_stem_result = []
    for r in topN:
        r = list(r)
        r[1] = round(r[1]/len(bigrams),4)
        r = tuple(r)
        non_stem_result.append(r)
    
    # Find topN_2grams stems
    stemmer = nltk.PorterStemmer()
    questions_stems = stemmer.stem(questions)
    sents_stems = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(questions_stems)]
    bigrams_stems = []
    for s in sents_stems:
        for g in nltk.bigrams(s):
            bigrams_stems.append(g)

    bigrams_stems_counter = collections.Counter(bigrams_stems)
    topN = bigrams_stems_counter.most_common(N)
    stem_result = []
    for r in topN:
        r = list(r)
        r[1] = round(r[1]/len(bigrams),4)
        r = tuple(r)
        stem_result.append(r)
    
    # Combine 2 topN_2grams results to 1 final array
    final_result = []
    final_result.append(stem_result)
    final_result.append(non_stem_result)
    final_result = tuple(final_result)
    return final_result

# Task 3 (5 marks)
def sim_tfidf(csv_file_path):
    """
    Example:
    >>> sim_tfidf('train.csv')
    output format would be like 0.54
    """
    data = pd.read_csv(csv_file_path)
    
    # Extract unique questions and answers for fitting the vectorizer
    questions = data['qtext'].unique().tolist()
    unique_data = pd.concat([data['qtext'], data['atext']]).unique()
    
    # Initialize the TfidfVectorizer
    tfidf = TfidfVectorizer(stop_words='english', norm='l2')
    tfidf.fit(unique_data)
    
    correct_answers = 0
    for i in range(data['qtext'].nunique()):
        # tfidf vectors for unique questions
        q_vector = tfidf.transform([questions[i]]).toarray()[0] 
        # tfidf vectors for corresponding answers
        a_vector = tfidf.transform(data[data.qtext==questions[i]]['atext']).toarray() 
        # Calculate cosine similarity
        cosine_similarity = np.dot(a_vector, q_vector)
        # Get the indices of the highest similarity for each question
        highest_sim_index = np.argmax(cosine_similarity) 
        # Get the index of the answer with the highest cosine similarity
        best_data_index = data[data.qtext==questions[i]].index[0] + highest_sim_index 
        # Check the label of the answer with the highest cosine similarity
        if data.iloc[best_data_index]['label'] == 1:
            correct_answers += 1
    
    # Calculate the proportion of accurately answered questions
    proportion = correct_answers / data['qtext'].nunique()

    return round(proportion,2)

# DO NOT MODIFY THE CODE BELOW
if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    print(topN_pos('train.csv', 3))
    print("------------------------")
    print(topN_2grams('train.csv', 3))
    print("------------------------")
    print(sim_tfidf('train.csv'))

