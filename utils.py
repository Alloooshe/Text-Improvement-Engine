import csv
import spacy 
import numpy as np

def read_csv(filename):
    ret = []
    with open(filename) as file:
        for row in csv.reader(file, delimiter='\n'):
          ret.append(row[0].lower())
    return ret

def read_text(filename):
    with open(filename,'r') as file : 
        txt = file.read()
    return txt

def k_largest_index(a, k):
    idx = np.argpartition(-a.ravel(),k)[:k]
    return np.column_stack(np.unravel_index(idx, a.shape))

def convert_text_to_list_sentences(txt):
    """convert a text to list of sentence using spacy en_core_web_lg model.

    Args:
        txt (string): input text

    Returns:
        list[string]: list of all sentences in the input text
    """
    nlp = spacy.load("en_core_web_lg")
    text = nlp(txt)
    all_sentences = [] 
    for sent in text.sents:
        lst = sent.text.split(',')
        all_sentences+= [x.strip() for x in lst]
    return all_sentences


def find_ngrams(sentence):
    """find all possible word ngrams in a sentence, with ngrams no less than two words.

    Args:
        sentence (string): input sentence

    Returns:
        list[string]: list of all ngrams in the sentence
    """
    max_N = len(sentence.split())
    if max_N <2: 
        return [sentence]
    words = sentence.split()
    ngrams =[]
    for N in range(2,max_N+1):
        ngrams += [ ' '.join(words [i: i + N]) for i in range(len(words) - N + 1)]
    return ngrams