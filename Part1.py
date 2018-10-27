#In order to run the program you will required to have NLTK packages for the python

#For mac users it can be installed as following:

#Install NLTK: run sudo pip install -U nltk
#Install Numpy (optional): run sudo pip install -U numpy
#Test installation: run python then type import nltk

#I have simply used the Sublimetext and "command + b" to run the code and for testing usage.

#You will also need the documents.txt file.

#Part1 from HW 1 is extended to create a positional inverted index for a given set of documents.
#The format of the index is modified a little. Index is still kept as a dictionary where the `term` is the key and the `index` is the value.

#I used tokenizer to find stop words and remove them.
#Then I used stemmer and lemmer to sort out similar words.
#Inverted Index for all 10 documents.

import re
import collections
import pickle

from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer

def sourceFile(source_file):
    beg_tag = re.compile(r'<DOC (\d)*>')
    end_tag = re.compile(r'</DOC>')
    
    data = []
    string = ""
    
    with open(source_file) as inputfile:
        tracer = inputfile.readlines()
        for current_tracer in tracer:
            if beg_tag.match(current_tracer):
                string = ""
            elif end_tag.match(current_tracer):
                data.append(string)
            else:
                string += current_tracer.strip()

    return data

#fileTokenizer will remove the stop words
def fileTokenizer(string):
    tokens = re.split(r'\s*[\',!.()="\-\s]*\s*', string)
    stopWords = ['']  # stopwords list changed, only space for stopwords
    washed_tokens = [word.lower() for word in tokens if word.lower() not in stopWords]
    #it will remove duplicate tokens
    #removed = list(set(washed_tokens))
    return washed_tokens  

#stemmer will help sort out simliar wods
def fileStemmer(string):
    porter_stemmer = PorterStemmer()
    return [porter_stemmer.stem(word) for word in string]


def fileLemmar(string):
    lemmar = WordNetLemmatizer()
    return [lemmar.lemmatize(word).encode('ascii') for word in string]

#this will check if there is an existing key or if there is not an existing key
def def_inverted_index(fileTokens, inverted_index, fieldId):
    doc_term_dic = {}
    term_pos = 0
    for word in fileTokens:
        term_pos += 1
        if word in doc_term_dic:  # if this is an existing key
            doc_term_dic[word].append(term_pos)
            doc_term_dic[word][1] += 1  # update tf
        else:
            doc_term_dic[word] = [fieldId, 1, term_pos]  # create docid,tf,first ocurrence pos

    for term in doc_term_dic:
        if term in inverted_index:
            inverted_index[term][0] += 1
            inverted_index[term].append(
                doc_term_dic[term])  # append the docid,tf, pos1, pos2 as a total list to inverted index
        else:
            positional_list = [1, doc_term_dic[term]]
            inverted_index[term] = positional_list
    return

#  modules are integrated here
#  data list will contain content of each document as a "String"
#  collections.OrderedDict will sort the tokens alphabetically
#  tokens_file will use for comparison in part 2 for queries
#  inverted_index_file will print the inverted_index.txt
def run_inverted_index(file_name, inverted_index_file, tokens_file):

    doc_list = sourceFile(file_name)

    doc_id = 1
    inverted_index = {}
    for doc in doc_list:
        token_list = fileStemmer(fileTokenizer(doc))
        def_inverted_index(token_list, inverted_index, doc_id)
        doc_id += 1

    inverted_index = collections.OrderedDict(sorted(inverted_index.items()))

    with open(tokens_file, 'wb') as output1:
        pickle.dump(inverted_index, output1)

    with open(inverted_index_file, 'w') as output2:
        for line in inverted_index:
            print >> output2, '{:18},{:3},{:3}'.format(line, inverted_index[line][0], inverted_index[line][1:])

    return doc_id-1

if __name__ == '__main__':
    print run_inverted_index('documents.txt', 'Part3a.txt', 'inverted_token_file')
