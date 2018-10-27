#In order to run the program you will required to have NLTK packages for the python

#For mac users it can be installed as following:

#Install NLTK: run sudo pip install -U nltk
#Install Numpy (optional): run sudo pip install -U numpy
#Test installation: run python then type import nltk

#I have simply used the Sublimetext and "command + b" to run the code and for testing usage.

#You will also need the documents.txt file.

#Before computing the quere, the querey is parsed into proximity operator.
#the function parse_query(input_query) is built for that reason.

#the evaluarot operator will build a new index for bigram in the proximity operator.
#Construct the tf_idf_matrix will compute and check its positional index and traverse the index
#to get the tf and calculate the scoring for that along with df.

import Part1
import re
import pickle
import math
import numpy as np

from nltk.stem.porter import *

#there are orginially 10 documents
#this program will evaluate free-text queries with proximity operators using a tf.idf scoring function
#which will generate a ranked result lsit of documents for each query.

def parse_query(input_query):
    pattern = re.compile(r'\s*[\s*\(\)]\s*')
    raw_terms = pattern.split(input_query)
    p_stemmer = PorterStemmer()
    query_terms = [p_stemmer.stem(word.lower()).encode('ascii') for word in raw_terms]

    final_query_terms = []
    x = 0
    while x < len(query_terms):
        if query_terms[x].isdigit():
            final_query_terms.append([True, query_terms[x], query_terms[x + 1], query_terms[x + 2]])
            x += 3
        elif query_terms[x] == '':
            x += 1
        else:
            final_query_terms.append([False, query_terms[x]])
            x += 1

    return final_query_terms

# proximity_evaluation will find both terms in one doc then perform the distance check for the terms
def proximity_evaluation(distance, term1, term2, positional_index):
    term1_index_list = positional_index[term1]
    term2_index_list = positional_index[term2]
    term1_df = term1_index_list[0]
    term2_df = term2_index_list[0]
    term1_index = 1
    term2_index = 1
    bigram_positional_index = []

    while term1_index <= term1_df and term2_index <= term2_df:
        if term1_index_list[term1_index][0] == term2_index_list[term2_index][0]:
            doc_id = term1_index_list[term1_index][0]
            term1_pos_list = term1_index_list[term1_index][1:]
            term2_pos_list = term2_index_list[term2_index][1:]
            term1_index += 1
            term2_index += 1
            term1_pos = 1
            term2_pos = 1
            term1_tf = term1_pos_list[0]
            term2_tf = term2_pos_list[0]
            bigram_tf = 0
            while term1_pos <= term1_tf and term2_pos <= term2_tf:
                if term2_pos_list[term2_pos] - term1_pos_list[term1_pos] > 0 and (
                                term2_pos_list[term2_pos] - term1_pos_list[term1_pos] <= (distance + 1)):
                    bigram_tf += 1
                    term1_pos += 1
                    term2_pos += 1
                elif term2_pos_list[term2_pos] - term1_pos_list[term1_pos] < 0:
                    term2_pos += 1
                elif term2_pos_list[term2_pos] - term1_pos_list[term1_pos] > distance + 1:
                    term1_pos += 1

            if bigram_tf > 0:
                bigram_positional_index.append([doc_id, bigram_tf])

        elif term1_index_list[term1_index][0] < term2_index_list[term2_index][0]:
            term1_index += 1
        else:
            term2_index += 1

    return [len(bigram_positional_index)] + bigram_positional_index

#we were asked to use log base 2 instead of provided log base 10 in the PDF
def cal_weight_td(df, tf, doc_num):
    wtd = (1 + np.log2(tf)) * (np.log2(doc_num) - np.log2(df))
    return wtd

# tf_idf_matrix will initial the matrix with 0
# this will also print bigram_list

def tf_idf_matrix(positional_index, query_term_list, doc_num):
    t_i_matrix = [[0 for y in range(doc_num)] for x in range(len(query_term_list))]
    for query_object in query_term_list:
        if query_object[0] == False:
            query_term = query_object[1]
            if query_term in positional_index:
                p_index = positional_index[query_term]
                term_df = p_index[0]
                for doc_position_index in p_index[1:]:
                    docid = doc_position_index[0]
                    term_tf = doc_position_index[1]
                    t_i_matrix[query_term_list.index(query_object)][docid - 1] = cal_weight_td(term_df, term_tf,
                                                                                               doc_num)
        else:
            distance = int(query_object[1])
            term1 = query_object[2]
            term2 = query_object[3]
            t_i_matrix[query_term_list.index(query_object)] = [-100] * doc_num
            if term1 in positional_index and term2 in positional_index:
                bigram_list = proximity_evaluation(distance, term1, term2, positional_index)
                bigram_df = bigram_list[0]
                if bigram_df != 0:
                    for bigram_doc_tf_list in bigram_list[1:]:
                        docid = bigram_doc_tf_list[0]
                        bigram_tf = bigram_doc_tf_list[1]
                        t_i_matrix[query_term_list.index(query_object)][docid - 1] = cal_weight_td(bigram_df, bigram_tf, doc_num)
    return t_i_matrix


# compute_tfidf_bydoc will print query term list,
# the rank of the query in sorted fashion
def compute_tfidf_bydoc(origin_query, source_index_file, doc_num):
    with open(source_index_file, 'rb') as input_file:
        positional_index = pickle.load(input_file)

    query_term_list = parse_query(origin_query)
    rank_matrix = tf_idf_matrix(positional_index, query_term_list, doc_num)
    raw_rank_list = map(sum, zip(*rank_matrix))
    rank_list = map(list, enumerate(raw_rank_list, 1))
    for score in rank_list:
        score[1] = score[1] if score[1] > 0 else 0

    sorted_rank = sorted(rank_list, key=lambda item: -item[1])

    return sorted_rank

#Mini Search Engine Code (This will generated the result for Part3b)

def mini_search_engine():
    origin_query1 = "nexus like love happy"
    origin_query2 = "asus repair"
    origin_query3 = "0(touch screen)"
    origin_query4 = "1(great tablet) 2(tablet fast)"
    origin_query6 = "tablet"
    
    source_txt_file = 'winedescription.txt'
    txtfile_name = 'outputtxt.txt'
    serialized_index_file = "inverted_token_file"
    
    doc_num = Part1.run_inverted_index(source_txt_file, txtfile_name, serialized_index_file)
    
    rank_result1 = filter_irrelevant_doc(compute_tfidf_bydoc(origin_query1, serialized_index_file, doc_num))
    rank_result2 = filter_irrelevant_doc(compute_tfidf_bydoc(origin_query2, serialized_index_file, doc_num))
    rank_result3 = filter_irrelevant_doc(compute_tfidf_bydoc(origin_query3, serialized_index_file, doc_num))
    rank_result4 = filter_irrelevant_doc(compute_tfidf_bydoc(origin_query4, serialized_index_file, doc_num))
    rank_result6 = filter_irrelevant_doc(compute_tfidf_bydoc(origin_query6, serialized_index_file, doc_num))
    
    print rank_result1
    print rank_result2
    print rank_result3
    print rank_result4
    print rank_result6

def filter_irrelevant_doc(rank_list):
    return [item for item in rank_list if item[1] > 0]

if __name__ == '__main__':
    mini_search_engine()

#READ IMPORTANT:

#Part3b is attached up here beacause only 3 files were allowed to upload on the iLearn.

#Part3b.txt
#the following results will be generated
#[[2, 9.4760282429871587], [6, 5.7958592832197748], [8, 4.0588936890535683], [1, 3.4739311883324122], [4, 2.3219280948873622]]
#[[8, 8.092002902000802], [3, 4.0588936890535683], [1, 3.4739311883324122]]
#[[3, 6.0020970546547456], [8, 2.3219280948873622]]
#[]
#[[4, 1.4739311883324122], [2, 0.73696559416620611], [3, 0.73696559416620611], [6, 0.73696559416620611], [9, 0.73696559416620611], [10, 0.73696559416620611]]
#[Finished in 0.7s]
