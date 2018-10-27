# Mini-Search-Engine-II
Positional Inverted Index &amp; Free-text Queries with Proximity Operator

## Files
Part1.py
Part2.py
Part3a.txt
Part3b.txt

## PART 1

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


## PART 2

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


