import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import numpy as np
import pandas as pd


class Corpus:
    def __init__(self, filePath, encoding, maxTokens):
        self.bagOfWords = Counter()
        self.listOfWords = ["removeMe"]  # So that the list is not empty
        self.docLen = list()
        self.numDocs = 0
        fileHandler = open(filePath, encoding=encoding)
        with fileHandler:
            for doc in fileHandler:
                self.numDocs += 1
                self.docLen.append(len(doc))
                tokens = doc.lower().split()
                self.bagOfWords.update(tokens)
                self.listOfWords = self.listOfWords + tokens
                if len(self.listOfWords) >= maxTokens + 1:
                    self.listOfWords.pop(0)  # Removed "removeMe
                    break
        fileHandler.close()


def estimateDocTopicProb(docId, topicParameter, docLen, docTopicFreq, numTopics):
    numerator = np.zeros(numTopics, dtype=float)
    for iTopicId in range(numTopics):
        numerator[iTopicId] = docTopicFreq[docId][iTopicId] + topicParameter
    denominator = docLen[docId] - 1 + numTopics * topicParameter
    return numerator / denominator


def estimateTopicWordProb(word, numTopics, dirParameter, wordTopicFreq):
    numerator = np.zeros(numTopics, dtype=float)
    for iTopicId in range(numTopics):
        numerator[iTopicId] = wordTopicFreq[word][iTopicId] + dirParameter
    sumWordsinToken = wordTopicFreq.sum(axis = 0)
    denominator = sumWordsinToken + maxTokens * dirParameter
    return numerator / denominator

filePath = "D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\books.txt"
fileEncoding = "ISO-8859-1"

maxTokens = 100000
books = Corpus(filePath, fileEncoding, maxTokens)
maxTokens = len(books.listOfWords)
numUniqueWords = len(books.bagOfWords)
numDocs = len(books.docLen)

numTopics = 10
dirParameter = 0.1
topicParameter = 0.1

wordTopic = np.random.randint(0, numTopics, maxTokens)
uniqueWords = books.bagOfWords.keys()
topicsMatrix = np.zeros((numUniqueWords, numTopics), dtype=int)
wordTopicFreq = dict(zip(uniqueWords, topicsMatrix))
wordTopicFreqDF = pd.DataFrame.from_dict(wordTopicFreq, orient="index")

docId = np.arange(0, numDocs, 1)
docsMatrix = np.zeros((numDocs, numTopics), dtype=int)
docTopicFreq = dict(zip(docId, docsMatrix))
docTopicFreqDF = pd.DataFrame.from_dict(docTopicFreq, orient="index")

# Random initialization matrix updates
jDocId = 0
for iWord in range(len(wordTopic)):
    wordTopicFreq[books.listOfWords[iWord]][wordTopic[iWord]] += 1
    if iWord > sum(books.docLen[:jDocId + 1]) - 1:
        jDocId += 1
    docTopicFreq[jDocId][wordTopic[iWord]] += 1

iDocId = 0
for iNumber, iWord in enumerate(books.listOfWords):
    print(iNumber)
    topicNumber = wordTopic[iNumber]
    wordTopicFreq[iWord][topicNumber] -= 1
    if iNumber > sum(books.docLen[:iDocId + 1]) - 1:
        iDocId += 1
    docTopicFreq[iDocId][topicNumber] -= 1
    docTopicProb = estimateDocTopicProb(iDocId, topicParameter, books.docLen, docTopicFreq, numTopics)
    wordTopicProb = estimateTopicWordProb(iWord, numTopics, dirParameter, wordTopicFreq)
    probWordInToken = np.multiply(docTopicProb, wordTopicProb)
    selectedTopic = np.argmax(probWordInToken)
    wordTopicFreq[iWord][selectedTopic] += 1
    docTopicFreq[iDocId][selectedTopic] += 1




