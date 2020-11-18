import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict
import numpy as np
import pandas as pd


class Corpus:
    def __init__(self, filePath, encoding, maxTokens):
        self.bagOfWords = Counter()
        self.listOfWords = ["removeMe"]  # So that the list is not empty
        self.wordInDocIndex = np.zeros(1, dtype=int)
        self.docLen = list()
        self.numDocs = 0
        docNumber = 0
        fileHandler = open(filePath, encoding=encoding)
        with fileHandler:
            for doc in fileHandler:
                self.numDocs += 1
                self.docLen.append(len(doc))
                tokens = doc.lower().split()
                self.bagOfWords.update(tokens)
                self.listOfWords = self.listOfWords + tokens
                tempArray = docNumber * np.ones(len(tokens), dtype=int)
                self.wordInDocIndex = np.concatenate((self.wordInDocIndex, tempArray), axis=None)
                docNumber += 1
                if len(self.listOfWords) >= maxTokens + 1:
                    self.listOfWords.pop(0)  # Removed "removeMe
                    self.wordInDocIndex = np.delete(self.wordInDocIndex, 0)
                    break
        fileHandler.close()


def estimateDocTopicProb(docId):
    numerator = docTopicFreqDF.loc[docId] + topicParameter
    # denominator = books.docLen[docId] - 1 + numTopics * topicParameter
    return numerator


def estimateTopicWordProb(word):
    numerator = wordTopicFreqDF.loc[word] + dirParameter
    sumWordsinToken = wordTopicFreqDF.sum(axis=0)
    denominator = sumWordsinToken + vocabulary * dirParameter
    return numerator / denominator


filePath = "D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\books.txt"
fileEncoding = "ISO-8859-1"

maxGibbsIterations = 100
maxTokens = 100000
books = Corpus(filePath, fileEncoding, maxTokens)
maxTokens = len(books.listOfWords)
numUniqueWords = len(books.bagOfWords)
numDocs = len(books.docLen)
vocabulary = len(books.bagOfWords)

numTopics = 10
dirParameter = 0.1
topicParameter = 0.1

wordTopic = np.random.randint(0, numTopics, maxTokens)
uniqueWords = books.bagOfWords.keys()
topicsMatrix = np.zeros((numUniqueWords, numTopics), dtype=int)
wordTopicFreq = dict(zip(uniqueWords, topicsMatrix))
wordTopicFreqDF = pd.DataFrame.from_dict(wordTopicFreq, orient="index")
wordTopicFreq.clear()

docId = np.arange(0, numDocs, 1)
docsMatrix = np.zeros((numDocs, numTopics), dtype=int)
docTopicFreq = dict(zip(docId, docsMatrix))
docTopicFreqDF = pd.DataFrame.from_dict(docTopicFreq, orient="index")
docTopicFreq.clear()

# Random initialization matrix updates
jDocId = 0
for iWord in range(len(wordTopic)):
    wordTopicFreqDF.loc[books.listOfWords[iWord], wordTopic[iWord]] += 1
    jDocId = books.wordInDocIndex[iWord]
    docTopicFreqDF.loc[jDocId, wordTopic[iWord]] += 1

iGibbs = 0
while iGibbs < maxGibbsIterations:
    iGibbs += 1
    iDocId = 0
    for iNumber, iWord in enumerate(books.listOfWords):
        if iNumber % 1000 == 0:
            print(iNumber)
        topicNumber = wordTopic[iNumber]
        wordTopicFreqDF.loc[iWord, topicNumber] -= 1
        iDocId = books.wordInDocIndex[iNumber]
        docTopicFreqDF.loc[iDocId, topicNumber] -= 1
        docTopicProb = estimateDocTopicProb(iDocId)
        wordTopicProb = estimateTopicWordProb(iWord)
        probWordInToken = np.multiply(docTopicProb, wordTopicProb)
        selectedTopic = np.random.multinomial(1, probWordInToken/probWordInToken.sum()).argmax()
        wordTopicFreqDF.loc[iWord, selectedTopic] += 1
        docTopicFreqDF.loc[iDocId, selectedTopic] += 1
        wordTopic[iNumber] = selectedTopic

print("hello")
