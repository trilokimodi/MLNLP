from collections import Counter
import numpy as np


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
                    self.listOfWords.pop(0)  # Removed "removeMe"
                    self.wordInDocIndex = np.delete(self.wordInDocIndex, 0)
                    break
        fileHandler.close()

    def most_common_word(self, numOfMostCommonWords, needToStore):
        if needToStore == 'Y':
            self.corpusMostCommonWords = Counter(self.bagOfWords).most_common(numOfMostCommonWords)
        elif needToStore == 'N':
            print(Counter(self.bagOfWords).most_common(numOfMostCommonWords))


class IntegerVocabulary:
    def __init__(self, mostCommonWords, maxVocabSize):
        self.integerVocab = dict()  # Changed from list to dict - See assignment 1
        wordCode = 0
        for iWord in range(maxVocabSize):
            self.integerVocab[mostCommonWords[iWord][0]] = wordCode
            wordCode += 1


def estimateDocTopicProb(docId):
    numerator = docTopicFreq[docId] + topicParameter
    denominator = books.docLen[docId] - 1 + numTopics * topicParameter
    return numerator / denominator


def estimateTopicWordProb(wordIndex, prevWordIndex):
    numerator = bigramTopicFreq[wordIndex, prevWordIndex] + dirParameter
    # sumWordsinToken = np.sum(bigramTopicFreq, axis=(0, 1))  # Takes plenty of time
    sumWordsinToken = np.sum(wordTopicFreq, axis=0)
    denominator = sumWordsinToken + numUniqueWords * dirParameter
    return numerator / denominator


filePath = "D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\books.txt"
fileEncoding = "ISO-8859-1"

maxGibbsIterations = 200
maxTokens = 50000
books = Corpus(filePath, fileEncoding, maxTokens)
maxTokens = len(books.listOfWords)
numUniqueWords = len(books.bagOfWords)
numDocs = len(books.docLen)

maxVocabSize = numUniqueWords
numOfMostCommonWords = maxVocabSize  # Not considering padding and out-of-vocabulary - See assignment 1
books.most_common_word(numOfMostCommonWords, 'Y')
booksIV = IntegerVocabulary(books.corpusMostCommonWords, maxVocabSize)

numTopics = 5
dirParameter = 0.1
topicParameter = 0.1

bigramTopicFreq = np.zeros((numUniqueWords, numUniqueWords, numTopics), dtype=int)  # This is transpose of what it
# suppose to be
wordTopicFreq = np.zeros((numUniqueWords, numTopics), dtype=int)
wordTopic = np.random.randint(0, numTopics, maxTokens)

docId = np.arange(0, numDocs, 1)
docTopicFreq = np.zeros((numDocs, numTopics), dtype=int)

# Random initialization matrix updates
jDocId = 0
for iNumber, iWord in enumerate(books.listOfWords[1:]):
    wordIdentity = booksIV.integerVocab[iWord]
    prevWord = books.listOfWords[iNumber - 1]
    prevWordIdentity = booksIV.integerVocab[prevWord]
    jDocId = books.wordInDocIndex[iNumber]
    jDocIdPrevWord = books.wordInDocIndex[iNumber - 1]
    if jDocId == jDocIdPrevWord:
        bigramTopicFreq[wordIdentity, prevWordIdentity, wordTopic[iNumber]] += 1
        wordTopicFreq[wordIdentity, wordTopic[iNumber]] += 1
        docTopicFreq[jDocId, wordTopic[iNumber]] += 1

iGibbs = 0

while iGibbs < maxGibbsIterations:
    print(iGibbs)
    iGibbs += 1
    iDocId = 0
    iDocIdPrevWord = 0
    for iNumber, iWord in enumerate(books.listOfWords[1:]):
        topicNumber = wordTopic[iNumber]
        wordIdentity = booksIV.integerVocab[iWord]
        prevWord = books.listOfWords[iNumber - 1]
        prevWordIdentity = booksIV.integerVocab[prevWord]
        iDocId = books.wordInDocIndex[iNumber]
        iDocIdPrevWord = books.wordInDocIndex[iNumber - 1]
        if iDocId == iDocIdPrevWord:
            bigramTopicFreq[wordIdentity, prevWordIdentity, topicNumber] -= 1
            wordTopicFreq[wordIdentity, topicNumber] -= 1
            docTopicFreq[iDocId, topicNumber] -= 1
            docTopicProb = estimateDocTopicProb(iDocId)
            wordTopicProb = estimateTopicWordProb(wordIdentity,
                                                  prevWordIdentity)  # Notice we have passed the integer index
            probWordInToken = np.multiply(docTopicProb, wordTopicProb)
            selectedTopic = np.random.multinomial(1, probWordInToken / probWordInToken.sum()).argmax()
            bigramTopicFreq[wordIdentity, prevWordIdentity, selectedTopic] += 1
            docTopicFreq[iDocId, selectedTopic] += 1
            wordTopicFreq[wordIdentity, selectedTopic] += 1
            wordTopic[iNumber] = selectedTopic

topicWordRelationByRawCount = list()
topicWordRelationByRelativeCount = list()
for iTopic in range(numTopics):
    topicWordRelationByRawCount.append(Counter())

for iNumber, iWord in enumerate(books.listOfWords):
    topicWordRelationByRawCount[wordTopic[iNumber]][iWord] += 1

for iTopic in range(numTopics):
    topicWordRelationByRelativeCount.append(topicWordRelationByRawCount[iTopic].copy())

for iTopic in range(numTopics):
    for iWord in topicWordRelationByRawCount[iTopic].keys():
        temp = topicWordRelationByRawCount[iTopic][iWord]
        topicWordRelationByRelativeCount[iTopic][iWord] = temp / books.bagOfWords[
            iWord]

for iTopic in range(numTopics):
    topicWordRelationByRawCount[iTopic] = sorted(topicWordRelationByRawCount[iTopic].items(), key=lambda x: x[1],
                                                 reverse=True)
    topicWordRelationByRelativeCount[iTopic] = sorted(topicWordRelationByRelativeCount[iTopic].items(),
                                                      key=lambda x: x[1], reverse=True)

numWordsToPrint = 30
for iTopic in range(numTopics):
    print("Topic number = %d, number of words in it = %d" % (iTopic, len(topicWordRelationByRelativeCount[iTopic])))
    for x in range(numWordsToPrint):
        print(topicWordRelationByRawCount[iTopic][x][0], end="\t")
    print("")
    for x in range(numWordsToPrint):
        print(topicWordRelationByRelativeCount[iTopic][x][0], end="\t")
    print("\n")
