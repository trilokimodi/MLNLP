from collections import Counter
import numpy as np
from spacy.lang.en import English

nlp = English()


class Corpus:
    def __init__(self, filePath, encoding, maxTokens):
        self.bagOfWords = Counter()
        self.listOfWords = ["removeMe"]  # So that the list is not empty
        self.wordInDocIndex = np.zeros(1, dtype=int)
        self.docLen = list()
        self.numDocs = 0
        docNumber = 0
        numActualToken = 0
        fileHandler = open(filePath, encoding=encoding)
        with fileHandler:
            for doc in fileHandler:
                self.numDocs += 1
                tokens = doc.lower().split()
                numActualToken += len(tokens)
                numTokenTemp = 0
                for iToken in tokens:
                    if not nlp.vocab[iToken].is_stop and iToken not in ['.', '&', '...', '-', ';', '/', '!', ',', '(',
                                                                        ')', "'", '"', ':', "?",
                                                                        '....'] and not iToken.isdigit():
                        self.bagOfWords[iToken] += 1
                        self.listOfWords = self.listOfWords + [iToken]
                        numTokenTemp += 1
                self.docLen.append(numTokenTemp)
                tempArray = docNumber * np.ones(numTokenTemp, dtype=int)
                self.wordInDocIndex = np.concatenate((self.wordInDocIndex, tempArray), axis=None)
                docNumber += 1
                if numActualToken >= maxTokens + 1:
                    self.listOfWords.pop(0)  # Removed "removeMe"
                    self.wordInDocIndex = np.delete(self.wordInDocIndex, 0)
                    break
        fileHandler.close()

    def most_common_word(self, numOfMostCommonWords):
        self.corpusMostCommonWords = Counter(self.bagOfWords).most_common(numOfMostCommonWords)


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


def estimateTopicWordProbFirstWords(wordIndex):  # The first words don't have preceeding word so this function is
    # picked from task 1 of this assignment.
    numerator = wordTopicFreq[wordIndex] + dirParameter
    sumWordsinToken = wordTopicFreq.sum(axis=0)
    denominator = sumWordsinToken + numUniqueWords * dirParameter
    return numerator / denominator


filePath = "D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\books.txt"
fileEncoding = "ISO-8859-1"
filePath2 = "D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\A2_Task2.txt"

maxGibbsIterations = 5
maxTokens = 1000
desiredWordsToBePrinted = 50

books = Corpus(filePath, fileEncoding, maxTokens)
maxTokens = len(books.listOfWords)
numUniqueWords = len(books.bagOfWords)
numDocs = len(books.docLen)

maxVocabSize = numUniqueWords
numOfMostCommonWords = maxVocabSize  # Not considering padding and out-of-vocabulary - See assignment 1
books.most_common_word(numOfMostCommonWords)
booksIV = IntegerVocabulary(books.corpusMostCommonWords, maxVocabSize)

numTopicsList = [5, 10]
parameterList = [(0.1, 0.1), (0.01, 0.01)]

for iTopicList in range(len(numTopicsList)):
    for iParameterList in range(len(parameterList)):
        numTopics = numTopicsList[iTopicList]
        dirParameter = parameterList[iParameterList][0]
        topicParameter = parameterList[iParameterList][1]

        wordTopic = np.random.randint(0, numTopics, maxTokens)
        bigramTopicFreq = np.zeros((numUniqueWords, numUniqueWords, numTopics),
                                   dtype=int)  # This is transpose of what it suppose to be
        wordTopicFreq = np.zeros((numUniqueWords, numTopics), dtype=int)

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
                else:
                    wordTopicFreq[wordIdentity, topicNumber] -= 1
                    docTopicFreq[iDocId, topicNumber] -= 1
                    docTopicProb = estimateDocTopicProb(iDocId)
                    wordTopicProb = estimateTopicWordProbFirstWords(
                        wordIdentity)  # First words are first word in each doc.
                    probWordInToken = np.multiply(docTopicProb, wordTopicProb)
                    selectedTopic = np.random.multinomial(1, probWordInToken / probWordInToken.sum()).argmax()
                    wordTopicFreq[booksIV.integerVocab[iWord], selectedTopic] += 1
                    docTopicFreq[iDocId, selectedTopic] += 1
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
            topicWordRelationByRawCount[iTopic] = sorted(topicWordRelationByRawCount[iTopic].items(),
                                                         key=lambda x: x[1],
                                                         reverse=True)
            topicWordRelationByRelativeCount[iTopic] = sorted(topicWordRelationByRelativeCount[iTopic].items(),
                                                              key=lambda x: x[1], reverse=True)

        maxWordsCanBePrinted = list()
        for iMax in range(numTopics):
            maxWordsCanBePrinted.append(len(topicWordRelationByRawCount[iMax]))

        numWordsToPrint = list()
        for iMin in range(numTopics):
            numWordsToPrint.append(min(maxWordsCanBePrinted[iMin], desiredWordsToBePrinted))
        # for iTopic in range(numTopics):
        #     print("Topic number = %d, number of words in it = %d" % (
        #         iTopic, len(topicWordRelationByRelativeCount[iTopic])))
        #     for x in range(numWordsToPrint[iTopic]):
        #         print(topicWordRelationByRawCount[iTopic][x][0], end="\t")
        #     print("")
        #     for x in range(numWordsToPrint[iTopic]):
        #         print(topicWordRelationByRelativeCount[iTopic][x][0], end="\t")
        #     print("\n")

        fileHandler2 = open(filePath2, 'a')
        with fileHandler2:
            fileHandler2.write("K = %s, alpha = beta = %s\n" % (str(numTopics), str(dirParameter)))
            for iTopic in range(numTopics):
                fileHandler2.write("\n\nTopic number = %s, number of words in it = %s\n" % (
                    str(iTopic), str(len(topicWordRelationByRelativeCount[iTopic]))))
                fileHandler2.write("\nBy raw count\n")
                for x in range(numWordsToPrint[iTopic]):
                    fileHandler2.write(topicWordRelationByRawCount[iTopic][x][0])
                    fileHandler2.write("\t")
                fileHandler2.write("\nBy relative count\n")
                for x in range(numWordsToPrint[iTopic]):
                    fileHandler2.write(topicWordRelationByRelativeCount[iTopic][x][0])
                    fileHandler2.write("\t")
            fileHandler2.write("\n\n\n")
        fileHandler2.close()
