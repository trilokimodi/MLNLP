from collections import Counter
import numpy as np
from spacy.lang.en import English
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

nlp = English()


class Corpus:
    def __init__(self, filePath, encoding, maxTokens):
        self.bagOfWords = Counter()
        self.listOfWords = ["removeMe"]  # So that the list is not empty
        self.wordInDocIndex = np.zeros(1, dtype=int)  # So that array is not empty
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


def estimateTopicWordProbUnPairedWords(wordIndex):  # The first words don't have preceeding word so this function is
    # picked from task 1 of this assignment.
    numerator = wordTopicFreq[wordIndex] + dirParameter
    sumWordsinToken = wordTopicFreq.sum(axis=0)
    denominator = sumWordsinToken + numUniqueWords * dirParameter  # Need to check if should be commented or included.
    return numerator / denominator


filePath = "D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment1\\a1_data\\books.txt"
fileEncoding = "ISO-8859-1"
filePathTask2Output = "D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment2\\A2_Task2.txt"

maxGibbsIterations = 200
maxTokens = 100000
desiredWordsToBePrinted = 20

books = Corpus(filePath, fileEncoding, maxTokens)
maxTokens = len(books.listOfWords)
numUniqueWords = len(books.bagOfWords)
numDocs = len(books.docLen)

maxVocabSize = numUniqueWords
numOfMostCommonWords = maxVocabSize  # Not considering padding and out-of-vocabulary - See assignment 1
books.most_common_word(numOfMostCommonWords)
booksIV = IntegerVocabulary(books.corpusMostCommonWords, maxVocabSize)

documentTopicsDF = pd.DataFrame()
figureNum = 0
numTopicsList = [5, 10]
parameterList = [(0.1, 0.1), (0.01, 0.01)]

wordTopicResultsT2 = list()
sumWordsinTokenResultsT2 = list()

iCase = 0
for iTopicList in range(len(numTopicsList)):
    for iParameterList in range(len(parameterList)):

        numTopics = numTopicsList[iTopicList]
        dirParameter = parameterList[iParameterList][0]
        topicParameter = parameterList[iParameterList][1]

        wordTopicResultsT2.append(list())
        sumWordsinTokenResultsT2.append(np.zeros(numTopics))

        # Random initialization matrix updates - Can be changed to final matrix from task 1 to avoid burn in
        wordTopic = np.random.randint(0, numTopics, maxTokens)
        bigramTopicFreq = np.zeros((numUniqueWords, numUniqueWords, numTopics),
                                   dtype=int)  # This is transpose of what it suppose to be
        wordTopicFreq = np.zeros((numUniqueWords, numTopics), dtype=int)

        docId = np.arange(0, numDocs, 1)
        docTopicFreq = np.zeros((numDocs, numTopics), dtype=int)

        jDocId = 0
        for iNumber, iWord in enumerate(books.listOfWords):
            if iNumber > 0:
                wordIdentity = booksIV.integerVocab[iWord]
                prevWord = books.listOfWords[iNumber - 1]
                prevWordIdentity = booksIV.integerVocab[prevWord]
                jDocId = books.wordInDocIndex[iNumber]
                jDocIdPrevWord = books.wordInDocIndex[iNumber - 1]
                if jDocId == jDocIdPrevWord:
                    bigramTopicFreq[wordIdentity, prevWordIdentity, wordTopic[iNumber]] += 1
                    wordTopicFreq[wordIdentity, wordTopic[iNumber]] += 1
                    docTopicFreq[jDocId, wordTopic[iNumber]] += 1
                else:
                    wordTopicFreq[wordIdentity, wordTopic[iNumber]] += 1
                    docTopicFreq[jDocId, wordTopic[iNumber]] += 1
            else:
                wordIdentity = booksIV.integerVocab[iWord]
                wordTopicFreq[wordIdentity, wordTopic[iNumber]] += 1
                docTopicFreq[jDocId, wordTopic[iNumber]] += 1

        iGibbs = 0
        while iGibbs < maxGibbsIterations:
            print(iGibbs)
            iGibbs += 1
            iDocId = 0
            iDocIdPrevWord = 0
            for iNumber, iWord in enumerate(books.listOfWords):
                if iNumber > 0:
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
                                                              prevWordIdentity)  # Notice, both are integer index
                        probWordInToken = np.multiply(docTopicProb, wordTopicProb)
                        selectedTopic = np.random.multinomial(1, probWordInToken / probWordInToken.sum()).argmax()
                        bigramTopicFreq[wordIdentity, prevWordIdentity, selectedTopic] += 1
                        docTopicFreq[iDocId, selectedTopic] += 1
                        wordTopicFreq[wordIdentity, selectedTopic] += 1
                        wordTopic[iNumber] = selectedTopic
                    else:  # Not necessary as only a few words will be affected.(The first word in every doc)
                        wordTopicFreq[wordIdentity, topicNumber] -= 1
                        docTopicFreq[iDocId, topicNumber] -= 1
                        docTopicProb = estimateDocTopicProb(iDocId)
                        wordTopicProb = estimateTopicWordProbUnPairedWords(
                            wordIdentity)  # First words are first word in each doc.
                        probWordInToken = np.multiply(docTopicProb, wordTopicProb)
                        selectedTopic = np.random.multinomial(1, probWordInToken / probWordInToken.sum()).argmax()
                        wordTopicFreq[booksIV.integerVocab[iWord], selectedTopic] += 1
                        docTopicFreq[iDocId, selectedTopic] += 1
                        wordTopic[iNumber] = selectedTopic
                else:  # This is only for first word in entire corpus. Not doing this shouldn't affect result at all. So
                    # this part can be skipped too
                    wordIdentity = booksIV.integerVocab[iWord]
                    topicNumber = wordTopic[iNumber]
                    wordTopicFreq[wordIdentity, topicNumber] -= 1
                    docTopicFreq[iDocId, topicNumber] -= 1
                    docTopicProb = estimateDocTopicProb(iDocId)
                    wordTopicProb = estimateTopicWordProbUnPairedWords(
                        wordIdentity)  # First words are first word in each doc.
                    probWordInToken = np.multiply(docTopicProb, wordTopicProb)
                    selectedTopic = np.random.multinomial(1, probWordInToken / probWordInToken.sum()).argmax()
                    wordTopicFreq[booksIV.integerVocab[iWord], selectedTopic] += 1
                    docTopicFreq[iDocId, selectedTopic] += 1
                    wordTopic[iNumber] = selectedTopic

        wordTopicResultsT2[iCase] = wordTopic
        sumWordsinTokenResultsT2[iCase] = wordTopicFreq.sum(axis=0)
        iCase += 1

# Results
topTopicsSize = 5
iCase = 0
for iTopicList in range(len(numTopicsList)):
    for iParameterList in range(len(parameterList)):
        numTopics = numTopicsList[iTopicList]
        dirParameter = parameterList[iParameterList][0]
        topicParameter = parameterList[iParameterList][1]

        print("Case %d, \u03B1 = %.2f, \u03B2 = %.2f, K = %d\n\n" % (iCase + 1, topicParameter, dirParameter, numTopics))

        # Result part - 1. Plots
        sumWordsinToken = sumWordsinTokenResultsT2[iCase].copy()
        figureNum += 1
        ax = plt.figure(figureNum).gca()
        plt.scatter(np.arange(0, numTopics), sumWordsinToken / maxTokens, label="All words")
        plt.xlabel("Topic Number")
        plt.ylabel("fraction of words")
        plt.title(r"Fraction of words, $\alpha = {}, \beta$ = {}, K = {}".format(str(topicParameter), str(dirParameter),
                                                                                 str(numTopics)))
        plt.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        file = "D:\\Masters Program Chalmers\\Projects and Labs\\MLNLP\\Assignment2\\A2_T2_case" + str(iCase) + ".png"
        plt.savefig(file)

        # Result part - 2
        topicWordRelationByRawCount = list()
        topicWordRelationByRelativeCount = list()
        topicWordRelationByRelInMaxRaw = list()
        for iTopic in range(numTopics):
            topicWordRelationByRawCount.append(Counter())
            topicWordRelationByRelInMaxRaw.append(Counter())

        for iNumber, iWord in enumerate(books.listOfWords):
            topicWordRelationByRawCount[wordTopicResultsT2[iCase][iNumber]][iWord] += 1

        for iTopic in range(numTopics):
            topicWordRelationByRelativeCount.append(topicWordRelationByRawCount[iTopic].copy())

        for iTopic in range(numTopics):
            for iWord in topicWordRelationByRawCount[iTopic].keys():
                temp = topicWordRelationByRawCount[iTopic][iWord]
                topicWordRelationByRelativeCount[iTopic][iWord] = temp / books.bagOfWords[
                    iWord]

        for iTopic in range(numTopics):
            tempDict = [topicWordRelationByRawCount[iTopic], topicWordRelationByRelativeCount[iTopic]]
            for iWord in topicWordRelationByRawCount[iTopic].keys():
                topicWordRelationByRelInMaxRaw[iTopic][iWord] = tuple(i[iWord] for i in tempDict)

        for iTopic in range(numTopics):
            topicWordRelationByRawCount[iTopic] = sorted(topicWordRelationByRawCount[iTopic].items(),
                                                         key=lambda x: x[1],
                                                         reverse=True)
            topicWordRelationByRelativeCount[iTopic] = sorted(topicWordRelationByRelativeCount[iTopic].items(),
                                                              key=lambda x: x[1], reverse=True)
            topicWordRelationByRelInMaxRaw[iTopic] = sorted(topicWordRelationByRelInMaxRaw[iTopic].items(),
                                                            key=lambda x: x[1][0], reverse=True)

        maxWordsCanBePrinted = list()
        for iMax in range(numTopics):
            maxWordsCanBePrinted.append(len(topicWordRelationByRawCount[iMax]))

        numWordsToPrint = list()
        for iMin in range(numTopics):
            numWordsToPrint.append(min(maxWordsCanBePrinted[iMin], desiredWordsToBePrinted))
            topicWordRelationByRelInMaxRaw[iMin] = topicWordRelationByRelInMaxRaw[iMin][:numWordsToPrint[iMin]]
            topicWordRelationByRelInMaxRaw[iMin] = sorted(topicWordRelationByRelInMaxRaw[iMin],
                                                          key=lambda x: x[1][1], reverse=True)

        uniqueWordsinToken = [len(topicWordRelationByRelativeCount[iTopic]) for iTopic in range(numTopics)]
        uniqueWordsinToken = np.array(uniqueWordsinToken)

        # Write to a text file ( Uncomment if needed )
        fileHandler2 = open(filePathTask2Output, 'a')
        with fileHandler2:
            fileHandler2.write("K = %s, alpha = beta = %s\n" % (str(numTopics), str(dirParameter)))
            for iTopic in range(numTopics):
                fileHandler2.write(
                    "\n\nTopic number = %s, number of unique words in it = %s and total number of words in it = %s\n" % (
                        str(iTopic), str(uniqueWordsinToken[iTopic]), str(sumWordsinToken[iTopic])))
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

        topTopics = sumWordsinToken.argsort()[numTopics - topTopicsSize:]
        listHeader = ["removeMe"]
        for i in range(len(topTopics)):
            listHeader = listHeader + ["Topic {}".format(topTopics[i])]
        listHeader.pop(0)
        colHeaders = pd.MultiIndex.from_product([listHeader, ['Raw', 'Rel', 'RelRaw']])
        resultTopicDF = pd.DataFrame()
        # for iDFRow in range(min(numWordsToPrint)):
        for iDFRow in range(desiredWordsToBePrinted):
            tempRow = list()
            for iDFCell in range(len(topTopics)):
                try:
                    tempRow.append(topicWordRelationByRawCount[topTopics[iDFCell]][iDFRow][0])
                except:
                    tempRow.append("NA")
                try:
                    tempRow.append(topicWordRelationByRelativeCount[topTopics[iDFCell]][iDFRow][0])
                except:
                    tempRow.append("NA")
                try:
                    tempRow.append(topicWordRelationByRelInMaxRaw[topTopics[iDFCell]][iDFRow][0])
                except:
                    tempRow.append("NA")
            tempDF = pd.DataFrame([tempRow])
            if len(tempRow) > 0:
                resultTopicDF = resultTopicDF.append(tempDF, ignore_index=True)
            tempRow.clear()
        resultTopicDF.columns = colHeaders
        print(resultTopicDF.head(10).transpose())

        print("\n\n")

        # Result part - 3. Works fine(Hopefully)
        topicCount = list()
        topicCountPerc = list()
        maxTopicNumPerc = np.zeros((numDocs, 2), dtype=float)
        wordsInMaxTopic = list()
        iPosition = 0
        jPosition = 0
        for iDoc in range(numDocs):
            topicCountPerc.append(np.zeros(numTopics, dtype=float))
            topicCount.append(np.zeros(numTopics, dtype=int))
            wordsInMaxTopic.append(list())

            for iWord in range(jPosition, jPosition + books.docLen[iDoc]):
                topicCount[iDoc][wordTopicResultsT2[iCase][iWord]] += 1
            jPosition += books.docLen[iDoc]
            topicCountPerc[iDoc] = topicCount[iDoc] / books.docLen[iDoc]

            maxTopicNumPerc[iDoc][0] = int(topicCount[iDoc].argmax())
            maxTopicNumPerc[iDoc][1] = max(topicCountPerc[iDoc])

            for iWord in range(iPosition, iPosition + books.docLen[iDoc]):
                # if backupRel[wordTopicResultsT1[iCase][iWord]][books.listOfWords[iWord]] == 1:
                if wordTopicResultsT2[iCase][iWord] == maxTopicNumPerc[iDoc][0]:
                    wordsInMaxTopic[iDoc].append(books.listOfWords[iWord])
            iPosition += books.docLen[iDoc]

        documentTopicsDF = pd.DataFrame()

        documentTopicsDF.insert(0, "Document Number", np.arange(0, numDocs, 1))
        documentTopicsDF.insert(1, "Dominant Topic", maxTopicNumPerc[:, 0])
        documentTopicsDF.insert(2, "Percentage", maxTopicNumPerc[:, 1])
        documentTopicsDF.insert(3, "Words in dominant topic", wordsInMaxTopic)
        documentTopicsDF = documentTopicsDF.sort_values("Percentage", ascending=False)
        print(documentTopicsDF.head(10))
        print("\n\n")
        iCase += 1

print("Hello")