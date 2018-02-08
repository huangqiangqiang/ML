from numpy import *
import operator
import re
import feedparser

#---------------------------------------------------------------------------
# 朴素贝叶斯 
# 朴素意思是1.把每个特征看作是相互独立的。2.每个特征同等重要。
# P(x,y|z)
# 这里的竖杆代表条件概率，表示在给定事件z的条件下，事件x和y都发生的概率 
#---------------------------------------------------------------------------


'''
生成数据集
'''
def loadDataSet():
    postingList = [
        ['my','dog','has','flea','problems','help','please'],
        ['maybe','not','take','him','to','dog','park','stupid'],
        ['my','dalmation','is','so','cute','i','love','him'],
        ['stop','posting','stupid','worthless','garbage'],
        ['mr','licks','ate','my','steak','how','to','stop','him'],
        ['quit','buying','worthless','dog','food','stupid']
    ]
    # 0代表正常言论，1代表侮辱性文字
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

'''
从数据集中创建词汇表
'''
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

'''
记录inputSet数据中出现的单词，出现记为1，没出现为0，
因为此处只记录有没有出现，所以是词集模型，如果还想记录出现的个数，就是词袋模型
'''
def setOfWords2Vec(vocabList, inputSet):
    # 初始化向量，长度和词汇表长度相同，并且元素索引一一对应
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # vocabList.index(word)查询word在vocabList中的索引，在returnVec中记为1
            returnVec[vocabList.index(word)] = 1
            # 词袋模型
            #returnVec[vocabList.index(word)] += 1
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return returnVec

'''
词袋模型
'''
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return returnVec

'''
朴素贝叶斯训练函数

trainMatrix     文档矩阵
trainCategory   类别标签向量
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 侮辱性文字的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    # 遍历数据集
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 侮辱性词出现的次数
            p1Num += trainMatrix[i]
            # 侮辱性语句总词数
            p1Denom += sum(trainMatrix[i])
        else:
            # 正常的词出现的次数
            p0Num += trainMatrix[i]
            # 正常语句总词数
            p0Denom += sum(trainMatrix[i])
    # p1Vect中每个元素的值代表属于该类的概率
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    # 重点：公式：p(c(i)|w) = p(w|c(i)) * p(c(i)) / p(w)
    # 这个就是预测公式，对于一个待预测的向量，就是p(w)，我们需要求的就是p(c(i)|w)，trainNB0这个函数就是根据训练集算出p(w|c(i))和p(c(i))
    # p0Vect和p1Vect相当于p(w|c(i))中i=0和i=1的概率，pAbusive是c(1)的概率，c(0)的概率是1-pAbusive
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # sum([0,1,0,0,1,0,...,0,0] * log(p1Num / p1Denom)) + log(pClass1)
    # sum([0,log(p1Num / p1Denom),0,0,log(p1Num / p1Denom),0,...,0,0])
    # log(p1Num / p1Denom) + log(p1Num / p1Denom) + ... + log(pClass1)
    # log(p1Num / p1Denom * p1Num / p1Denom * ... * pClass1)
    # p1Num / p1Denom * p1Num / p1Denom * ... * pClass1
    # 因为sum一项算出来是带log的，所以pClass也要加上log，logAB = logA + logB
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def textParse(bigString):
    listsOfTokens = re.split('\\W*', bigString)
    return [token.lower() for token in listsOfTokens if len(token)>0]

'''
测试算法（验证垃圾邮件）
'''
def spamTest():
    # 存放每条邮件的词
    docList = []
    classList = []
    fulltext = []
    # 生成类别标签向量，
    for i in range(1, 26):
        # filePath = 'email/spam/%d.txt' % i
        with open('email/spam/%d.txt' % i, encoding='utf-8') as fr:
            wordList = textParse(fr.read())
        docList.append(wordList)
        fulltext.extend(wordList)
        classList.append(1)
        with open('email/ham/%d.txt' % i, encoding='utf-8') as fr:
            wordList = textParse(fr.read())
        docList.append(wordList)
        fulltext.extend(wordList)
        classList.append(0)
    # 生成词汇表
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    # 生成测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    # 生成训练集
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    # 训练参数
    p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    # 用测试集测试错误率
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))

'''
返回前30个高频词汇
'''
def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    docList = []
    classList = []
    fulltext = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fulltext.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fulltext.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fulltext)
    # 取出前30个高频词汇
    for pairW in  top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        # 因为去掉了高频词汇，所以有些词没有在词汇表里
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))
    # 参数key=lambda pair: pair[1] 表示按每个元素的第一个参数排序
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print('SF**' * 20)
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print('NY**' * 20)
    for item in sortedNY:
        print(item[0])


if __name__ == "__main__":
    # 垃圾邮件分类
    # spamTest()

    # 通过rss分析不同地域相关用词
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    getTopWords(ny, sf)


