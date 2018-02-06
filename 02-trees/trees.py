from math import log
import operator
import treePlotter

'''
关于信息熵的理解可以参考，简单来说熵值越高表示信息量越大，越复杂
https://www.zhihu.com/question/22178202/answer/49929786
'''

'''
计算熵
'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 为所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 熵
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key] / numEntries)
        # 以2为底求对数
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

'''
划分数据集
'''
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # 把第axis个特征之前的元素存在新向量中
            reducedFeatVec = featVec[:axis]
            # 把第axis个特征之后的元素拼接到新向量中
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''
创建数据集
'''
def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
        ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

'''
选择最好的数据集划分方式
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        # example中取出第i个元素组成列表
        featList = [example[i] for example in dataSet]
        # 转成set集合
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # 把使用第i个特征拆分后的数据的熵加起来，这里乘上prob保证newEntropy不会大于baseEntropy
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 原来的熵值减去新的熵值，这里newEntropy越小，说明分割的越好
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

'''
返回类别最多的项
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

'''
创建树
创建树函数并不需要labels参数，但是为了给数据明确的含义，才提供这个参数
'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]

    # 如果类别完全相同，停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 递归时使用完所有的特征，仍不能将数据集划分成仅包含唯一类别的分组，则使用出现次数最多的类别作为返回值
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最好的特征划分数据集
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 取出该特征对应的说明
    bestFeatLabel = labels[bestFeat]
    # 使用字典存储树信息
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 复制labels，保证labels的原始值
        subLabels = labels[:]
        # 按该特征的每个枚举值拆分数据集
        subDataMat = splitDataSet(dataSet, bestFeat, value)
        subMyTree = createTree(subDataMat, subLabels)
        myTree[bestFeatLabel][value] = subMyTree
    return myTree

'''
决策树预测函数
'''
def classify(inputTree, featureLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featureIndex = featureLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featureIndex] == key:
            if type(secondDict[key]).__name__ == "dict":
                classLabel = classify(secondDict[key], featureLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

'''
保存已生成的决策树
'''
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb+')
    pickle.dump(inputTree, fw)
    fw.close()

'''
读取决策树
'''
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

def test():
    dataSet, labels = createDataSet()
    subLabels = labels[:]
    myTree = createTree(dataSet, subLabels)
    # treePlotter.createPlot(myTree)
    result = classify(myTree, labels, [1, 1])
    print(result)

if __name__ == "__main__":
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    treePlotter.createPlot(lensesTree)
    print(lensesTree)