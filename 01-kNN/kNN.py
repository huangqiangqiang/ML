from numpy import *
import operator
# 数据可视化模块
import matplotlib
import matplotlib.pyplot as plt

from os import listdir


"""
kNN分类算法

inX      输入向量   
dataSet  训练集
labels   训练集的标签向量，个数和训练集匹配
k        选择最近邻的数目
"""
def classify0(inX, dataSet, labels, k):
    # shape函数查看矩阵或数组的维数
    dataSetSize = dataSet.shape[0]
    # 1. 计算距离
    # tile(A,(B,C))表示A元素在行方向重复B次，列方向重复C次
    arr = tile(inX, (dataSetSize, 1))
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 把diffMat每个元素进行平方
    sqDiffMat = diffMat ** 2
    # axis=1 表示将一个矩阵的每一行向量相加
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5
    # 将 distances 中的元素从小到大排列，提取其对应的index(索引)，然后输出到 sortedDistIndicies
    # 所以 sortedDistIndicies 中存放的是 distances 中从小到大排列的索引值
    sortedDistIndicies = distances.argsort()
    classCount = {}

    # 2. 选择距离最小的k个点
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # 3. 排序
    # reverse=True 降序排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#------------------------------------------------------------------------------
'''
约会网站配对系统
'''

'''
从文件中读特征数据
'''
def file2matrix(filename):
    # 1. 得到文件行数
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    # 2. 创建矩阵
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 3. 解析文件数据
    for line in arrayOLines:
        # 去掉line结尾的\n（回车）
        line = line.strip()
        # 拆分成数组
        listFromLine = line.split('\t')
        # 设置每一行的矩阵数据 
        # e.g. 
        # mat[i,j:k]表示取出第i行的第j到第k个元素，左闭右开，省略j，k表示所有元素
        # mat[i:j,k]表示取出第i行到第j行里面的第k个元素，省略i，j表示所有行
        returnMat[index,:] = listFromLine[0:3]
        # 分类标签
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

'''
归一化特征值
newValue = (oldValue - min) / (max - min)
'''
def autoNorm(dataSet):
    # min(0)表示取每一列的最小值，min(1)表示取每一行的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    # / 表示normDataSet矩阵中的每个元素对应相除，这里并不是矩阵除法
    normDataSet = normDataSet / tile(ranges, (m , 1))
    return normDataSet, ranges, minVals

'''
测试分类器
'''
def datingClassTest():
    hoRatio = 0.10
    # 获取测试集
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # 特征归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    # 取出10%作为测试样本
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print('the total error rate is: %f' % (errorCount / numTestVecs))

'''
图形展示特征
'''
def pl():
    dataSetMat, dataLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minValues = autoNorm(dataSetMat)
    # figure类似画板的概念，是画纸的载体，但是具体画图等操作是在画纸上完成的。在pyplot中，画纸的概念对应的就是Axes/Subplot。
    fig = plt.figure()
    # add_subplot理解成画纸，里面传入的三个数字，前两个数字代表要生成几行几列的子图矩阵，底单个数字代表选中的子图位置。
    ax = fig.add_subplot(121)
    # scatter表示散点图，这里表示使用矩阵的第二，第三列特征值画出散点图，s表示形状大小，c表示颜色
    ax.scatter(dataSetMat[:,1],dataSetMat[:,2], s = 15.0 * array(dataLabels), c = 15.0 * array(dataLabels))
    ax = fig.add_subplot(122)
    ax.scatter(dataSetMat[:,0],dataSetMat[:,1], s = 15.0 * array(dataLabels), c = 15.0 * array(dataLabels))
    plt.show()

'''
预测函数
'''
def classifyPerson():
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    resultList = ['not at all', 'in small doses', 'in large doses']
    inArr = [ffMiles, percentTats, iceCream]
    dataSetMat, dataLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(dataSetMat)
    # 输入特征值归一化
    normInArr = (inArr - minVals) / ranges
    classifierResult = classify0(normInArr, normMat, dataLabels, 3)
    print('you will probably like this person: %s' % resultList[classifierResult - 1])


#------------------------------------------------------------------------------
'''
手写数字识别系统
'''

'''
图片转向量，每张图片是32*32
'''
def img2vector(filename):
    fr = open(filename)
    # 创建1行1024列的矩阵
    returnVect = zeros((1, 1024))
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect

'''
手写数字识别系统测试代码
'''
def handWritingClassTest():
    # 1. 训练集
    # 分类标签数据
    hwLabels = []
    path = 'digits/trainingDigits'
    trainingFileList = listdir(path)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        filename = trainingFileList[i]
        filepath = path + '/' + filename
        # 设置训练集
        trainingMat[i,:] = img2vector(filepath)
        # 设置训练集标签数据
        filename = filename.split('.')[0]
        classNumStr = filename.split('_')[0]
        hwLabels.append(int(classNumStr))
    # 2. 测试集
    path = 'digits/testDigits'
    testingFileList = listdir(path)
    m = len(testingFileList)
    testingMat = zeros((m, 1024))
    errorCount = 0.0
    for i in range(m):
        filename = testingFileList[i]
        filepath = path + '/' + filename
        testingMat[i,:] = img2vector(filepath)
        classifyResult = classify0(testingMat[i,:],trainingMat, hwLabels, 3)
        filename = filename.split('.')[0]
        classNumStr = filename.split('_')[0]
        if classifyResult != int(classNumStr):
            errorCount += 1.0
        print('the classifier came back with: %d, the real answer is: %d' % (classifyResult, int(classNumStr)))
    print('the total number of errors is: %d' % errorCount)
    print('the total error rate is: %f' % (errorCount / m))

if __name__ == "__main__":
    # datingClassTest()
    # pl()
    # classifyPerson()
    handWritingClassTest()