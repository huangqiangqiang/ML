from numpy import *
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------
# logistic回归
# 极大似然法：对于最大似然法，当从模型总体随机抽取n组样本观测值后，最合理的参数估计量应该使得从模型中抽取该n组样本观测值的概率最大。
# 极大似然法：对于最大似然法，当从数据集中随机抽取n组样本的y值后，最合理的w向量应该使得从模型中抽取该n组样本y值的概率最大。
#---------------------------------------------------------------------------


def loadDataSet():
    dataMat = []
    labels = []
    with open('testSet.txt', encoding='utf-8') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labels.append(int(lineArr[-1]))
    return dataMat, labels

'''
sigmoid函数
'''
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

'''
梯度上升算法
'''
def gradAscent(dataSet, labels):
    # 转换成numpy中的矩阵类型
    dataMatrix = mat(dataSet)
    # transpose 表示对矩阵转置
    labelsMatrix = mat(labels).transpose()
    # 矩阵的维度
    m,n = shape(dataMatrix)
    # 梯度上升算法的步长
    alpha = 0.001
    # 最大迭代次数
    maxCycles = 500
    # weights就是最后要求的系数矩阵，初始化为n行1列的矩阵
    weights = ones((n,1))
    for i in range(maxCycles):
        # dataMatrix * weights 的结果是weights[0] * dataMatrix[0] + weights[1] * dataMatrix[1] + ... + weights[n] * dataMatrix[n]
        # y = w0x0 + w1x1 + w2x2
        # 这里假设我们的预测函数是sigmoid(dataMatrix * weights)，算出来的h是预测值
        h = sigmoid(dataMatrix * weights)
        # 误差 = 真实值 - 预测值
        error = labelsMatrix - h
        # 迭代weights
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

'''
随机梯度上升
'''
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

'''
随机梯度上升
'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 每次迭代调整a
            alpha = 0.01 + 4 / (1.0 + i + j)
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

'''
画出决策边界
'''
def plotBestFit(weights):
    dataMat, labelsMat = loadDataSet()
    dataArr = array(dataMat)
    # 训练数据条数
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    # 保存数据集的两个特征为x，y坐标
    for i in range(n):
        if int(labelsMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # 画出决策边界
    # x为numpy.arange格式，并且以0.1为步长从-3.0到3.0切分
    x = arange(-3.0, 3.0, 0.1)
    # 因为坐标轴是X2和X1，画出决策边界y=kx+b就可以把X2看做y，X1看做x，weight[1]就是k
    # 0 = w0x0+w1x1+w2x2
    # w2x2 = -w0x0 - w1x1
    # x2 = (-w0x0 - w1x1) / w2
    # 因为x0 = 1，把x看做x1，那么y就是对应x1一个矩阵
    # y = (-w0 - w1 * x) / w2 
    y = (-weights[0] - weights[1] * x) / weights[2]
    # 这里的x和y都是矩阵，按照矩阵中对应的x和y一个个画出来，画出来的一个个点会以直线相连
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


#----------------------------------------------------------------------
# 案例
#----------------------------------------------------------------------
'''
预测函数
'''
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0

def colicTest():
    trainingSet = []
    trainingLabels = []
    # 使用训练集训练参数
    with open('horseColicTraining.txt') as frTrain:
        for line in frTrain.readlines():
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    # 使用测试集测试错误率
    with open('horseColicTest.txt') as frTest:
        for line in frTest.readlines():
            numTestVec += 1.0
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
                errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print('the error rate of this test is: %f' % errorRate)
    return errorRate

'''
多次调用colicTest求平均值
'''
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in  range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is %f' % (numTests, errorSum/float(numTests)))

if __name__ == '__main__':
    # dataMat, labelsMat = loadDataSet()
    # weights = gradAscent(dataMat, labelsMat)
    # getA() 将weights矩阵转换为数组
    # plotBestFit(weights.getA())

    # dataMat, labelsMat = loadDataSet()
    # weights = stocGradAscent1(array(dataMat), labelsMat)
    # plotBestFit(weights)

    multiTest()
