from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

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


if __name__ == "__main__":
    group, labels = createDataSet()
    result = classify0([0,0], group, labels, 3)
    print(result)