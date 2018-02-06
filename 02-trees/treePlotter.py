import matplotlib.pyplot as plt

# 解决中文乱码
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei']  

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

'''
绘制节点
nodeTxt：    节点说明文字
centerPt：   线段指向的位置
parentPt：   线段开始的位置
nodeType：   节点的样式
'''
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    
    createPlot.ax1.annotate(
        nodeTxt,                # 文字
        xy=parentPt,            # 箭头指向的终点坐标
        # xycoords='axes fraction',
        xytext=centerPt,        # 文字起始坐标
        # textcoords='axes fraction', 
        va='center',            # 垂直对齐
        ha='center',            # 水平对齐
        bbox=nodeType,          # 给文字设置边框的样式
        arrowprops=arrow_args   # 设置箭头
    )

'''
计算叶节点的个数
'''
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

'''
计算树的深度（决策节点的个数）
'''
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

'''
给两个节点之间的线段添加说明文字
'''
def plotMidText(centerPt, parentPt, txt):
    xMid = (parentPt[0] - centerPt[0]) / 2.0 + centerPt[0]
    yMid = (parentPt[1] - centerPt[1]) / 2.0 + centerPt[1]
    createPlot.ax1.text(xMid, yMid, txt)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    # 计算节点中心点的坐标
    centerPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 绘制线段的描述
    plotMidText(centerPt, parentPt, nodeTxt)
    # 绘制决策节点
    plotNode(firstStr, centerPt, parentPt, decisionNode)
    # 更新y偏移量，绘制下一层
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            # 决策节点
            plotTree(secondDict[key], centerPt, str(key))
        else:
            # 只剩下叶节点，开始绘制叶节点
            # 更新x的偏移量
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            # 绘制叶节点
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), centerPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), centerPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(myTree):
    # 这里的write是设置背景颜色
    fig = plt.figure(1, facecolor='white')
    # 清空当前画图窗口
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 可以把ax1看成画布 ，frameon=False 表示不绘制图形边框，axprops参数为了去掉xy坐标
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 树的宽度
    plotTree.totalW = float(getNumLeafs(myTree))
    # 树的深度
    plotTree.totalD = float(getTreeDepth(myTree))
    # 起始节点x的偏移量     1/plotTree.totalW表示让宽度等分，但此时还在中心点，再乘以0.5就是x的偏移量（假设节点的中心点在原点时x的偏移量）
    plotTree.xOff = -0.5/plotTree.totalW
    # 起始节点y的偏移量
    plotTree.yOff = 1.0
    # 开始绘制，传入起始坐标
    plotTree(myTree, (0.5, 1.0), '')
    plt.show()


if __name__ == "__main__":
    myTree = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}, 3: 'maybe'}}
    createPlot(myTree)