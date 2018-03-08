#-*- coding:utf-8 -*-

import operator
from math import log

def createDataSet():
    """
    create the dataset
    :return: dataset,labels
    """
    dataSet=[
        [1,1,'yes'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no'],
    ]
    labels=['no surfacing','flippers']
    return dataSet,labels

def calcShannonEnt(dataSet):
    """

    :param dataSet:
    :return: the dataset's entropy
    """
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    """

    :param dataSet:
    :param axis: the special feature
    :param value: feature's value
    :return: dataset with the special feature's value=value
    """
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """

    :param dataSet:
    :return: the order of the bestfeature
    """
    numFeatures=len(dataSet[0])-1
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet]
        uniqueVals=set(featList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEnt(subDataSet)
        infoGain=baseEntropy-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

def majorityCnt(classList):
    """

    :param classList:
    :return: the most class in classList
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    """

    :param dataSet:
    :param labels:
    :return: Decision Tree
    """
    classList=[example[-1] for example in dataSet]
    #dataset has only one class,return the class
    if classList.count(classList[0])==len(classList):
        return classList[0]
    #dataset has no any feature,return the majorityCnt
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

def getNumLeafs(myTree):
    numLeafs=0
    firstStr=myTree.keys()[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth=0
    firstStr=myTree.keys()[0]
    secondDict=myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
        if thisDepth >maxDepth:
            maxDepth=thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees=[
        {'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},
        {'no surfacing':{0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}},
    ]
    return listOfTrees[i]

# myDat,labels=createDataSet()
# print calcShannonEnt(myDat)
# print splitDataSet(myDat,0,1)
# myTree=createTree(myDat,labels)
# myTree=retrieveTree(0)


import matplotlib.pyplot as plt

decisionNode=dict(boxstyle='sawtooth',fc='0.8')
leafNode=dict(boxstyle='round4',fc='0.8')
arrow_args=dict(arrowstyle='<-')

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    #plot the annotate
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',
                            xytext=centerPt,textcoords='axes fraction',
                            va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def plotMidText(cntrPt,parentPt,textString):
    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]
    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]
    createPlot.ax1.text(xMid,yMid,textString)

def plotTree(myTree,parentPt,nodeText):
    numLeafs=getNumLeafs(myTree)
    depth=getTreeDepth(myTree)
    firstStr=myTree.keys()[0]
    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeText)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))
    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD


def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff=-0.5/plotTree.totalW
    plotTree.yOff =1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()

# myTree['no surfacing'][3]='maybe'
# createPlot(myTree)

def classify(inputTree,featLabes,testVec):
    firstStr=inputTree.keys()[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabes.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classLabel=classify(secondDict[key],featLabes,testVec)
            else:
                classLabel=secondDict[key]
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    with open(filename,'w') as f:
        pickle.dump(inputTree,f)

def grabTree(filename):
    import pickle
    with open(filename,'r') as f:
        return pickle.load(f)


myDat,labels=createDataSet()
myTree=retrieveTree(0)
print classify(myTree,labels,[1,1])
print classify(myTree,labels,[1,0])
