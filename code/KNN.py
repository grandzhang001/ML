#-*- coding:utf-8 -*-

import numpy as np
import operator

def createDataSet():
    """
    create the dataset
    :return: tuple(x1,x2):x1 data, x2 label
    """
    group=np.array([[1.0,1.0],[1.0,1.1],[0.0,0.0],[0.0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify(inX,dataSet,labels,k):
    """
    classify the input and return the label
    :param inX: the features of input
    :param dataSet: the sample dataset
    :param labels: the sample label
    :param k: KNN
    :return: input's label
    """
    #the sample dataset's numbers
    dataSetSize=dataSet.shape[0]
    #generate the same dimension set by inX with numpy.tile,then minus the sample
    #tile(A,b) repeat A with b times,if b is tuple, then repeat in every dimension
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #sort and return the data's order
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #sorted(),operator.itemgetter(n) return the n-1 dimension 's value
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

group,labels=createDataSet()

print classify([0,0],group,labels,3)