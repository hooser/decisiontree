import pandas as pd 
from math import log

'''
函数说明：读取csv原始数据，将str类型转换成数字
返回值：二维数字表(纵向)，标签列表，转换表
'''
def get_traindata():
	da = pd.read_csv('tennis.txt',header = None)
	lieshu = da.shape[1]
	labels = []
	labels = [da[i][0] for i in range(lieshu)]      #dataframe的属性标签
	da_no_labels = da[:][1:]                        #不带标签的原始文件
	transformeddata = []                            #转换后的数字总列表(二维)
	transformtable = {}                             #str 到 num 的转换表 {属性1:{取值1:0，取值2:1},属性2:{取值1:0，取值2:1}}       
	for i in range(lieshu):      
		num = 0
		transformeddatacolumn = []                  #记录每一列属性转换成数字
		word_to_num = {}                            #内部字典：记录 da_no_labels[i](每一个属性)的 每个取值 对应的数字 
		transformtable[labels[i]] = word_to_num
		for word in da_no_labels[i]:
			if word not in word_to_num.keys():
				word_to_num[word] = num             #记录str转换num
				num = num + 1
			transformeddatacolumn.append(word_to_num[word])
		transformeddata.append(transformeddatacolumn)
	'''
	transformeddata从横向转为纵向
	'''
	labelsnum = len(transformeddata)                #标签数目
	atrsnum = len(transformeddata[0])               #一个标签的属性数目
	dataset = []
	for i in range(atrsnum):
		rowdata = []
		for j in range(labelsnum):
			rowdata.append(transformeddata[j][i])
		dataset.append(rowdata)
	return dataset,labels,transformtable

'''
函数功能：获取测试数据并转换为二重list形式
'''
def get_testdata():
	testdf = pd.read_csv('test.txt',header = None)
	testlist = []
	for j in range(testdf.shape[0]):
		templist = []
		for i in range(testdf.shape[1]):
			templist.append(testdf[i][j])
		testlist.append(templist)
	return testlist

'''
函数功能：计算香农熵
返回值：香农熵
'''
def calcShannon(dataset):
	classrow = [example[-1] for example in dataset]                #截取类别的list
	numentires = len(classrow)                     #总的类别数目
	classnums = {}                                 #每种类别的数目
	for kind in classrow:
		if kind not in classnums.keys():
			classnums[kind] = 1
		else:
			 classnums[kind] += 1
	ent = 0.0
	for key in classnums:
		prob = float(classnums[key]) / numentires
		#print('calcShannon,prob=',prob)
		ent -= prob * log(prob,2)
	return ent

'''
函数功能：计算属性内在的信息率 -> 信息增益率 = 信息熵增益 / H
参数：axis->第axis列
'''
def calcInsideInfo(dataset,axis):
	atrsrow = [example[axis] for example in dataset]
	numentires = len(atrsrow)                      #属性列的总数目
	atrsnum = {}
	for atr in atrsrow:
		if atr not in atrsnum.keys():
			atrsnum[atr] = 1
		else:
			atrsnum[atr] += 1
	H = 0.0
	for key in atrsnum:
		prob = float(atrsnum[key] / numentires)
		#print('prob=',prob)
		H -= prob * log(prob,2)
	return H

'''
函数功能：将数据集按照某个标签(label)的某个属性分裂
返回值：子数据集
参数：axis->dataset的第axis个列表
'''
def splitDataset(dataset,axis,value):
	subDataset = []                         #分裂后的子数据集
	for featrow in dataset:
		subDatasetrow = []                        
		if featrow[axis] == value:
			subDatasetrow = featrow[:axis]
			subDatasetrow.extend(featrow[axis+1:])
			subDataset.append(subDatasetrow)
	return subDataset

'''
函数功能：计算单个属性的数据集的数目最多的类别
返回值：属性类别的数字形式
实现方式：反转字典 -> {数目：类别}
'''
def majority(dataset):
	classList = [example[-1] for example in dataset]
	classnums = {}                                    #记录每种类别的数目
	for kind in classList:
		if kind not in classnums.keys():
			classnums[kind] = 1
		else:
			classnums[kind] += 1
	classnums_t = {v:k for k,v in classnums.items()}  # 次数:类别
	temp = 0
	for key in classnums_t:
		if key > temp:
			temp = key
	return classnums_t[key]
'''
函数功能：选取最优(信息增益率最大)的属性分裂
返回值：dataset的纵向属性序号(axis)
'''
def Choosebestsplitfeat(dataset):
	featuresnum = len(dataset[0]) - 1       #特征数目
	bestfeature = -1                        #最优属性的序号
	bestInfoGainRatio = 0.0                 #最优信息增益率

	ent_before_split = calcShannon(dataset)    #分裂前的信息熵

	for i in range(featuresnum):
		atrscolumn = [example[i] for example in dataset]     #截取第i个特征的列(属性列表)
		DatasetLen = len(dataset)                            #截取的列总atr数目
		uniqueatrs = set(atrscolumn)                         #独立的属性列表
		InfoGain = 0.0                                       #信息增益
		InfoGainRatio = 0.0
		ent_split = 0.0
		for atr in uniqueatrs:
			subDataset = splitDataset(dataset,i,atr)
			subDatasetlen = len(subDataset)
			prob = float(subDatasetlen) / DatasetLen   
			ent_split += prob * calcShannon(subDataset)
		InfoGain = ent_before_split - ent_split              
		H = calcInsideInfo(dataset,i) 
		InfoGainRatio = InfoGain / H
		if InfoGainRatio > bestInfoGainRatio:
			bestInfoGainRatio = InfoGainRatio
			bestfeature = i
	return bestfeature

'''
函数功能：创建类别字典
参数：className->类别名称, 字典形式->{'no':1,'yes':0}
'''
def create_classDict(className,transformtable):
	classDict = {}
	tempDict = transformtable[className]
	classDict = {v:k for k,v in tempDict.items()}
	return classDict

'''
函数功能：递归建立决策树 递归终止条件-> 1.全部样本属于一个类别 2.只剩一个特征
参数：dataset->数字化处理后的数据集 labels->标签(包含类别) className->类别特征名
'''
def createDT(dataset,labels,classDict):
	classList = [example[-1] for example in dataset]
	if classList.count(classList[0]) == len(classList):   #只有一个类别
		return classDict[classList[0]]
	if len(dataset[0]) == 2:                              #只有一个属性,返回该属性中较多的类别
		return classDict[majority(dataset)]
	bestfeatureindex = Choosebestsplitfeat(dataset)       #最优属性标签序号
	bestfeaturelabel = labels[bestfeatureindex]           #最优属性标签
	new_labels = labels[:bestfeatureindex]
	new_labels.extend(labels[bestfeatureindex+1:])
	myTree = {bestfeaturelabel:{}}
	featValues = [example[bestfeatureindex] for example in dataset]
	uniqueVal = set(featValues)
	for value in uniqueVal:                                 
		myTree[bestfeaturelabel][value] = createDT(splitDataset(dataset,bestfeatureindex,value),new_labels,classDict)
	return myTree

'''
函数功能：找出给定标签(key)在测试数字串中对应的数字
参数：labels->标签数组 numberlist->数字串(单列表) label->key
'''
def label_to_num(labels_noclass,numberlist,label):  
	labelslen = len(labels)
	for i in range(labelslen):
		if labels[i] == label:
			return numberlist[i]

'''
函数功能：判断给定的数字串的类别，对于超出了决策树判断范围的输入测试会进行提示
参数： labels->标签数组 numberlist->数字串
'''
def getinto(dic,labels_noclass,numberlist):  
	if dic in ['Yes','No']:                 #假定为二分类，只有yes和no两个类别
		return dic
	else:
		nextnum = label_to_num(labels_noclass,numberlist,list(dic.keys())[0])
		newdic = dic
		newdic = newdic[list(dic.keys())[0]]      #读字符串   
		if nextnum not in newdic.keys():
			print('该决策树不能处理这个输入样本！')  
			return -1
		else:                      
			newdic = newdic[nextnum]
		return getinto(newdic,labels,numberlist)

'''
函数功能：将测试集中的一串字符串转换为数字串
返回值：strlist根据tranformtable得到的数字串
'''
def strlist_to_numberlist(transformtable,strlist,labels_noclass):
	numberlist = []
	lenstrlist = len(strlist)               #字符串列表的长度
	for i in range(lenstrlist):

		numberlist.extend([transformtable[labels_noclass[i]][strlist[i]]])
	return numberlist

'''
函数功能：建立决策树，使用决策树对测试集做决策
'''
def makedecisions(dataset,labels_noclass,classDict,testSet,transformtable):
	myTree = createDT(dataset,labels_noclass,classDict)
	numtestSet = len(testSet)                              #测试集的样本数目
	answer = []
	for i in range(numtestSet):
		numberlist = strlist_to_numberlist(transformtable,testSet[i],labels_noclass)
		newmyTree = myTree
		temp_ans = getinto(newmyTree,labels_noclass,numberlist)
		if temp_ans == -1:
			print(testSet[i])
		else:
			answer.append(temp_ans)
	return answer

if __name__ == '__main__':
	dataset,labels,transformtable = get_traindata()
	className = labels[-1]
	labels_noclass = labels[:len(labels) - 1]
	classDict = create_classDict(className,transformtable)
	testlist = get_testdata()
	answer = makedecisions(dataset,labels_noclass,classDict,testlist,transformtable)
	print(answer)
	

	

