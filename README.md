基于C4.5算法构建决策树
---------------------------
训练集和测试集：格式参考tennis.txt和test.txt，属性数目为任意正整数（但是当属性数目为1时就没有使用决策树的意义了）

构造树：使用构造树函数(createDT)会根据训练集构造一个多重字典，类似于{label1:{属性1:{label2:{}},属性2:{label3:{}}}},
如果想看具体构建的决策树，可以把createDT函数中的变量myTree打印。构造的方法为递归，递归终止条件为1.所有数据属于同一个样
本。 2.只有一个属性（此时，就根据该属性中占比最大的类别来定义该属性对应的类别）

决策：利用得到的决策树（myTree，多重字典），通过递归的匹配的方法对输入的测试数据进行分类，该部分的核心代码详见getinto函数。
makedecisions函数包括了构建树和决策的过程。

