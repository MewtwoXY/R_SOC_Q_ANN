# 新能源汽车应用场景，基于Python，通过神经网络训练锂离子电池使用相关数据，预测电池当前最大容量


1、功能设计
   界面设计：PyQt5、sys、os
   线程设计：threading、inspect、ctypes，numpy，
   pandas，prettytable，matplotlib，warnings
   相关文件：MainWindowFun.py、HelpWindowFun.py
   
   ===============================================================================
2、功能实现
   相关性分析：
   1.	对用户的输入数据进行检测，方防止输入错误数据
   2.	对用户的选择进行判断，因为输入时是将平台上所有的数据传入实例，但是用户选择的功能有不同的输入输入，所以需要对传入的数据进行判断，根据输入情况保存对应的选择编号data_choose
   3.	根据用户输入的数据及功能选择情况，对Excel数据进行处理，使其能够用于后面的分析
   4.	给出处理后第一组数据的情况，让用户判断自己的输入是否成功
   5.	进行相关性分析，使用了numpy库的相关性分析函数，同时用prettytable展示出来
   6.	保存分析结果至指定路径
   相关文件：Correlation.py
   神经网络训练：
   1.	开始的过程与相关性分析类似，检测输入内容、用户选择判断、处理Excel数据、用户输入判断，只增加了一个根据选择情况编写C语言函数名的步骤
   2.	神经网络层数导入，由于神经网络隐藏层神经元数量导入的是像“10 8 6”这种的用空格隔开数据的格式，需要对层数进行处理
   3.	归一化处理，由于数据具有不同的单位和数量级，需要对数据进行归一化处理，方便训练
   4.	神经网络训练，采用随机梯度下降算法对数据进行训练
   5.	对训练的结果进行整理，将训练输出的归一化结果还原为原数据
   6.	使用还原后数据对训练结果进行分析
   7.	导出Excel数据结果，与相关性分析类似
   8.	根据训练给出的系数编写C语言函数，使用字符串编写，最后保存为c文件
   相关文件：ANN.py
   线性拟合：
   1.	线性拟合的过程中，当自变量有soc时，拟合的过程为先根据0~100%的SOC值进行分组，在每个组里单独进行线性拟合	
   相关文件：LINE.py
