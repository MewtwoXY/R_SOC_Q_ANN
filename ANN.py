import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
import os


class NeuralNet(object):
    # 神经网络训练类
    # 初始化神经网络
    def __init__(self, data_sizes):
        self.sizes = data_sizes
        self.num_layers_ = len(data_sizes)  # 层数

        # weights、biases初始化为正态分布随机数
        self.weights = [np.random.randn(j, i) for i, j in zip(data_sizes[:-1], data_sizes[1:])]
        self.biases = [np.random.randn(j, 1) for j in data_sizes[1:]]

    # 激活函数
    def actfun(self, data_wb, act):
        if act == 0:    # Sigmoid函数
            return 1.0 / (1.0 + np.exp(-data_wb))
        elif act == 1:    # tanh函数
            return (np.exp(data_wb) - np.exp(-data_wb)) / (np.exp(data_wb) + np.exp(-data_wb))
        elif act == 2:    # Relu函数
            return (np.sign(data_wb) + 1) * data_wb / 2
        else:    # Leaky ReLU函数
            return (np.sign(data_wb) + 1) * 0.45 * data_wb + 0.1 * data_wb

    # 激活函数的导函数
    def actfun_derivative(self, data_wb, act):
        if act == 0:
            return self.actfun(data_wb, act) * (1 - self.actfun(data_wb, act))
        elif act == 1:
            return 1 - self.actfun(data_wb, act) * self.actfun(data_wb, act)
        elif act == 2:
            return (np.sign(data_wb) + 1) / 2
        else:
            return (np.sign(data_wb) + 1) * 0.45 + 0.1

    # 前馈
    def feedforward(self, x, act):
        for b, w in zip(self.biases, self.weights):
            x = self.actfun(np.dot(w, x) + b, act)
        return x

    # 反向传播
    def backprop(self, x, y, act):
        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        nabla_biases = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            data_wb = np.dot(w, activation) + b
            zs.append(data_wb)
            activation = self.actfun(data_wb, act)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * \
            self.actfun_derivative(zs[-1], act)
        nabla_weights[-1] = np.dot(delta, activations[-2].transpose())
        nabla_biases[-1] = delta

        for i in range(2, self.num_layers_):
            data_wb = zs[-i]
            sp = self.actfun_derivative(data_wb, act)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sp
            nabla_weights[-i] = np.dot(delta, activations[-i - 1].transpose())
            nabla_biases[-i] = delta
        return nabla_biases, nabla_weights

    # 更新权值和截距
    def update_mini_batch(self, mini_batch, eta, act):
        nabla_biases = [np.zeros(bias.shape) for bias in self.biases]
        nabla_weights = [np.zeros(weight.shape) for weight in self.weights]
        for i, j in mini_batch:
            delta_nabla_biases, delta_nabla_weights = self.backprop(i, j, act)
            nabla_biases = [nb + dnb for nb, dnb in zip(nabla_biases, delta_nabla_biases)]
            nabla_weights = [nw + dnw for nw, dnw in zip(nabla_weights, delta_nabla_weights)]
        self.weights = [weight - (eta / len(mini_batch)) * nw for weight, nw in zip(self.weights, nabla_weights)]
        self.biases = [bias - (eta / len(mini_batch)) * nb for bias, nb in zip(self.biases, nabla_biases)]

    # 随机梯度下降算法
    def sgd(self, training_data, epochs, mini_batch_size, eta, act):
        n = len(training_data)
        for j in range(epochs):
            time_start = time.time()
            random.shuffle(training_data)
            mini_batches = [training_data[i:i + mini_batch_size] for i in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, act)
            time_end = time.time()
            time_need = (time_end - time_start) * (epochs - j - 1)
            minute, second = divmod(time_need, 60)
            hour, minute = divmod(minute, 60)
            print("\r·训练中....{:.2%},预计剩余时间：{:0}:{:02}:{:02}".format((j + 1) / epochs, int(hour), int(minute),
                                                                    int(second)), end="")
        print()

        return True

    def cost_derivative(self, output_activations, y):
        return output_activations - y

    # 预测
    def predict(self, in_var, act):
        value = self.feedforward(in_var, act)
        return value


class AnnWindow:
    # 神经网络训练窗口类
    def __init__(self):
        self.user_input = -1            # 用户输入
        self.data_check = False         # 运行正常检测
        self.time = ""                  # 系统时间
        self.data_excel = ""            # 用户输入的Excel文件路径
        self.data_choose = 0            # 用户的选择情况
        self.data_train_result = ""     # 用户输入的相关性分析文件保存路径
        self.data_c_save = ""           # 用户输入的保存C函数文件路径
        self.data_r = 0                 # 用户输入的电阻所在列数
        self.data_soc = ""              # 用户输入的SOC所在列数
        self.data_t = ""                # 用户输入的温度所在列数
        self.data_c = ""                # 用户输入的容量所在列数
        self.data_soh = ""              # 用户输入的SOH所在列数
        self.data_neuron = []           # 用户输入的中间层神经元数量
        self.data_epochs = 0            # 用户输入的训练次数
        self.data_mini_batch_size = 0   # 用户输入的每次训练样本数
        self.data_eta = 0               # 用户输入的学习率
        self.data_act = 0               # 用户输入的阈值函数类型
        self.data_read = []             # 读入的Excel数据
        self.data_read_x = []           # 读入的Excel自变量数据
        self.data_read_y = []           # 读入的Excel因变量数据
        self.data_cor = ""              # 相关性分析结果
        self.data_floor_input = 1       # 输入层变量数
        self.data_floor_output = 1      # 输出层变量数
        self.data_floor = [1, 1]        # 神经网络层数信息
        self.data_norm = ""             # 归一化数据
        self.data_x_max = []            # 归一化系数-自变量最大值
        self.data_x_min = []            # 归一化系数-自变量最小值
        self.data_y_max = 0             # 归一化系数-因变量最大值
        self.data_y_min = 0             # 归一化系数-因变量最小值
        self.data_arrange = []          # 训练整理结果

        self.data_net = NeuralNet(self.data_floor)  # 神经网络训练类实例

    def run(self):
        # 运行人工神经网络训练功能
        try:
            os.system("cls")
            print("\n·【注意：不要关闭此控制台窗口，否则会使整个平台关闭】")
            print("\n·【人工神经网络训练】\n")

            # 用户输入数据检测
            self.data_check = False
            self.data_check = self.data_input()
            if not self.data_check:
                return False

            # 获取系统时间
            self.time = str(time.strftime("%Y%m%d%H%M%S", time.localtime()))

            # 用户选择判断
            self.data_check = False
            self.data_check = self.data_user_choose()
            if not self.data_check:
                return False

            # C语言函数文件名设置
            self.data_check = False
            self.data_check = self.name_change()
            if not self.data_check:
                return False

            # 处理Excel数据文件
            self.data_check = False
            self.data_check = self.data()
            if not self.data_check:
                return False

            # 数据确认
            self.data_check = False
            self.data_check = self.data_right()
            if not self.data_check:
                return False

            # 神经网络层数导入
            self.data_check = False
            self.data_check = self.floor_init()
            if not self.data_check:
                return False

            # 神经网络归一化
            self.data_check = False
            self.data_check = self.norm()
            if not self.data_check:
                return False

            # 神经网络训练
            self.data_check = False
            self.data_check = self.data_net.sgd(self.data_norm, self.data_epochs, self.data_mini_batch_size,
                                                self.data_eta, self.data_act)
            if not self.data_check:
                return False

            # 训练结果整理
            self.data_check = False
            self.data_check = self.arrange()
            if not self.data_check:
                return False

            # 训练结果评估
            self.data_check = False
            self.data_check = self.evaluate()
            if not self.data_check:
                return False

            # 保存训练结果
            self.data_check = False
            self.data_check = self.excel_save()
            if not self.data_check:
                return False

            # 编写C语言函数
            self.data_check = False
            self.data_check = self.c_write()
            if not self.data_check:
                return False

        except Exception as err:
            print("\n·人工神经网络训练失败")
            print("·错误类型:\n·{}".format(err))
            return False

        print("\n·人工神经网络训练完毕")
        print("\n·【注意：不要关闭此控制台窗口，否则会使整个平台关闭】\n")

        return True

    def data_input(self):
        # 检测用户输入数据

        # 检测Excel数据文件
        print("·正在检测Excel数据文件...")
        if not os.path.isfile(self.data_excel):
            print("·数据检测失败，找不到Excel数据文件\n")
            return False
        elif self.data_excel[-4:] != ".xls" and self.data_excel[-5:] != ".xlsx":
            print("·数据检测失败，不是指定的Excel文件（xls、xlsx）\n")
            return False
        print("·Excel数据文件检测成功\n")

        # 检测训练结果保存路径
        print("·正在检测训练结果保存路径...")
        if not os.path.isdir(self.data_train_result):
            print("·数据检测失败，训练结果保存路径不是正确路径\n")
            return False
        print("·训练结果保存路径检测成功\n")

        # 检测C语言函数保存路径
        print("·正在检测C语言函数保存路径...")
        if not os.path.isdir(self.data_c_save):
            print("·数据检测失败，C语言函数保存路径不是正确路径\n")
            return False
        print("·C语言函数保存路径检测成功\n")

        # 检测电阻所在列数
        print("·正在检测电阻所在列数...")
        try:
            self.data_r = int(self.data_r)
        except ValueError:
            print("·数据检测失败，电阻所在列数应为整数\n")
            return False
        print("·电阻所在列数检测成功\n")

        # 检测SOC所在列数
        if self.data_soc != "":
            print("·正在检测SOC所在列数...")
            try:
                self.data_soc = int(self.data_soc)
            except ValueError:
                print("·数据检测失败，SOC所在列数应为整数\n")
                return False
            print("·SOC所在列数检测成功\n")

        # 检测温度所在列数
        if self.data_t != "":
            print("·正在检测温度所在列数...")
            try:
                self.data_t = int(self.data_t)
            except ValueError:
                print("·数据检测失败，温度所在列数应为整数\n")
                return False
            print("·温度所在列数检测成功\n")

        # 检测容量/SOH所在列数
        if self.data_c != "" and self.data_soh == "":
            print("·正在检测容量所在列数...")
            try:
                self.data_c = int(self.data_c)
            except ValueError:
                print("·数据检测失败，容量所在列数应为整数\n")
                return False
            print("·容量所在列数检测成功\n")

        elif self.data_c == "" and self.data_soh != "":
            print("·正在检测SOH所在列数...")
            try:
                self.data_soh = int(self.data_soh)
            except ValueError:
                print("·数据检测失败，SOH所在列数应为整数\n")
                return False
            print("·SOH所在列数检测成功\n")
        else:
            print("·数据检测失败，缺少因变量所在列数\n")
            return False

        # 检测中间层神经元数量
        print("·正在检测中间层神经元数量...")
        try:
            self.data_neuron = self.data_neuron.split(" ")
            self.data_neuron = [int(i) for i in self.data_neuron]
        except ValueError:
            print("·数据检测失败，中间层神经元数量应为空格隔开的整数\n")
            return False
        print("·中间层神经元数量检测成功\n")

        # 检测训练次数
        print("·正在检测训练次数...")
        try:
            self.data_epochs = int(self.data_epochs)
        except ValueError:
            print("·数据检测失败，训练次数应为整数\n")
            return False
        print("·训练次数检测成功\n")

        # 检测每次训练样本数
        print("·正在检测每次训练样本数...")
        try:
            self.data_mini_batch_size = int(self.data_mini_batch_size)
        except ValueError:
            print("·数据检测失败，每次训练样本数应为整数\n")
            return False
        print("·每次训练样本数检测成功\n")

        # 检测学习率
        print("·正在检测学习率...")
        try:
            self.data_eta = float(self.data_eta)
        except ValueError:
            print("·数据检测失败，学习率应为数字\n")
            return False
        print("·学习率检测成功\n")

        # 导入Excel数据
        print("·正在导入Excel数据...")
        try:
            self.data_read = np.array(pd.read_excel(self.data_excel))
        except Exception as err:
            print("·Excel数据导入失败\n")
            print("·错误类型:{}\n".format(err))
            return False
        print("·Excel数据导入成功\n")
        print("·数据检测成功\n")

        return True

    def data_user_choose(self):
        # 用户选择判断
        if self.data_soc != "":
            self.data_choose = self.data_choose + 100
        if self.data_t != "":
            self.data_choose = self.data_choose + 10
        if self.data_soh != "":
            self.data_choose = self.data_choose + 1

        return True

    def name_change(self):
        # C语言函数文件名设置
        if self.data_choose == 0:
            self.data_c_save = self.data_c_save + "\\r_c_ann_" + self.time + ".c"
        elif self.data_choose == 1:
            self.data_c_save = self.data_c_save + "\\r_soh_ann_" + self.time + ".c"
        elif self.data_choose == 10:
            self.data_c_save = self.data_c_save + "\\rt_c_ann_" + self.time + ".c"
        elif self.data_choose == 11:
            self.data_c_save = self.data_c_save + "\\rt_soh_ann_" + self.time + ".c"
        elif self.data_choose == 100:
            self.data_c_save = self.data_c_save + "\\rsoc_c_ann_" + self.time + ".c"
        elif self.data_choose == 101:
            self.data_c_save = self.data_c_save + "\\rsoc_soh_ann_" + self.time + ".c"
        elif self.data_choose == 110:
            self.data_c_save = self.data_c_save + "\\rsoct_c_ann_" + self.time + ".c"
        else:
            self.data_c_save = self.data_c_save + "\\rsoct_soh_ann_" + self.time + ".c"

        return True

    def data(self):
        # 读取数据文件
        self.data_read_x = list()
        if self.data_choose in [0, 1]:
            for i in self.data_read:
                self.data_read_x.append([i[self.data_r]])
        elif self.data_choose in [10, 11]:
            for i in self.data_read:
                self.data_read_x.append([i[self.data_r], i[self.data_t]])
        elif self.data_choose in [100, 101]:
            for i in self.data_read:
                self.data_read_x.append([i[self.data_r], i[self.data_soc]])
        else:
            for i in self.data_read:
                self.data_read_x.append([i[self.data_r], i[self.data_soc], i[self.data_t]])

        if self.data_choose in [0, 10, 100, 110]:
            self.data_read_y = np.zeros((len(self.data_read[:, self.data_c]), 1), dtype=float)
            for i in range(len(self.data_read[:, self.data_c])):
                self.data_read_y[i] = self.data_read[i][self.data_c]
        else:
            self.data_read_y = np.zeros((len(self.data_read[:, self.data_soh]), 1), dtype=float)
            for i in range(len(self.data_read[:, self.data_soh])):
                self.data_read_y[i] = self.data_read[i][self.data_soh]

        return True

    def data_right(self):
        # 输入数据确认
        if self.data_choose == 0:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的容量：{}mAh\n".format(self.data_read_x[0][0],
                                                                       self.data_read_y[0][0]))
        elif self.data_choose == 1:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的SOH：{}%\n".format(self.data_read_x[0][0],
                                                                      self.data_read_y[0][0]))
        elif self.data_choose == 10:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的温度：{}℃\n·第一个数据的容量：{}mAh\n".format(self
                                                                                      .data_read_x[0][0], self
                                                                                      .data_read_x[0][1], self
                                                                                      .data_read_y[0][0]))
        elif self.data_choose == 11:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的温度：{}℃\n·第一个数据的SOH：{}%\n".format(self
                                                                                     .data_read_x[0][0], self
                                                                                     .data_read_x[0][1], self
                                                                                     .data_read_y[0][0]))
        elif self.data_choose == 100:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的SOC：{}%\n·第一个数据的容量：{}mAh\n".format(
                self.data_read_x[0][0], self.data_read_x[0][1], self.data_read_y[0][0]))
        elif self.data_choose == 101:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的SOC：{}%\n·第一个数据的SOH：{}%\n".format(
                self.data_read_x[0][0], self.data_read_x[0][1], self.data_read_y[0][0]))
        elif self.data_choose == 110:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的SOC：{}%\n·第一个数据的温度：{}℃\n·第一个数据的"
                  "容量：{}mAh\n".format(self.data_read_x[0][0], self.data_read_x[0][1], self.data_read_x[0][2],
                                      self.data_read_y[0][0]))
        else:
            print( "·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的SOC：{}%\n·第一个数据的温度：{}℃\n·第一个数据的"
                   "SOH：{}%\n".format(self.data_read_x[0][0], self.data_read_x[0][1], self.data_read_x[0][2],
                                      self.data_read_y[0][0]))
        self.user_input = input("·如果数据正确请输入1，否则请输入0:")
        while self.user_input not in ["0", "1"]:
            print("·输入有误，请重新输入:")
            self.user_input = input("·如果数据正确请输入1，否则请输入0:")
        if self.user_input == "0":
            print("·数据导入确认失败，您可以重新设置数据")
            return False
        self.user_input = -1
        print("·数据导入确认成功\n")

        return True

    def floor_init(self):
        # 神经网络层导入
        self.data_floor_input = 1 + int(self.data_soc != "") + int(self.data_t != "")
        self.data_floor_output = 1
        self.data_floor = [self.data_floor_input]
        for i in range(len(self.data_neuron)):
            self.data_floor.append(self.data_neuron[i])
        self.data_floor.append(self.data_floor_output)
        self.data_net = NeuralNet(self.data_floor)

        return True

    def norm(self):
        # 对自变量进行归一化处理
        norm_x = [np.reshape(i, (self.data_floor_input, 1)) for i in self.data_read_x]
        self.data_x_max = list()
        self.data_x_min = list()
        for j in range(self.data_floor_input):
            self.data_x_max.append(float(max([i[j] for i in norm_x])))
            self.data_x_min.append(float(min([i[j] for i in norm_x])))
        for i in range(len(norm_x)):
            for j in range(self.data_floor_input):
                norm_x[i][j] = (norm_x[i][j] - self.data_x_min[j]) / (self.data_x_max[j] - self.data_x_min[j])
        # 对因变量进行归一化处理
        norm_y = self.data_read_y
        self.data_y_max = float(max([i for i in norm_y]))
        self.data_y_min = float(min([i for i in norm_y]))
        for i in range(len(norm_y)):
            norm_y[i] = (norm_y[i] - self.data_y_min) / (self.data_y_max - self.data_y_min)

        self.data_norm = list(zip(norm_x, norm_y))

        return True

    def arrange(self):
        # 将训练结果整理成由自变量，原始因变量，预测因变量组成的数组
        if self.data_floor_input == 1:
            for test_feature in self.data_norm:
                self.data_arrange.append([float(test_feature[0][0]), float(
                    float(test_feature[1][0]) * (self.data_y_max - self.data_y_min) + self.data_y_min), float(
                    self.data_net.predict(test_feature[0], self.data_act) * (
                            self.data_y_max - self.data_y_min) + self.data_y_min)])
        elif self.data_floor_input == 2:
            for test_feature in self.data_norm:
                self.data_arrange.append([float(test_feature[0][0]), float(test_feature[0][1]), float(
                    float(test_feature[1][0]) * (self.data_y_max - self.data_y_min) + self.data_y_min), float(
                    self.data_net.predict(test_feature[0], self.data_act) * (
                            self.data_y_max - self.data_y_min) + self.data_y_min)])
        else:
            for test_feature in self.data_norm:
                self.data_arrange.append(
                    [float(test_feature[0][0]), float(test_feature[0][1]), float(test_feature[0][2]),
                     float(float(test_feature[1][0]) * (self.data_y_max - self.data_y_min) + self.data_y_min), float(
                        self.data_net.predict(test_feature[0], self.data_act) * (
                                self.data_y_max - self.data_y_min) + self.data_y_min)])
        self.data_arrange = np.array(self.data_arrange)
        for i in self.data_arrange:
            for j in range(self.data_floor_input):
                i[j] = i[j] * (self.data_x_max[j] - self.data_x_min[j]) + self.data_x_min[j]

        return True

    def evaluate(self):
        # 对训练结果进行评估
        fy = self.data_arrange[:, -1]
        y = self.data_arrange[:, -2]
        y_d = fy - y  # 拟合结果误差统计
        ave_dis = np.mean(abs(y_d))
        ave_dis_rate = np.mean(abs(y_d) / y)
        st = y * 0.05
        if self.data_choose in [0, 10, 100, 110]:
            print("\n·拟合平均误差为:{:.2f}mAh，误差比例{:.2%}".format(ave_dis, ave_dis_rate))
        else:
            print("\n·拟合平均误差为:{:.2f}%，误差比例{:.2%}".format(ave_dis, ave_dis_rate))
        co = 0
        for i in range(len(y)):
            if abs(y_d[i]) <= st[i]:
                co = co + 1
        print("·误差小于等于5%的样本比例为：{:.2%}".format(co / len(y)))
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['Simhei']
        plt.rcParams['axes.unicode_minus'] = False
        if self.data_choose in [0, 10, 100, 110]:
            plt.xlabel("样本误差(mAh)")
        else:
            plt.xlabel("样本误差(%)")
        plt.ylabel("样本个数(个)")
        plt.title("误差统计")
        plt.hist(y_d, bins=25)
        plt.savefig(self.data_train_result + "\\error_statistics_0_100_" + self.time + ".png")
        plt.close()

        if self.data_choose in [100, 101, 110, 111]:
            data_x_soc = self.data_arrange[:, 1]
            c_fit2 = []
            c_test = []
            for i in range(len(y)):  # 去除左右SOC20%的结果输出
                if 20 <= int(data_x_soc[i]) <= 80:
                    c_fit2.append(fy[i])
                    c_test.append(y[i])
            c_test = np.array(c_test)
            print("\n·去除SOC:0~20%及80~100%的拟合结果：去除{}个数据，占比为{:.2%}".format(len(y) - len(c_test),
                                                                         (len(y) - len(c_test)) / len(y)))
            y_d = c_fit2 - c_test  # 拟合结果误差统计
            ave_dis = np.mean(abs(y_d))
            ave_dis_rate = np.mean(abs(y_d) / c_test)
            if self.data_choose in [100, 110]:
                print("\n·拟合平均误差为：{:.2f}mAh，误差比例{:.2%}".format(ave_dis, ave_dis_rate))
            else:
                print("\n·拟合平均误差为：{:.2f}%，误差比例{:.2%}".format(ave_dis, ave_dis_rate))
            co = 0
            st = c_test * 0.05
            for i in range(len(c_test)):
                if abs(y_d[i]) <= st[i]:
                    co = co + 1
            print("·误差小于等于5%的样本比例为：{:.2%}".format(co / len(c_test)))
            plt.figure()
            plt.rcParams['font.sans-serif'] = ['Simhei']
            plt.rcParams['axes.unicode_minus'] = False
            if self.data_choose in [100, 110]:
                plt.xlabel("样本误差(mAh)")
            else:
                plt.xlabel("样本误差(%)")
            plt.ylabel("样本个数(个)")
            plt.title("误差统计")
            plt.hist(y_d, bins=25)
            plt.savefig(self.data_train_result + "\\error_statistics_20_80_" + self.time + ".png")
            plt.close()

        return True

    def excel_save(self):
        # 保存训练结果
        print("\n·保存数据中...")
        self.data_train_result = self.data_train_result + "\\data_ann_train_result_" + self.time + ".xls"
        if self.data_choose in [0, 1]:
            ex = [self.data_arrange[:, 0], self.data_arrange[:, 1], self.data_arrange[:, 2],
                  self.data_arrange[:, 2] - self.data_arrange[:, 1]]
        elif self.data_choose in [10, 11, 100, 101]:
            ex = [self.data_arrange[:, 0], self.data_arrange[:, 1], self.data_arrange[:, 2], self.data_arrange[:, 3],
                  self.data_arrange[:, 3] - self.data_arrange[:, 2]]
        else:
            ex = [self.data_arrange[:, 0], self.data_arrange[:, 1], self.data_arrange[:, 2], self.data_arrange[:, 3],
                  self.data_arrange[:, 4], self.data_arrange[:, 4] - self.data_arrange[:, 3]]
        ex = np.array(np.transpose(ex))
        sh = pd.DataFrame(ex)
        if self.data_choose == 0:
            sh.rename(columns={0: "R/mΩ", 1: "C/mAh", 2: "C'/mAh", 3: "dis/mAh"}, inplace=True)
        elif self.data_choose == 1:
            sh.rename(columns={0: "R/mΩ", 1: "SOH/%", 2: "SOH'/%", 3: "dis/%"}, inplace=True)
        elif self.data_choose == 10:
            sh.rename(columns={0: "R/mΩ", 1: "T/℃", 2: "C/mAh", 3: "C'/mAh", 4: "dis/mAh"}, inplace=True)
        elif self.data_choose == 11:
            sh.rename(columns={0: "R/mΩ", 1: "T/℃", 2: "SOH/%", 3: "SOH'/%", 4: "dis/%"}, inplace=True)
        elif self.data_choose == 100:
            sh.rename(columns={0: "R/mΩ", 1: "SOC/%", 2: "C/mAh", 3: "C'/mAh", 4: "dis/mAh"}, inplace=True)
        elif self.data_choose == 101:
            sh.rename(columns={0: "R/mΩ", 1: "SOC/%", 2: "SOH/%", 3: "SOH'/%", 4: "dis/%"}, inplace=True)
        elif self.data_choose == 110:
            sh.rename(columns={0: "R/mΩ", 1: "SOC/%", 2: "T/℃", 3: "C/mAh", 4: "C'/mAh", 5: "dis/mAh"}, inplace=True)
        else:
            sh.rename(columns={0: "R/mΩ", 1: "SOC/%", 2: "T/℃", 3: "SOH/%", 4: "SOH'/%", 5: "dis/%"}, inplace=True)
        with pd.ExcelWriter(self.data_train_result) as writer:
            sh.to_excel(writer, sheet_name="数据结果")
        print("\n·数据保存成功")

        return True

    def list_py_c(self, list_py):
        # 将训练得到的权值和截距进行处理使其能够转化成C语言代码
        list_c = str(list_py)
        list_c = list_c.replace("]\n [", "},\n{")
        list_c = list_c.replace("[", "{")
        list_c = list_c.replace("]]", "}}")
        list_c = list_c.replace("\n ", " ")
        while "  " in list_c:
            list_c = list_c.replace("  ", " ")
        for i in "0123456789.":
            for j in "0123456789-":
                comma_need = i + " " + j
                comma_finish = i + ", " + j
                list_c = list_c.replace(comma_need, comma_finish)
        return list_c

    def c_write(self):
        # 编写C语言函数
        print("\n·编写C语言函数中...")
        np.set_printoptions(threshold=np.inf)  # 防止数组过长输出省略号
        p1 = "#include <stdio.h>\n#include <math.h>\n"
        p2 = ""
        for i in range(len(self.data_floor) - 1):
            p2 = p2 + "double w" + str(i + 1) + "[" + str(self.data_floor[i + 1]) + "][" + str(
                self.data_floor[i]) + "] = "
            p2 = p2 + self.list_py_c(self.data_net.weights[i]) + ";\n"
            p2 = p2 + "double b" + str(i + 1) + "[" + str(self.data_floor[i + 1]) + "][1] = "
            p2 = p2 + self.list_py_c(self.data_net.biases[i]) + ";\n"
            p2 = p2 + "double a" + str(i + 1) + "[" + str(self.data_floor[i + 1]) + "];\n"
        if self.data_act == 0:
            p3 = "double self.data_act_fun(double z)\n{\n    return 1 / (1 + exp(-z));\n}\n\n"
        elif self.data_act == 1:
            p3 = "double self.data_act_fun(double z)\n{\n    return (exp(z) - exp(-z)) / (exp(z) + exp(-z));\n}\n\n"
        elif self.data_act == 2:
            p3 = "double self.data_act_fun(double z)\n{\n    return z > 0 ? z : 0;\n}\n\n"
        else:
            p3 = "double self.data_act_fun(double z)\n{\n    return z > 0 ? z : 0.1 * z;\n}\n\n"
        if self.data_choose in [0, 1]:
            p2 = p2 + "double r_max = " + str(self.data_x_max[0]) + ";\ndouble r_min = " + str(self.data_x_min[0])
            if self.data_choose == 0:
                p4 = "double r_c_ann(double r)\n{\n    int i;\n\n    r = (r - r_min) / (r_max - r_min);\n\n    "
            else:
                p4 = "double r_soh_ann(double r)\n{\n    int i;\n\n    r = (r - r_min) / (r_max - r_min);\n\n    "
            p5 = "for(i=0;i<" + str(self.data_floor[1]) + ";i++)\n    {\n        a1[i] = self.data_act_fun(w1[i][0]" \
                                                          " * r + b1[i][0]);\n    }\n"
        elif self.data_choose in [10, 11]:
            p2 = p2 + "double r_max = " + str(self.data_x_max[0]) + ";\ndouble r_min = " + str(self.data_x_min[0]) +\
                 ";\ndouble t_max = " + str(self.data_x_max[1]) + ";\ndouble t_min = " + str(self.data_x_min[1])
            if self.data_choose == 10:
                p4 = "double rt_c_ann(double r, double t)\n{\n    int i;\n\n    r = (r - r_min) / (r_max - r_min);" \
                     "\n\n    t = (t - t_min) / (t_max - t_min);\n\n    "
            else:
                p4 = "double rt_soh_ann(double r, double t)\n{\n    int i;\n\n    r = (r - r_min) / (r_max - r_min);" \
                     "\n\n    t = (t - t_min) / (t_max - t_min);\n\n    "
            p5 = "for(i=0;i<" + str(self.data_floor[1]) + ";i++)\n    {\n        a1[i] = self.data_act_fun(w1[i][0]" \
                                                          " * r + w1[i][1] * t + b1[i][0]);\n    }\n"
        elif self.data_choose in [100, 101]:
            p2 = p2 + "double r_max = " + str(self.data_x_max[0]) + ";\ndouble r_min = " + str(self.data_x_min[0]) +\
                 ";\ndouble soc_max = " + str(self.data_x_max[1]) + ";\ndouble soc_min = " + str(self.data_x_min[1])
            if self.data_choose == 100:
                p4 = "double rsoc_c_ann(double r, double soc)\n{\n    int i;\n\n    r = (r - r_min) / (r_max - r_min)" \
                     ";\n\n    soc = (soc - soc_min) / (soc_max - soc_min);\n\n    "
            else:
                p4 = "double rsoc_soh_ann(double r, double soc)\n{\n    int i;\n\n    r = (r - r_min) / (r_max - " \
                     "r_min);\n\n    soc = (soc - soc_min) / (soc_max - soc_min);\n\n    "
            p5 = "for(i=0;i<" + str(self.data_floor[1]) + ";i++)\n    {\n        a1[i] = self.data_act_fun(w1[i][0]" \
                                                          " * r + w1[i][1] * soc + b1[i][0]);\n    }\n"
        else:
            p2 = p2 + "double r_max = " + str(self.data_x_max[0]) + ";\ndouble r_min = " + str(self.data_x_min[0]) +\
                 ";\ndouble soc_max = " + str(self.data_x_max[1]) + ";\ndouble soc_min = " + str(self.data_x_min[1]) +\
                 ";\ndouble t_max = " + str(self.data_x_max[2]) + ";\ndouble t_min = " + str(self.data_x_min[2])
            if self.data_choose == 110:
                p4 = "double rsoct_c_ann(double r, double soc, double t)\n{\n    int i;\n\n    r = (r - r_min) / " \
                     "(r_max - r_min);\n\n    soc = (soc - soc_min) / (soc_max - soc_min);\n\n    t = (t - t_min) / " \
                     "(t_max - t_min);\n\n    "
            else:
                p4 = "double rsoct_soh_ann(double r, double soc, double t)\n{\n    int i;\n\n    r = (r - r_min) / " \
                     "(r_max - r_min);\n\n    soc = (soc - soc_min) / (soc_max - soc_min);\n\n    t = (t - t_min) / " \
                     "(t_max - t_min);\n\n    "
            p5 = "for(i=0;i<" + str(self.data_floor[1]) + ";i++)\n    {\n        a1[i] = self.data_act_fun(w1[i][0]" \
                                                          " * r + w1[i][1] * soc + w1[i][2] * t + b1[i][0]);\n    }\n"

        if self.data_choose in [0, 10, 100, 110]:
            p2 = p2 + ";\ndouble c_max = " + str(self.data_y_max) + ";\ndouble c_min = " + str(self.data_y_min)\
                 + ";\n\n"
        else:
            p2 = p2 + ";\ndouble soh_max = " + str(self.data_y_max) + ";\ndouble soh_min = " + str(self.data_y_min)\
                 + ";\n\n"

        for i in range(len(self.data_floor) - 2):
            p5 = p5 + "    for(i=0;i<" + str(self.data_floor[i + 2]) + ";i++)\n    {\n        a" + str(
                i + 2) + "[i] = self.data_act_fun("
            for j in range(self.data_floor[i + 1]):
                p5 = p5 + "w" + str(i + 2) + "[i][" + str(j) + "] * a" + str(i + 1) + "[" + str(j) + "] + "
            p5 = p5 + "b" + str(i + 2) + "[i][0]);\n    }\n"
        if self.data_choose in [0, 10, 100, 110]:
            p6 = "\n    return a" + str(len(self.data_floor) - 1) + "[0] * (c_max - c_min) + c_min;\n}\n "
        else:
            p6 = "\n    return a" + str(len(self.data_floor) - 1) + "[0] * (soh_max - soh_min) + soh_min;\n}\n "
        p = p1 + p2 + p3 + p4 + p5 + p6
        f = open(self.data_c_save, "w+", encoding='utf-8')
        f.writelines(p)
        f.close()
        print("\n·编写C语言函数完成...")

        return True
