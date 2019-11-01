import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings(action='ignore')


class LineWindow:
    # 神经网络训练窗口类
    def __init__(self):
        self.user_input = -1            # 用户输入
        self.data_check = False         # 运行正常检测
        self.time = ""                  # 系统时间
        self.data_excel = ""            # 用户输入的Excel文件路径
        self.data_choose = 0            # 用户的选择情况
        self.data_train_result = ""     # 用户输入的训练结果保存路径
        self.data_c_save = ""           # 用户输入的保存C函数文件路径
        self.data_r = 0                 # 用户输入的电阻所在列数
        self.data_soc = ""              # 用户输入的SOC所在列数
        self.data_c = ""                # 用户输入的容量所在列数
        self.data_soh = ""              # 用户输入的SOH所在列数
        self.data_order = 1             # 用户输入的线性拟合阶数
        self.data_read = []             # 读入的Excel数据
        self.data_fit = []              # Excel数据根据SOC分组，共101组
        self.data_fx = []               # 根据SOC分成的101个拟合函数
        self.data_fit_result = []       # 拟合测试结果数组（SOC:0-100）
        self.data_fit_result2 = []      # 拟合测试结果数组（SOC:20-80）

    def run(self):
        # 运行线性拟合功能
        try:
            os.system("cls")
            print("\n·【注意：不要关闭此控制台窗口，否则会使整个平台关闭】")
            print("\n·【线性拟合】 阶数：{}阶\n".format(self.data_order))

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

            # 线性拟合
            self.data_check = False
            self.data_check = self.lin_fit()
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
            print("\n·线性拟合失败")
            print("·错误类型:\n·{}".format(err))
            return False

        print("·线性拟合完毕")
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

        # 导入Excel数据
        print("·正在导入Excel数据...")
        try:
            self.data_read = np.array(pd.read_excel(self.data_excel))
        except Exception as err:
            print("·Excel数据导入失败")
            print("·错误类型:{}\n".format(err))
            return False
        print("·Excel数据导入成功\n")
        print("·数据检测成功\n")

        return True

    def data_user_choose(self):
        # 用户选择判断
        self.data_choose = 0
        if self.data_soc != "":
            self.data_choose = self.data_choose + 10
        if self.data_soh != "":
            self.data_choose = self.data_choose + 1

        return True

    def name_change(self):
        # C语言函数文件名设置
        if self.data_choose == 0:
            self.data_c_save = self.data_c_save + "\\r_c_linfit_" + self.time + ".c"
        elif self.data_choose == 1:
            self.data_c_save = self.data_c_save + "\\r_soh_linfit_" + self.time + ".c"
        elif self.data_choose == 10:
            self.data_c_save = self.data_c_save + "\\rsoc_c_linfit_" + self.time + ".c"
        else:
            self.data_c_save = self.data_c_save + "\\rsoc_soh_linfit_" + self.time + ".c"

        return True

    def data(self):
        # 读取数据文件
        self.data_read = np.array(pd.read_excel(self.data_excel))  # 读取目标Excel数据并转化为数组
        if self.data_choose == 0:
            self.data_read = np.array([[i[self.data_r], i[self.data_c]] for i in self.data_read])
        elif self.data_choose == 1:
            self.data_read = np.array([[i[self.data_r], i[self.data_soh]] for i in self.data_read])
        elif self.data_choose == 10:
            self.data_read = np.array([[i[self.data_r], i[self.data_soc], i[self.data_c]] for i in self.data_read])
        else:
            self.data_read = np.array([[i[self.data_r], i[self.data_soc], i[self.data_soh]] for i in self.data_read])
        self.data_read = self.data_read.astype('float32')  # 将数据转化为float32格式

        return True

    def data_right(self):
        # 用户输入数据确认
        if self.data_choose == 0:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的容量：{}mAh\n".format(self.data_read[0][0],
                                                                       self.data_read[0][1]))
        elif self.data_choose == 1:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的SOH：{}%\n".format(self.data_read[0][0],
                                                                      self.data_read[0][1]))
        elif self.data_choose == 10:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的SOC：{}%\n·第一个数据的容量：{}mAh\n".format(
                self.data_read[0][0], self.data_read[0][1], self.data_read[0][2]))
        else:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的SOC：{}%\n·第一个数据的SOH：{}%\n".format(
                self.data_read[0][0], self.data_read[0][1], self.data_read[0][2]))

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

    def lin_fit(self):
        # 进行线性拟合
        if self.data_choose in [10, 11]:
            self.data_fit = [[] for i in range(101)]  # 建立空的101个拟合数据组
            for i in range(len(self.data_read)):
                self.data_fit[int(self.data_read[i, 1])].extend(
                    [[self.data_read[i, 0], self.data_read[i, 2]]])  # 将训练组数据根据SOC情况分成101个组
            co = 0
            for i in range(101):
                if self.data_fit[i]:
                    co = co + 1
                    print("\r·训练完整度:{:.2%}".format(co / 101), end="")
            if co != 101:
                print("·训练数据不完整，不包含SOC 0-100% 的全部数据，无法训练")
                return False
            print()
            self.data_fit = np.array(self.data_fit)  # 将列表转化为数组
            self.data_fit = self.data_fit.astype("float32")  # 将数据转化为float32格式
            self.data_fx = [[] for i in range(101)]  # 建立空的101个函数拟合组
            for i in range(101):
                fx = np.polyfit(self.data_fit[i, :, 0], self.data_fit[i, :, 1], self.data_order)  # 对每个SOC组进行拟合
                self.data_fx[i] = np.poly1d(fx)  # 存储每个拟合函数
        else:
            self.data_fit = self.data_read
            fx = np.polyfit(self.data_fit[:, 0], self.data_fit[:, 1], self.data_order)
            self.data_fx = np.poly1d(fx)

        for i in range(len(self.data_read)):  # 根据训练组拟合结果预测测试组
            if self.data_choose in [10, 11]:
                self.data_fit_result += [self.data_fx[int(self.data_read[i][1])](self.data_read[i][0])]
            else:
                self.data_fit_result += [self.data_fx(self.data_read[i][0])]

        return True

    def evaluate(self):
        # 对训练结果进行评估
        if self.data_choose in [10, 11]:
            # 拟合图像保存选择

            self.user_input = input("·保存指定SOC的拟合图像请输入2，保存每个SOC的拟合图像请输入1，不保存任何拟合图像请输入0:")
            while self.user_input not in ["0", "1", "2"]:
                print("·输入有误，请重新输入:")
                self.user_input = input("·如果数据正确请输入1，否则请输入0:")
            soc = int(self.user_input)
            self.user_input = -1

            # 根据选择保存图像
            if soc:
                if soc - 1:
                    i = 0
                    while i != -1:  # 绘制指定SOC的拟合图像
                        self.user_input = input("\n·请输入你想要保存拟合图像的SOC数值（整数），不想继续保存请输入-1：")
                        i_ok = 0
                        while i_ok == 0:
                            try:
                                i = int(self.user_input)
                                i_ok = 1
                                if i < -1 or i > 100:
                                    i_ok = 0
                            except ValueError:
                                i_ok = 0
                                print("·输入有误，请重新输入:")
                                self.user_input = input("\n·请输入你想要保存拟合图像的SOC数值（整数），不想继续保存请输入-1：")
                        self.user_input = -1
                        if i != -1:
                            y_d = self.data_fx[i](self.data_fit[i, :, 0]) - self.data_fit[i, :, 1]
                            ave = np.mean(abs(y_d))
                            ave_rate = np.mean(abs(y_d) / self.data_fit[i, :, 1])
                            if self.data_choose == 10:
                                print(
                                    "·SOC:{}%拟合平均误差:{:.2f}mAh，误差比例{:.2%}".format(i, ave, ave_rate))
                            else:
                                print(
                                    "·SOC:{}%拟合平均误差:{:.2f}%，误差比例{:.2%}".format(i, ave, ave_rate))
                            plt.figure()
                            plt.scatter(self.data_fit[i, :, 0], self.data_fit[i, :, 1])
                            x = np.arange(min(self.data_fit[i, :, 0]), max(self.data_fit[i, :, 0]), 0.1)
                            y = self.data_fx[i](x)
                            plt.scatter(x, y, c="r")
                            plt.rcParams['font.sans-serif'] = ['Simhei']
                            plt.rcParams['axes.unicode_minus'] = False
                            plt.xlabel('R/mΩ')
                            if self.data_choose == 10:
                                plt.ylabel('C/mAh')
                                plt.title('C-R(SOC:{}%)'.format(i))
                                plt.savefig(self.data_train_result + "\\C_R(SOC_{})_".format(i) + self.time + ".png")
                            else:
                                plt.ylabel('SOH/%')
                                plt.title('SOH-R(SOC:{}%)'.format(i))
                                plt.savefig(self.data_train_result + "\\SOH_R(SOC_{})_".format(i) + self.time + ".png")
                            plt.close()

                else:
                    for i in range(101):  # 绘制每个SOC的拟合图像
                        y_d = self.data_fx[i](self.data_fit[i, :, 0]) - self.data_fit[i, :, 1]
                        ave = np.mean(abs(y_d))
                        ave_rate = np.mean(abs(y_d) / self.data_fit[i, :, 1])
                        if self.data_choose == 10:
                            print(
                                "·SOC:{}%拟合平均误差:{:.2f}mAh，误差比例{:.2%}".format(i, ave, ave_rate))
                        else:
                            print(
                                "·SOC:{}%拟合平均误差:{:.2f}%，误差比例{:.2%}".format(i, ave, ave_rate))
                        plt.figure()
                        plt.scatter(self.data_fit[i, :, 0], self.data_fit[i, :, 1])
                        x = np.arange(min(self.data_fit[i, :, 0]), max(self.data_fit[i, :, 0]), 0.1)
                        y = self.data_fx[i](x)
                        plt.scatter(x, y, c="r")
                        plt.rcParams['font.sans-serif'] = ['Simhei']
                        plt.rcParams['axes.unicode_minus'] = False
                        plt.xlabel('R/mΩ')
                        if self.data_choose == 10:
                            plt.ylabel('C/mAh')
                            plt.title('C-R(SOC:{}%)'.format(i))
                            plt.savefig(self.data_train_result + "\\C_R(SOC_{})_".format(i) + self.time + ".png")
                        else:
                            plt.ylabel('SOH/%')
                            plt.title('SOH-R(SOC:{}%)'.format(i))
                            plt.savefig(self.data_train_result + "\\SOH_R(SOC_{})_".format(i) + self.time + ".png")
                        plt.close()
        else:
            plt.figure()
            plt.scatter(self.data_fit[:, 0], self.data_fit[:, 1])
            x = np.arange(min(self.data_fit[:, 0]), max(self.data_fit[:, 0]), 0.1)
            y = self.data_fx(x)
            plt.scatter(x, y, c="r")
            plt.rcParams['font.sans-serif'] = ['Simhei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.xlabel('R/mΩ')
            if self.data_choose == 0:
                plt.ylabel('C/mAh')
                plt.title('C-R')
                plt.savefig(self.data_train_result + "\\C_R_" + self.time + ".png")
            else:
                plt.ylabel('SOH/%')
                plt.title('SOH-R')
                plt.savefig(self.data_train_result + "\\SOH_R_" + self.time + ".png")
            plt.close()
        st = self.data_read[:, 1 + int(self.data_choose / 10)] * 0.05
        y_d = self.data_fit_result - self.data_read[:,
                                     1 + int(self.data_choose / 10)]  # 拟合结果误差统计
        ave_dis = np.mean(abs(y_d))
        ave_dis_rate = np.mean(abs(y_d) / self.data_read[:, 1 + int(self.data_choose / 10)])
        if self.data_choose in [0, 10]:
            print(
                "\n·拟合平均误差为:{:.2f}mAh，误差比例{:.2%}".format(ave_dis, ave_dis_rate))
        else:
            print(
                "\n·拟合平均误差为:{:.2f}%，误差比例{:.2%}".format(ave_dis, ave_dis_rate))
        co = 0
        for i in range(len(self.data_read)):
            if abs(y_d[i]) <= st[i]:
                co = co + 1
        print(
            "·误差小于等于5%的样本比例为：{:.2%}".format(co / len(self.data_read)))
        plt.figure()
        plt.rcParams['font.sans-serif'] = ['Simhei']
        plt.rcParams['axes.unicode_minus'] = False
        if self.data_choose in [0, 10]:
            plt.xlabel("样本误差(mAh)")
        else:
            plt.xlabel("样本误差(%)")
        plt.ylabel("样本个数(个)")
        plt.title("误差统计")
        plt.hist(y_d, bins=25)
        plt.savefig(self.data_train_result + "\\error_statistics_0_100_" + self.time + ".png")
        plt.close()

        if self.data_choose in [10, 11]:
            c_test = []
            for i in range(len(self.data_read)):  # 去除左右SOC20%的结果输出
                if 20 <= self.data_read[i][1] <= 80:
                    self.data_fit_result2 += [self.data_fx[int(self.data_read[i][1])](self.data_read[i][0])]
                    c_test += [self.data_read[i][2]]
            c_test = np.array(c_test)
            print("\n·去除SOC:0~20%及80~100%的拟合结果：去除{}个数据，占比为{:.2%}".format(len(
                self.data_read) - len(c_test), (len(self.data_read) - len(c_test)) / len(self.data_read)))
            y_d = self.data_fit_result2 - c_test  # 拟合结果误差统计
            ave_dis = np.mean(abs(y_d))
            ave_dis_rate = np.mean(abs(y_d) / c_test)
            if self.data_choose == 10:
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
            if self.data_choose == 10:
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
        if self.data_choose in [10, 11]:
            ex = [self.data_read[:, 0], self.data_read[:, 1], self.data_read[:, 2], self.data_fit_result,
                  self.data_fit_result - self.data_read[:, 2]]
            ex = np.array(np.transpose(ex))
            sh = pd.DataFrame(ex)
            if self.data_choose == 10:
                sh.rename(columns={0: "R/mΩ", 1: "SOC/%", 2: "C/mAh", 3: "C'/mAh", 4: "dis/mAh"}, inplace=True)
            else:
                sh.rename(columns={0: "R/mΩ", 1: "SOC/%", 2: "SOH/%", 3: "SOH'/%", 4: "dis/%"}, inplace=True)
        else:
            ex = [self.data_read[:, 0], self.data_read[:, 1], self.data_fit_result,
                  self.data_fit_result - self.data_read[:, 1]]
            ex = np.array(np.transpose(ex))
            sh = pd.DataFrame(ex)
            if self.data_choose == 0:
                sh.rename(columns={0: "R/mΩ", 1: "C/mAh", 2: "C'/mAh", 3: "dis/mAh"}, inplace=True)
            else:
                sh.rename(columns={0: "R/mΩ", 1: "SOH/%", 2: "SOH'/%", 3: "dis/%"}, inplace=True)
        with pd.ExcelWriter(self.data_train_result+ "\\data_line_train_result_" + self.time + ".xls") as writer:
            sh.to_excel(writer, sheet_name="数据结果")
        print("\n·数据保存成功")

        return True

    def c_write(self):
        # 编写C语言函数
        print("\n·编写C语言函数中...")
        if self.data_choose in [10, 11]:
            p1 = "#include <stdio.h>\n"
            fx = str([self.data_fx[:][i] for i in range(len(self.data_fx))])
            fx = fx.replace("poly1d([", "{")
            fx = fx.replace("])", "}")
            fx = fx.replace("[", "{")
            fx = fx.replace("]", "}")
            fx = fx.replace(" ", "")
            fx = fx.replace("},{", "},\n{")
            p2 = "double net[101][" + str(self.data_order + 1) + "] = " + fx + ";\n"
            if self.data_choose == 10:
                p3 = "double rsoc_c_linfit(double r,int soc)\n{\n  return "
            else:
                p3 = "double rsoc_soh_linfit(double r,int soc)\n{\n    return "
            p4 = ""
            for i in range(self.data_order + 1):
                p0 = "net[soc][" + str(i) + "]" + "*r" * (self.data_order - i)
                p4 = p4 + p0
                if self.data_order != i:
                    p4 = p4 + " + "
            p4 = p4 + ";\n}"
        else:
            p1 = "#include <stdio.h>\n"
            fx = str([self.data_fx[i] for i in range(len(self.data_fx) + 1)])
            fx = fx.replace("poly1d([", "{")
            fx = fx.replace("])", "}")
            fx = fx.replace("[", "{")
            fx = fx.replace("]", "}")
            fx = fx.replace(" ", "")
            fx = fx.replace("},{", "},\n{")
            p2 = "double net[" + str(self.data_order + 1) + "] = " + fx + ";\n"
            if self.data_choose == 0:
                p3 = "double r_c_linfit(double r)\n{\n return "
            else:
                p3 = "double r_soh_linfit(double r)\n{\n   return "
            p4 = ""
            for i in range(self.data_order + 1):
                p0 = "net[" + str(i) + "]" + "*r" * (self.data_order - i)
                p4 = p4 + p0
                if self.data_order != i:
                    p4 = p4 + " + "
            p4 = p4 + ";\n}"
        p = p1 + p2 + p3 + p4
        f = open(self.data_c_save, "w+", encoding='utf-8')
        f.writelines(p)
        f.close()
        print("\n·编写C语言函数完成...")

        return True
