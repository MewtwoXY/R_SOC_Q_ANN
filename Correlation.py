import numpy as np
import pandas as pd
import prettytable
import os
import time


class CorrelationWindow:
    # 相关性分析功能

    def __init__(self):
        # 相关性分析初始化
        self.user_input = -1            # 用户输入
        self.data_check = False         # 运行正常检测
        self.time = ""                  # 系统时间
        self.data_excel = ""            # 用户输入的Excel文件路径
        self.data_choose = 0            # 用户的选择情况
        self.data_train_result = ""     # 用户输入的相关性分析文件保存路径
        self.data_r = 0                 # 用户输入的电阻所在列数
        self.data_soc = ""              # 用户输入的SOC所在列数
        self.data_t = ""                # 用户输入的温度所在列数
        self.data_c = ""                # 用户输入的容量所在列数
        self.data_soh = ""              # 用户输入的SOH所在列数
        self.data_read = ""             # 读入的Excel数据
        self.data_cor = ""              # 相关性分析结果

    def run(self):
        # 运行相关性分析功能
        try:
            os.system("cls")
            print("\n·【注意：不要关闭此控制台窗口，否则会使整个平台关闭】\n")
            print("\n·【相关性分析】\n")

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

            # 进行相关性分析
            self.data_check = False
            self.data_check = self.corr_coefficient()
            if not self.data_check:
                return False

            # 保存分析文件
            self.data_check = False
            self.data_check = self.excel_save()
            if not self.data_check:
                return False

        except Exception as err:
            print("·相关性分析失败")
            print("·错误类型:\n·{}".format(err))
            return False

        print("\n·相关性分析完毕")
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

        # 检测相关性分析保存路径
        print("·正在检测相关性分析保存路径...")
        if not os.path.isdir(self.data_train_result):
            print("·数据检测失败，相关性分析保存路径不是正确路径\n")
            return False
        print("·相关性分析保存路径检测成功\n")

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

    def data(self):
        # 处理Excel数据文件
        print("·正在处理Excel数据...")
        if self.data_choose == 0:
            self.data_read = np.array([[i[self.data_r], i[self.data_c]] for i in self.data_read])
        elif self.data_choose == 1:
            self.data_read = np.array([[i[self.data_r], i[self.data_soh]] for i in self.data_read])
        elif self.data_choose == 10:
            self.data_read = np.array([[i[self.data_r], i[self.data_t], i[self.data_c]] for i in self.data_read])
        elif self.data_choose == 11:
            self.data_read = np.array([[i[self.data_r], i[self.data_t], i[self.data_soh]] for i in self.data_read])
        elif self.data_choose == 100:
            self.data_read = np.array([[i[self.data_r], i[self.data_soc], i[self.data_c]] for i in self.data_read])
        elif self.data_choose == 101:
            self.data_read = np.array([[i[self.data_r], i[self.data_soc], i[self.data_soh]] for i in self.data_read])
        elif self.data_choose == 110:
            self.data_read = np.array([[i[self.data_r], i[self.data_soc], i[self.data_t], i[self.data_c]] for i in
                                       self.data_read])
        else:
            self.data_read = np.array([[i[self.data_r], i[self.data_soc], i[self.data_t], i[self.data_soh]] for i in
                                       self.data_read])
        self.data_read = self.data_read.astype('float32')  # 将数据转化为float32格式
        print("·Excel数据处理成功\n")

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
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的温度：{}℃\n·第一个数据的容量：{}mAh\n"
                  .format(self.data_read[0][0], self.data_read[0][1], self.data_read[0][2]))
        elif self.data_choose == 11:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的温度：{}℃\n·第一个数据的SOH：{}%\n"
                  .format(self.data_read[0][0], self.data_read[0][1],self.data_read[0][2]))
        elif self.data_choose == 100:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的SOC：{}%\n·第一个数据的容量：{}mAh\n"
                  .format(self.data_read[0][0], self.data_read[0][1], self.data_read[0][2]))
        elif self.data_choose == 101:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的SOC：{}%\n·第一个数据的SOH：{}%\n"
                  .format(self.data_read[0][0], self.data_read[0][1], self.data_read[0][2]))
        elif self.data_choose == 110:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的SOC：{}%\n·第一个数据的温度：{}℃\n·第一个数据的"
                  "容量：{}mAh\n".format(self.data_read[0][0], self.data_read[0][1],self.data_read[0][2], self
                                      .data_read[0][3]))
        else:
            print("·导入数据确认：\n·第一个数据的电阻：{}mΩ\n·第一个数据的SOC：{}%\n·第一个数据的温度：{}℃\n·第一个数据的"
                  "SOH：{}%\n".format(self.data_read[0][0], self.data_read[0][1],self.data_read[0][2],
                                     self.data_read[0][3]))

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

    def corr_coefficient(self):
        # 进行相关性分析
        self.data_read = np.vstack(self.data_read)
        print("·相关性分析结果")
        if self.data_choose == 0:
            self.data_cor = np.corrcoef([self.data_read[:, 0], self.data_read[:, 1]])
            table = prettytable.PrettyTable(["", "电阻", "容量"])
            table.add_row(["  电阻  ", round(self.data_cor[0, 0], 4), round(self.data_cor[0, 1], 4)])
            table.add_row(["  容量  ", round(self.data_cor[1, 0], 4), round(self.data_cor[1, 1], 4)])
        elif self.data_choose == 1:
            self.data_cor = np.corrcoef([self.data_read[:, 0], self.data_read[:, 1]])
            table = prettytable.PrettyTable(["", "电阻", "SOH"])
            table.add_row(["  电阻  ", round(self.data_cor[0, 0], 4), round(self.data_cor[0, 1], 4)])
            table.add_row(["  SOH  ", round(self.data_cor[1, 0], 4), round(self.data_cor[1, 1], 4)])
        elif self.data_choose == 10:
            self.data_cor = np.corrcoef([self.data_read[:, 0], self.data_read[:, 1], self.data_read[:, 2]])
            table = prettytable.PrettyTable(["", "电阻", "温度", "容量"])
            table.add_row(["  电阻  ", round(self.data_cor[0, 0], 4), round(self.data_cor[0, 1], 4),
                           round(self.data_cor[0, 2], 4)])
            table.add_row(["  温度  ", round(self.data_cor[1, 0], 4), round(self.data_cor[1, 1], 4),
                           round(self.data_cor[1, 2], 4)])
            table.add_row(["  容量  ", round(self.data_cor[2, 0], 4), round(self.data_cor[2, 1], 4),
                           round(self.data_cor[2, 2], 4)])
        elif self.data_choose == 11:
            self.data_cor = np.corrcoef([self.data_read[:, 0], self.data_read[:, 1], self.data_read[:, 2]])
            table = prettytable.PrettyTable(["", "电阻", "温度", "容量"])
            table.add_row(["  电阻  ", round(self.data_cor[0, 0], 4), round(self.data_cor[0, 1], 4),
                           round(self.data_cor[0, 2], 4)])
            table.add_row(["  温度  ", round(self.data_cor[1, 0], 4), round(self.data_cor[1, 1], 4),
                           round(self.data_cor[1, 2], 4)])
            table.add_row(["  SOH  ", round(self.data_cor[2, 0], 4), round(self.data_cor[2, 1], 4),
                           round(self.data_cor[2, 2], 4)])
        elif self.data_choose == 100:
            self.data_cor = np.corrcoef([self.data_read[:, 0], self.data_read[:, 1], self.data_read[:, 2]])
            table = prettytable.PrettyTable(["", "电阻", "SOC", "容量"])
            table.add_row(["  电阻  ", round(self.data_cor[0, 0], 4), round(self.data_cor[0, 1], 4),
                           round(self.data_cor[0, 2], 4)])
            table.add_row(["  SOC  ", round(self.data_cor[1, 0], 4), round(self.data_cor[1, 1], 4),
                           round(self.data_cor[1, 2], 4)])
            table.add_row(["  容量  ", round(self.data_cor[2, 0], 4), round(self.data_cor[2, 1], 4),
                           round(self.data_cor[2, 2], 4)])
        elif self.data_choose == 101:
            self.data_cor = np.corrcoef([self.data_read[:, 0], self.data_read[:, 1], self.data_read[:, 2]])
            table = prettytable.PrettyTable(["", "电阻", "SOC", "SOH"])
            table.add_row(["  电阻  ", round(self.data_cor[0, 0], 4), round(self.data_cor[0, 1], 4),
                           round(self.data_cor[0, 2], 4)])
            table.add_row(["  SOC  ", round(self.data_cor[1, 0], 4), round(self.data_cor[1, 1], 4),
                           round(self.data_cor[1, 2], 4)])
            table.add_row(["  SOH  ", round(self.data_cor[2, 0], 4), round(self.data_cor[2, 1], 4),
                           round(self.data_cor[2, 2], 4)])
        elif self.data_choose == 110:
            self.data_cor = np.corrcoef(
                [self.data_read[:, 0], self.data_read[:, 1], self.data_read[:, 2], self.data_read[:, 3]])
            table = prettytable.PrettyTable(["", "电阻", "SOC", "温度", "容量"])
            table.add_row(
                ["  电阻  ", round(self.data_cor[0, 0], 4), round(self.data_cor[0, 1], 4), round(self.data_cor[0, 2], 4),
                 round(self.data_cor[0, 3], 4)])
            table.add_row(
                ["  SOC  ", round(self.data_cor[1, 0], 4), round(self.data_cor[1, 1], 4), round(self.data_cor[1, 2], 4),
                 round(self.data_cor[1, 3], 4)])
            table.add_row(
                ["  温度  ", round(self.data_cor[2, 0], 4), round(self.data_cor[2, 1], 4), round(self.data_cor[2, 2], 4),
                 round(self.data_cor[2, 3], 4)])
            table.add_row(
                ["  容量  ", round(self.data_cor[3, 0], 4), round(self.data_cor[3, 1], 4), round(self.data_cor[3, 2], 4),
                 round(self.data_cor[3, 3], 4)])
        else:
            self.data_cor = np.corrcoef(
                [self.data_read[:, 0], self.data_read[:, 1], self.data_read[:, 2], self.data_read[:, 3]])
            table = prettytable.PrettyTable(["", "电阻", "SOC", "温度", "SOH"])
            table.add_row(
                ["  电阻  ", round(self.data_cor[0, 0], 4), round(self.data_cor[0, 1], 4), round(self.data_cor[0, 2], 4),
                 round(self.data_cor[0, 3], 4)])
            table.add_row(
                ["  SOC  ", round(self.data_cor[1, 0], 4), round(self.data_cor[1, 1], 4), round(self.data_cor[1, 2], 4),
                 round(self.data_cor[1, 3], 4)])
            table.add_row(
                ["  温度  ", round(self.data_cor[2, 0], 4), round(self.data_cor[2, 1], 4), round(self.data_cor[2, 2], 4),
                 round(self.data_cor[2, 3], 4)])
            table.add_row(
                ["  SOH  ", round(self.data_cor[3, 0], 4), round(self.data_cor[3, 1], 4), round(self.data_cor[3, 2], 4),
                 round(self.data_cor[3, 3], 4)])
        print(str(table) + "\n")

        return True

    def excel_save(self):
        # 保存相关性分析结果
        print("\n·保存数据中...")
        self.data_train_result = self.data_train_result + "\\Correlation_" + self.time + ".xls"
        if self.data_choose == 0:
            ex = [self.data_cor[:, 0], self.data_cor[:, 1]]
            ex = np.array(np.transpose(ex))
            sh = pd.DataFrame(ex)
            sh.rename(index={0: "电阻", 1: "容量"}, columns={0: "电阻", 1: "容量"}, inplace=True)
        elif self.data_choose == 1:
            ex = [self.data_cor[:, 0], self.data_cor[:, 1]]
            ex = np.array(np.transpose(ex))
            sh = pd.DataFrame(ex)
            sh.rename(index={0: "电阻", 1: "SOH"}, columns={0: "电阻", 1: "SOH"}, inplace=True)
        elif self.data_choose == 10:
            ex = [self.data_cor[:, 0], self.data_cor[:, 1], self.data_cor[:, 2]]
            ex = np.array(np.transpose(ex))
            sh = pd.DataFrame(ex)
            sh.rename(index={0: "电阻", 1: "温度", 2: "容量"}, columns={0: "电阻", 1: "温度", 2: "容量"}, inplace=True)
        elif self.data_choose == 11:
            ex = [self.data_cor[:, 0], self.data_cor[:, 1], self.data_cor[:, 2]]
            ex = np.array(np.transpose(ex))
            sh = pd.DataFrame(ex)
            sh.rename(index={0: "电阻", 1: "温度", 2: "SOH"}, columns={0: "电阻", 1: "温度", 2: "SOH"}, inplace=True)
        elif self.data_choose == 100:
            ex = [self.data_cor[:, 0], self.data_cor[:, 1], self.data_cor[:, 2]]
            ex = np.array(np.transpose(ex))
            sh = pd.DataFrame(ex)
            sh.rename(index={0: "电阻", 1: "SOC", 2: "容量"}, columns={0: "电阻", 1: "SOC", 2: "容量"}, inplace=True)
        elif self.data_choose == 101:
            ex = [self.data_cor[:, 0], self.data_cor[:, 1], self.data_cor[:, 2]]
            ex = np.array(np.transpose(ex))
            sh = pd.DataFrame(ex)
            sh.rename(index={0: "电阻", 1: "SOC", 2: "SOH"}, columns={0: "电阻", 1: "SOC", 2: "SOH"}, inplace=True)
        elif self.data_choose == 110:
            ex = [self.data_cor[:, 0], self.data_cor[:, 1], self.data_cor[:, 2], self.data_cor[:, 3]]
            ex = np.array(np.transpose(ex))
            sh = pd.DataFrame(ex)
            sh.rename(index={0: "电阻", 1: "SOC", 2: "温度", 3: "容量"}, columns={0: "电阻", 1: "SOC", 2: "温度", 3: "容量"},
                      inplace=True)
        else:
            ex = [self.data_cor[:, 0], self.data_cor[:, 1], self.data_cor[:, 2], self.data_cor[:, 3]]
            ex = np.array(np.transpose(ex))
            sh = pd.DataFrame(ex)
            sh.rename(index={0: "电阻", 1: "SOC", 2: "温度", 3: "SOH"}, columns={0: "电阻", 1: "SOC", 2: "温度", 3: "SOH"},
                      inplace=True)

        with pd.ExcelWriter(self.data_train_result) as writer:
            sh.to_excel(writer, sheet_name="相关性")
        print("\n·数据保存成功")

        return True
