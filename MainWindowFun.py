from MainWindow import Ui_main_window
from HelpWindowFun import MyHelpWindow
from Correlation import CorrelationWindow
from ANN import AnnWindow
from LINE import LineWindow
import threading
import inspect
import ctypes
import sys
import os
if hasattr(sys, 'frozen'):  # PyQt5打包有bug需要自己设定PATH
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets


# 强制关闭线程
def _async_raise(tid, exc_type):
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exc_type):
        exc_type = type(exc_type)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exc_type))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:

        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


class MyThread(threading.Thread):
    # 设置多线程，用于在主窗口中运行后台程序，防止主窗口因阻塞而未响应
    def __init__(self, thread_id, name, class_instance):
        # 线程初始化，class_instance是具体功能类的实例
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name
        self.class_instance = class_instance

    def run(self):
        self.class_instance.run()


class MyMainWindow(QtWidgets.QMainWindow, Ui_main_window):
    # 主窗口功能

    def __init__(self):
        # 主窗口初始化
        super(MyMainWindow, self).__init__()
        self.setupUi(self)
        self.tool_button_find_excel.clicked.connect(self.find_excel)
        self.tool_button_save_train_result.clicked.connect(self.save_train_result)
        self.tool_button_save_c_fun.clicked.connect(self.save_c_fun)
        self.check_box_choose_soc.clicked.connect(self.choose_soc)
        self.check_box_choose_t.clicked.connect(self.choose_t)
        self.radio_button_choose_c.clicked.connect(self.choose_c)
        self.radio_button_choose_soh.clicked.connect(self.choose_soh)
        self.combo_box_act.currentIndexChanged.connect(self.act)
        self.push_button_correlation.clicked.connect(self.correlation)
        self.push_button_ann.clicked.connect(self.ann)
        self.push_button_line.clicked.connect(self.line)
        self.push_button_help.clicked.connect(self.help)
        self.push_button_about_train_data.clicked.connect(self.about_train_data)
        self.push_button_about_ann.clicked.connect(self.about_ann)

        # 帮助功能
        self.help_window_ui = MyHelpWindow()

        # 相关性分析功能
        self.window_correlation = CorrelationWindow()

        # 神经网络训练功能
        self.window_ann = AnnWindow()

        # 线性拟合功能
        self.window_line = LineWindow()

    def closeEvent(self, event):
        # 重写closeEvent
        reply = QtWidgets.QMessageBox.question(self, '退出', "是否要退出程序？",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def find_excel(self):
        # 选择Excel数据文件按钮功能
        self.text_browser_tips.setText("·设置您的Excel数据文件所在的位置，支持xls及xlsx格式")
        file_name, file_type = QFileDialog.getOpenFileName(self, "选取文件", "", "Excel Files(*.xls;*.xlsx)")
        self.line_edit_excel.setText(file_name)

    def save_train_result(self):
        # 选择保存训练结果及相关性分析按钮功能
        self.text_browser_tips.setText("·设置您想要保存的训练结果数据及相关性分析结果的位置，在该目录下将生成相关的Excel文件"
                                       "及拟合图像、误差统计图")
        file_name = QFileDialog.getExistingDirectory(self, "选取文件路径")
        self.line_edit_train_result.setText(file_name)

    def save_c_fun(self):
        # 选择C函数保存位置按钮功能
        self.text_browser_tips.setText("·设置您想要保存的C语言函数的位置，在该目录下将生成相关的C语言函数文件")
        file_name = QFileDialog.getExistingDirectory(self, "选取文件路径")
        self.line_edit_c_fun.setText(file_name)

    def choose_soc(self, checked):
        # 选择SOC为自变量按钮功能
        if checked == 1:
            self.text_browser_tips.setText("·已设置SOC为自变量")
            self.line_edit_soc.setEnabled(1)
        else:
            self.text_browser_tips.setText("·已取消SOC为自变量")
            self.line_edit_soc.setEnabled(0)
            self.line_edit_soc.setText("")

    def choose_t(self, checked):
        # 选择温度为自变量按钮功能
        if checked == 1:
            self.text_browser_tips.setText("·已设置温度为自变量\n·注意：线性拟合功能不支持设置温度为自变量")
            self.line_edit_t.setEnabled(1)
            self.push_button_line.setEnabled(0)
            self.spin_box_order.setEnabled(0)
        else:
            self.text_browser_tips.setText("·已取消温度为自变量")
            self.line_edit_t.setEnabled(0)
            self.line_edit_t.setText("")
            self.push_button_line.setEnabled(1)
            self.spin_box_order.setEnabled(1)

    def choose_c(self, checked):
        # 选择容量为因变量按钮功能
        if checked == 0:
            self.line_edit_soh.setEnabled(1)
            self.line_edit_c.setEnabled(0)
            self.line_edit_c.setText("")
        else:
            self.text_browser_tips.setText("·已设置容量为因变量\n·注意：仅支持容量、SOH中的一个为因变量")
            self.line_edit_c.setEnabled(1)
            self.line_edit_soh.setEnabled(0)
            self.line_edit_soh.setText("")

    def choose_soh(self, checked):
        # 选择SOH为因变量按钮功能
        if checked == 0:
            self.line_edit_c.setEnabled(1)
            self.line_edit_soh.setEnabled(0)
            self.line_edit_soh.setText("")
        else:
            self.text_browser_tips.setText("·已设置SOH为因变量\n·注意：仅支持容量、SOH中的一个为因变量")
            self.line_edit_soh.setEnabled(1)
            self.line_edit_c.setEnabled(0)
            self.line_edit_c.setText("")

    def act(self, index):
        # 阈值函数选择功能
        if index == 0:
            self.text_browser_tips.setText("·【sigmoid函数】\n·表达式\n·g(z) = 1 / (1 + exp(-z))")
        elif index == 1:
            self.text_browser_tips.setText("·【tanh函数】\n·表达式\n·g(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))")
        elif index == 2:
            self.text_browser_tips.setText("·【ReLU函数】\n·表达式\n·g(z) = z (if z>=0), 0 (if z<0)\n·【注意】这种激活函"
                                           "数容易出现神经元死亡的情况，可以通过调整参数避免这种情况")
        else:
            self.text_browser_tips.setText("·【Leaky ReLU函数】\n·表达式\n·g(z) = z (if z>=0), 0.1z (if z<0)\n·【注意】"
                                           "这种激活函数容易出现神经元死亡的情况，可以通过调整参数避免这种情况")

    def help(self):
        # 帮助窗口
        self.help_window_ui.show()

    def about_train_data(self):
        # 训练数据帮助提示
        self.text_browser_tips.setText("·【训练数据设置】\n·[基本要求]：输入整数。\n·[特殊规则]：Excel的第一列（A列）记为0，"
                                       "第二列（B列）记为1，以此类推。\n·[输入举例]：\n·Excel：电阻在A列，SOC在C列，容量在E"
                                       "列\n·输入：电阻所在列数：0  SOC所在列数：2  容量所在列数：4")

    def about_ann(self):
        # 人工神经网络帮助提示
        self.text_browser_tips.setText("·【神经网络每个隐藏层神经元数量】\n·[基本要求]：输入一个或多个整数，并用空格隔开。\n"
                                       "·[特殊规则]：每个整数代表每个隐藏层的神经元个数，整数的数量代表隐藏层神经元层数。\n·"
                                       "[输入举例]：\n·需求：五个隐藏层，分别设置5、4、3、2、1个神经元  \n·输入：神经网络每"
                                       "个隐藏层神经元数量：5 4 3 2 1\n\n·【学习率】\n·[基本概念]：每次训练过程中进行反向传"
                                       "递时的一个参数，用于修正weights和biases。\n·[输入规则]：一般在0~1之间取值，设置的过"
                                       "小会使得训练缓慢，过大可能会得不到最优解。")

    def data_write(self, class_instance):
        # 公共数据传入
        class_instance.data_excel = self.line_edit_excel.text()
        class_instance.data_train_result = self.line_edit_train_result.text()
        class_instance.data_r = self.line_edit_r.text()
        class_instance.data_soc = self.line_edit_soc.text()
        class_instance.data_c = self.line_edit_c.text()
        class_instance.data_soh = self.line_edit_soh.text()

    def correlation(self):
        # 相关性分析按钮功能
        self.text_browser_tips.setText("·启动相关性分析功能")
        self.window_correlation.__init__()

        # 相关性分析数据传入
        self.data_write(self.window_correlation)
        self.window_correlation.data_t = self.line_edit_t.text()

        # 通过线程运行相关性分析程序
        if threading.activeCount() != 1:
            stop_thread(self.thread)
        self.thread = MyThread(1, "Correlation", self.window_correlation)
        self.thread.start()

    def ann(self):
        # 人工神经网络训练按钮功能
        self.text_browser_tips.setText("·启动人工神经网训练功能")
        self.window_ann.__init__()

        # 人工神经网络训练数据传入
        self.data_write(self.window_ann)
        self.window_ann.data_c_save = self.line_edit_c_fun.text()
        self.window_ann.data_t = self.line_edit_t.text()
        self.window_ann.data_neuron = self.line_edit_neuron.text()
        self.window_ann.data_epochs = self.line_edit_epochs.text()
        self.window_ann.data_mini_batch_size = self.line_edit_mini_batch_size.text()
        self.window_ann.data_eta = self.line_edit_eta.text()
        self.window_ann.data_act = self.combo_box_act.currentIndex()

        # 通过线程运行人工神经网络训练程序
        if threading.activeCount() != 1:
            stop_thread(self.thread)
        self.thread = MyThread(2, "ANN", self.window_ann)
        self.thread.start()

    def line(self):
        # 线性拟合按钮功能
        self.text_browser_tips.setText("·启动线性拟合功能")
        self.window_line.__init__()

        # 线性拟合数据传入
        self.data_write(self.window_line)
        self.window_line.data_c_save = self.line_edit_c_fun.text()
        self.window_line.data_order = self.spin_box_order.value()

        # 通过线程运行人工神经网络训练程序

        if threading.activeCount() != 1:
            stop_thread(self.thread)
        self.thread = MyThread(3, "LINE", self.window_line)
        self.thread.start()

