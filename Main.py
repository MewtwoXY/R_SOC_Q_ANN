import sys
import os
from MainWindowFun import MyMainWindow
if hasattr(sys, 'frozen'):  # PyQt5打包有bug需要自己设定PATH
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5 import QtWidgets


if __name__ == '__main__':
    # 开始运行窗口
    app = QtWidgets.QApplication(sys.argv)
    main_window_ui = MyMainWindow()
    main_window_ui.show()
    os.system("cls")
    print("\n·【注意：不要关闭此控制台窗口，否则会使整个平台关闭】\n")
    sys.exit(app.exec_())
