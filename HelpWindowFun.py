from HelpWindow import Ui_help_window
import sys
import os
if hasattr(sys, 'frozen'):  # PyQt5打包有bug需要自己设定PATH
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
from PyQt5 import QtWidgets


class MyHelpWindow(QtWidgets.QWidget, Ui_help_window):

    def __init__(self):
        super(MyHelpWindow, self).__init__()
        self.setupUi(self)
