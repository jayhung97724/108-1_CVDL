# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Q5.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(520, 439)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(100, 80, 321, 291))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.button5_1 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.button5_1.setObjectName("button5_1")
        self.verticalLayout.addWidget(self.button5_1)
        self.button5_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.button5_2.setObjectName("button5_2")
        self.verticalLayout.addWidget(self.button5_2)
        self.button5_3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.button5_3.setObjectName("button5_3")
        self.verticalLayout.addWidget(self.button5_3)
        self.button5_4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.button5_4.setObjectName("button5_4")
        self.verticalLayout.addWidget(self.button5_4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.lineEdit.setCursorPosition(0)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.button5_5 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.button5_5.setObjectName("button5_5")
        self.verticalLayout.addWidget(self.button5_5)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(100, 50, 321, 16))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.button5_1.setText(_translate("MainWindow", "5.1 Show Train Images"))
        self.button5_2.setText(_translate("MainWindow", "5.2 Show Hyperparameters"))
        self.button5_3.setText(_translate("MainWindow", "5.3 Train 1 Epoch"))
        self.button5_4.setText(_translate("MainWindow", "5.4 Show Training Result"))
        self.label.setText(_translate("MainWindow", "Test Image Index:"))
        self.lineEdit.setPlaceholderText(_translate("MainWindow", "(0~99999"))
        self.button5_5.setText(_translate("MainWindow", "5.5 Inference"))
        self.label_2.setText(_translate("MainWindow", "5. Train Cifar-10 Classifier Using LeNet-5 "))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

