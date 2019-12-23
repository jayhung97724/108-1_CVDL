# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\hw2.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(796, 578)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(60, 80, 311, 201))
        self.groupBox.setObjectName("groupBox")
        self.pushButton_Q1 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_Q1.setGeometry(QtCore.QRect(90, 90, 121, 31))
        self.pushButton_Q1.setObjectName("pushButton_Q1")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(420, 80, 311, 441))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_Q3_1 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_Q3_1.setGeometry(QtCore.QRect(30, 60, 241, 31))
        self.pushButton_Q3_1.setObjectName("pushButton_Q3_1")
        self.pushButton_Q3_2 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_Q3_2.setGeometry(QtCore.QRect(30, 110, 241, 31))
        self.pushButton_Q3_2.setObjectName("pushButton_Q3_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(60, 320, 311, 201))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_Q2 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_Q2.setGeometry(QtCore.QRect(90, 100, 121, 31))
        self.pushButton_Q2.setObjectName("pushButton_Q2")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Stereo"))
        self.pushButton_Q1.setText(_translate("MainWindow", "1.1 Disparity"))
        self.groupBox_2.setTitle(_translate("MainWindow", "3. SIFT"))
        self.pushButton_Q3_1.setText(_translate("MainWindow", "3.1 Keypoints"))
        self.pushButton_Q3_2.setText(_translate("MainWindow", "3.2 Matched keypoints"))
        self.groupBox_3.setTitle(_translate("MainWindow", "2. Normalized Cross Correlation"))
        self.pushButton_Q2.setText(_translate("MainWindow", "2.1 NCC"))

