import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from Q5ui import Ui_MainWindow
from PyQt5.uic import loadUi

def go():
    w.label.setText( "答案：" + str(float(w.lineEdit.text()) + float(w.lineEdit_2.text())))

app = QApplication(sys.argv)
w = loadUi('Q5.ui')
# w.button5_1.clicked.connect(go)
w.show()
sys.exit(app.exec_())