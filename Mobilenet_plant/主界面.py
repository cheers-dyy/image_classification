from PyQt5 import QtCore, QtGui, QtWidgets
from main import Ui_MainWindow
from login import Ui_MainWindow0
from register import Ui_MainWindow1
from show import Ui_MainWindow2
from user import Ui_MainWindow3
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class userWindow(QtWidgets.QMainWindow,Ui_MainWindow3):
    def __init__(self):
        super(userWindow,self).__init__()
        self.setupUi(self)


class showWindow(QtWidgets.QMainWindow,Ui_MainWindow2):
    def __init__(self):
        super(showWindow,self).__init__()
        self.setupUi(self)



#主界面
class mainWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(mainWindow,self).__init__()
        self.setupUi(self)
        self.user.clicked.connect(self.use)
        self.his.clicked.connect(self.sh)

    def sh(self):
        self.log = showWindow()
        self.log.show()


    def use(self):
        self.log = userWindow()
        self.log.show()






#登录界面
class zuichu(QtWidgets.QMainWindow,Ui_MainWindow0):
    def __init__(self):
        super(zuichu,self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.go)
        self.pushButton_6.clicked.connect(self.zhuce)
        # self.pushButton_5.clicked.connect(self.read)
        # self.pushButton_3.clicked.connect(self.history)
        # self.pushButton_4.clicked.connect(self.paper)
        self.persons={}
        self.names=[]

    def go(self):
        import csv
        with open('information.csv', encoding='utf-8')as fp:
            reader = csv.DictReader(fp)
            for i in reader:
                #print(i['name'],i['password'])
                self.names.append(i['name'])
                self.persons[i['name']]=i['password']
        if  self.lineEdit.text() not in self.names:
            eply = QtWidgets.QMessageBox.information(None, '提示', '用户不存在!', buttons=QtWidgets.QMessageBox.Ok)
            self.lineEdit.setText('')
            self.lineEdit_2.setText('')
        else:
            if self.lineEdit_2.text()!=self.persons[self.lineEdit.text()]:
                eply = QtWidgets.QMessageBox.information(None, '提示', '密码错误!', buttons=QtWidgets.QMessageBox.Ok)
                self.lineEdit_2.setText('')
            else:
                with open('name.txt','w')as fb:
                    fb.write(self.lineEdit.text())
                self.hide()           #隐藏此窗口
                self.log = mainWindow()
                self.log.show()

    def zhuce(self):
        self.hide()  # 隐藏此窗口
        self.log = zhuce1()
        self.log.show()


class zhuce1(QtWidgets.QMainWindow, Ui_MainWindow1):
    def __init__(self):
        super(zhuce1, self).__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.hui)
    def hui(self):
        self.hide()  # 隐藏此窗口
        self.log = zuichu()
        self.log.show()

#运行窗口Login
if __name__=="__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    login_show = zuichu()
    login_show.show()
    sys.exit(app.exec_())