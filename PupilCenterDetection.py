import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QTextEdit, QComboBox, QMainWindow
from PyQt5.QtGui import QPixmap
import cv2 as cv
import os
import tempfile
from pupil import Pupil

class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.__title = "Pupil Center Detection"
        self.__right = 200
        self.__down = 100
        self.__width = 800
        self.__height = 300
        self.dir = tempfile.TemporaryDirectory()
        self.img_path = ""
        self.temp_dir = os.path.join(self.dir.name, 'browsed.png')
        self.method = "parallelogram"
        print(self.temp_dir)


        self.Init()

    def Init(self):
        self.setWindowTitle(self.__title)
        self.setGeometry(self.__right, self.__down, self.__width, self.__height)

        btn_browse = QPushButton("Browse Image", self)
        btn_browse.setGeometry(100, 10, 100, 22)
        btn_browse.clicked.connect(self.getImage)

        btn_process = QPushButton("Find center", self)
        btn_process.setGeometry(600, 10, 100, 22)
        btn_process.clicked.connect(self.processImage)

        self.img_browsed = QLabel(self)
        self.img_browsed.setText("Browsed image")
        self.img_browsed.setGeometry(50, 50, 200, 200)

        self.img_processed = QLabel(self)
        self.img_processed.setText("Processed image")
        self.img_processed.setGeometry(550, 50, 200, 200)

        self.comboBox = QComboBox(self)
        self.comboBox.addItem("Parallelogram Method")
        self.comboBox.move(340, 10)
        self.comboBox.currentIndexChanged.connect(self.comboChange)

        self.textbox = QTextEdit(self)
        self.textbox.setGeometry(300, 50, 200, 200)
        self.textbox.setAlignment(Qt.AlignLeft)

        self.show()

    def getImage(self):
        file_name = QFileDialog.getOpenFileName(self, 'Open file',
                                            'c:\\', "Image files (*.jpg *.gif *.bmp *.tif *.png)")
        self.img_path = file_name[0]
        img = cv.imread(self.img_path, 0)
        img = cv.resize(img, dsize=(200, 200), interpolation=cv.INTER_AREA)
        cv.imwrite(self.temp_dir, img)
        pix_map = QPixmap(self.temp_dir)
        self.img_browsed.setPixmap(QPixmap(pix_map))

    def comboChange(self):
        if self.comboBox.currentText() == "Parallelogram Method":
            self.method = "parallelogram"

    def processImage(self):
        x, y, a, b, fangle = 0, 0, 0, 0, 0

        if self.img_path != "":
            processed_path = os.path.join(self.dir.name, 'processed.png')
            print(processed_path)
            pupil = Pupil(self.img_path, processed_path)
            if self.method == "parallelogram":
                x, y, a, b, fangle = pupil.parallelogram()
                text = f"Pupil center:\nX = {x}\nY = {y}\n\nPupil elipse:\na = {a}\nb = {b}\nfangle = {fangle}"
                self.textbox.setText(text)
            img = cv.imread(processed_path, 1)
            # cv.imshow("Pieknie", img)
            img = cv.resize(img, dsize=(200, 200), interpolation=cv.INTER_AREA)
            resized_path = os.path.join(self.dir.name, 'resized.png')
            cv.imwrite(resized_path, img)
            pix_map = QPixmap(resized_path)
            self.img_processed.setPixmap(QPixmap(pix_map))


App = QApplication(sys.argv)
window = Window()
sys.exit(App.exec())

