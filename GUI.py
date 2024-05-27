import cv2
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QGraphicsScene, QGraphicsView, QFrame
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt, QRect
from PIL import Image, ImageTk
import numpy as np
from math import atan2, cos, sin, sqrt, pi
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
import mask
import serial
from time import sleep
import serial



class ImageProcessingApp(QMainWindow):

    ser = serial.Serial("COM9", 9600)

    def Reset(self,ser):
        self.ser.write(str(1).encode())


    def __init__(self):
        super(ImageProcessingApp, self).__init__()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.setObjectName("ImageProcessingApp")
        self.resize(1599, 920)
        self.setStyleSheet("background-color: rgb(255, 250, 221)")


        self.label = QLabel(self)
        self.label.setGeometry(350, 40, 111, 171)
        self.label.setPixmap(QtGui.QPixmap("logo_BK.png"))
        self.label.setScaledContents(True)

        self.label_6 = QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(1250, 20, 111, 171))
        self.label_6.setPixmap(QtGui.QPixmap("logo_SME.png"))
        self.label_6.setScaledContents(True)


        self.label_2 = QLabel("<html><head/><body><p align=\"center\"><span style=\" font-size:22pt; font-weight:600;\">ĐẠI HỌC BÁCH KHOA HÀ NỘI</span></p><p align=\"center\"><span style=\" font-size:22pt; font-weight:600;\">TRƯỜNG CƠ KHÍ - KHOA CƠ ĐIỆN TỬ</span></p><p align=\"center\"><span style=\" font-size:20pt; font-weight:600;\">BÀI TẬP LỚN XỬ LÝ ẢNH</span></p></body></html>", self)
        self.label_2.setWordWrap(True)
        self.label_2.setGeometry(470, 10, 751, 201)
        self.label_2.setAlignment(Qt.AlignCenter)

        self.label_3 = QLabel(self)
        self.label_3.setGeometry(110, 230, 900, 671)
        self.label_3.setAlignment(Qt.AlignCenter)
        self.label_3.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_3.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.label_3.setLineWidth(3)

        self.label_4 = QLabel(self)
        self.label_4.setGeometry(1130, 230, 431, 221)
        self.label_4.setAlignment(Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_4.setFont(font)
        self.label_4.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.label_4.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.label_4.setLineWidth(3)

        self.label_5 = QLabel("<b>CAMERA<b>", self)
        self.label_5.setWordWrap(True)
        font = QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setGeometry(510, 210, 121, 41)
        self.label_5.setAlignment(Qt.AlignCenter)

        self.label_7 = QLabel("<b>THÔNG TIN SẢN PHẨM<b>",self)
        self.label_7.setWordWrap(True)
        font = QFont()
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setGeometry(1210, 210, 281, 41)
        self.label_7.setAlignment(Qt.AlignCenter)



        # self.scene = QGraphicsScene(self)
        # # self.scene.setSceneRect(10, 250, 691, 141)
        # self.video_view = QGraphicsView(self.scene)
        # self.layout = QVBoxLayout(self.central_widget)
        #
        # self.layout.addWidget(self.video_view)


        self.btn_process = QPushButton("Process Image", self)
        font = QFont()
        font.setPointSize(12)
        self.btn_process.setFont(font)
        self.btn_process.setGeometry(1130, 480, 431, 50)
        self.btn_process.setStyleSheet("background-color: green;")

        self.btn_stop = QPushButton("Stop Processing", self)
        font = QFont()
        font.setPointSize(12)
        self.btn_stop.setFont(font)
        self.btn_stop.setGeometry(1130, 550, 431, 50)
        self.btn_stop.setStyleSheet("background-color: red;")

        self.btn_back = QPushButton("Go Back", self)
        font = QFont()
        font.setPointSize(12)
        self.btn_back.setFont(font)
        self.btn_back.setGeometry(1130, 620, 431, 50)
        self.btn_back.setStyleSheet("background-color: cyan;")

        self.btn_measure = QPushButton("Measure Object", self)
        font = QFont()
        font.setPointSize(12)
        self.btn_measure.setFont(font)
        self.btn_measure.setGeometry(1130, 690, 431, 50)
        self.btn_measure.setStyleSheet("background-color: cyan;")

        self.btn_send = QPushButton("Send The Angel", self)
        font = QFont()
        font.setPointSize(12)
        self.btn_send.setFont(font)
        self.btn_send.setGeometry(1130, 760, 431, 50)
        self.btn_send.setStyleSheet("background-color: green;")

        self.btn_reset = QPushButton("Reset", self)
        font = QFont()
        font.setPointSize(12)
        self.btn_reset.setFont(font)
        self.btn_reset.setGeometry(1130, 830, 431, 50)
        self.btn_reset.setStyleSheet("background-color: green;")

        self.btn_process.clicked.connect(self.start_processing)
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_back.clicked.connect(self.go_back)
        self.btn_measure.clicked.connect(self.switch_to_object_measure)
        self.btn_send.clicked.connect(self.Rotate)
        self.btn_reset.clicked.connect(self.Reset)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.processing = False
        self.object_measure_mode = False
        self.initial_frame = None
        self.processed_frame = None

    def start_processing(self):
        self.processing = True
        self.btn_process.setEnabled(False)
        self.btn_measure.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_back.setEnabled(True)

    def stop_processing(self):
        self.processing = False
        self.object_measure_mode = False
        self.btn_process.setEnabled(True)
        self.btn_measure.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_back.setEnabled(False)

    def go_back(self):
        self.processing = False
        self.object_measure_mode = False
        self.btn_process.setEnabled(True)
        self.btn_measure.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_back.setEnabled(False)
        self.display_frame(self.initial_frame)

    def switch_to_object_measure(self):
        self.processing = True
        self.object_measure_mode = True
        self.btn_process.setEnabled(False)
        self.btn_measure.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_back.setEnabled(True)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            if self.processing and not self.object_measure_mode:
                self.processed_frame = self.process_frame(frame)
                self.display_frame(self.processed_frame)
            elif self.object_measure_mode:
                measured_object = self.measure_object(frame)
                self.display_frame(measured_object)
            else:
                self.display_frame(frame)
    def process_frame(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert frame to binary
        _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find all the contours in the thresholded frame
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Process each contour
        for i, c in enumerate(contours):
            # Calculate the area of each contour
            area = cv2.contourArea(c)

            # Ignore contours that are too small or too large
            if area < 1000 or 120000 < area:
                continue

            # Draw each contour only for visualization purposes
            cv2.drawContours(frame, contours, i, (0, 0, 255), 2)

            # Find the orientation of each shape
            getOrientation(c, frame)
            self.label_4.setText(str(round(90 - (-np.rad2deg(getOrientation(c, frame))),0)) + '' + 'degree')

        # Show the processed frame

        return frame

    def Rotate(self,ser):

        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)

            if area < 2000 or 270000 < area:
                continue
            sz = len(c)
            data_c = np.empty((sz, 2), dtype=np.float64)
            for i in range(data_c.shape[0]):
                data_c[i, 0] = c[i, 0, 0]
                data_c[i, 1] = c[i, 0, 1]
            mean = np.empty((0))
            mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_c, mean)
            prevangle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])
            angle = int(np.rad2deg(prevangle))
            rot = 90 - angle
            print(rot)
            self.ser.write(str(94-rot).encode())

    def measure_object(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3,0), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

        result_img = closing.copy()
        contours, hierachy = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pixelsPerMetric = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000 or area > 120000:
                continue
            box = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(frame, [box.astype("int")], -1, (0, 255, 64), 2)

            for (x, y) in box:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 64), -1)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)


            cv2.circle(frame, (int(tltrX), int(tltrY)), 0, (0, 255, 64), 5)
            cv2.circle(frame, (int(blbrX), int(blbrY)), 0, (0, 255, 64), 5)
            cv2.circle(frame, (int(tlblX), int(tlblY)), 0, (0, 255, 64), 5)
            cv2.circle(frame, (int(trbrX), int(trbrY)), 0, (0, 255, 64), 5)


            cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 2)
            cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 2)


            width_pixel = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            length_pixel = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))


            if pixelsPerMetric is None:
                pixelsPerMetric = width_pixel
                pixelsPerMetric = length_pixel
            width = width_pixel
            length = length_pixel


            cv2.putText(frame, "L: {:.1f}CM".format(width_pixel / 25.5/3.9), (int(trbrX + 10), int(trbrY)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "P: {:.1f}CM".format(length_pixel / 25.5/3.3), (int(tltrX - 15), int(tltrY - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            self.label_4.setText(str(round(((width_pixel / 25.5))/3.9,1)) + 'x' + str(round((length_pixel / 25.5)/3.3,1)) + '(cm)' )

        return frame

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.label_3.clear()
        pixmap = pixmap.scaled(self.label_3.size(), aspectRatioMode=QtCore.Qt.KeepAspectRatio)
        self.label_3.setPixmap(pixmap)

    def run(self):
        self.show()
        self.timer.start(15)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    cntr = (int(mean[0, 0]), int(mean[0, 1]))

    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0, 0] * eigenvalues[0, 0], cntr[1] + 0.02 * eigenvectors[0, 1] * eigenvalues[0, 0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1, 0] * eigenvalues[1, 0], cntr[1] - 0.02 * eigenvectors[1, 1] * eigenvalues[1, 0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    label = "  Rotation Angle: " + str(-(-int(np.rad2deg(angle)) - 90)) + " degrees"
    textbox = cv2.rectangle(img, (cntr[0], cntr[1] - 25), (cntr[0] + 250, cntr[1] + 10), (255, 255, 255), -1)
    cv2.putText(img, label, (cntr[0], cntr[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


    return angle


def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)

    angle = atan2(p[1] - q[1], p[0] - q[0])
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))

    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv2.LINE_AA)

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def main():
    app = QApplication(sys.argv)
    image_processing_app = ImageProcessingApp()
    image_processing_app.run()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
