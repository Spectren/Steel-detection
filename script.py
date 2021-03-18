from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
import PyQt5

from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog

from PIL import Image

import os
import sys


class NeuralNetProcessor(QObject):
    start = pyqtSignal(np.ndarray)
    predicted = pyqtSignal(np.ndarray)

    def process_image(self, data):
        prediction = model.predict(data)
        
        self.predicted.emit(prediction)



class MainUI(QMainWindow):
    def __init__(self):
        super(MainUI, self).__init__()
        
        uic.loadUi('ui.ui', self)
        
        self.processor = None

        self.model_thread = QThread()
        self.model_thread.start()

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        
        self.target_image, *self.subplots, self.result_image = self.figure.subplots(6, 1)
        
        self.target_image.set_title('Картина образца стали')
        self.target_image.axis('off')

        self.result_image.set_title('Итоговое предсказание')
        self.result_image.axis('off')

        self.target_data = self.target_image.imshow(np.zeros((128, 800)), vmin=0.0, vmax=1.0)
        self.result_data = self.result_image.imshow(np.zeros((128, 800)), vmin=0.0, vmax=1.0)
        self.defect_data = []
        
        for i, subplot in enumerate(self.subplots, start=1):
            subplot.set_title(f'Дефект #{i}')
            subplot.axis('off')
            self.defect_data.append(subplot.imshow(np.zeros((128, 800)), vmin=0.0, vmax=1.0))
        
        self.figure.tight_layout()
        self.target_layout.addWidget(self.canvas)
        
        self.open_btn.clicked.connect(self.get_image)
    
    def get_image(self):
        file, _ = QFileDialog.getOpenFileName(self, 'Выберите изображение', '.', "Изображения (*.jpg, *.png)")
        
        if not file:
            return

        try:
            img = Image.open(file)
        except Exception:
            error_dlg = QMessageBox()
            
            error_dlg.setIcon(QMessageBox.Critical)
            error_dlg.setText('Ошибка открытия изображения!')
            error_dlg.setStandardButtons(QMessageBox.Ok)
            
            error_dlg.exec_()
            return

        try:
            data = np.array(img.resize((800, 128))).reshape((1, 128, 800, 3))
        except Exception:
            error_dlg = QMessageBox()
            
            error_dlg.setIcon(QMessageBox.Critical)
            error_dlg.setText('Ошибка преобразования изображения!')
            error_dlg.setStandardButtons(QMessageBox.Ok)
            
            error_dlg.exec_()
            return

        self.target_data.set_data((data / 255).reshape((128, 800, 3)))
        self.canvas.draw()
        
        self.process_image(data)

    def process_image(self, data):
        self.processor = NeuralNetProcessor()
        
        self.processor.moveToThread(self.model_thread)

        self.processor.start.connect(self.processor.process_image)
        self.processor.start.emit(data)
        self.processor.predicted.connect(self.process_prediction)

    @pyqtSlot(np.ndarray)
    def process_prediction(self, prediction):
        for pred, datum in zip(prediction.swapaxes(0, 3), self.defect_data):
            datum.set_data(pred.reshape((128, 800)))
        self.canvas.draw()



app = QApplication(sys.argv)
window = MainUI()
window.show()

app.exec_()




