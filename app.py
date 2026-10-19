import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QMessageBox, QComboBox,
    QSizePolicy
)
from PyQt5.QtGui import QImage, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint

from predict import predict

class Canvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(400, 400)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("background-color: white; border: 2px solid #444;")
        self.image = QImage(400, 400, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.last_point = QPoint()
        self.drawing = False
        self.brush_size = 10
        self.brush_color = QColor("black")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing and (event.buttons() & Qt.LeftButton):
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size,
                                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            painter.end()
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def get_numpy_gray(self):
        img = self.image.convertToFormat(QImage.Format_Grayscale8)
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.width())
        return arr.copy()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Sketch — Reconnaissance de dessins")
        self.setGeometry(100, 100, 750, 500)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)

        # ----- Canvas (gauche) -----
        canvas_layout = QVBoxLayout()
        self.canvas = Canvas()
        canvas_layout.addWidget(self.canvas)
        canvas_layout.addStretch()
        main_layout.addLayout(canvas_layout, 2)

        # ----- Panneau de contrôle (droite) -----
        right = QVBoxLayout()

        right.addWidget(QLabel("Taille du pinceau :"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(40)
        self.slider.setValue(10)
        self.slider.valueChanged.connect(self.change_brush)
        right.addWidget(self.slider)

        right.addWidget(QLabel("Couleur :"))
        self.color_box = QComboBox()
        self.color_box.addItems(["Noir", "Rouge", "Bleu", "Vert"])
        self.color_box.currentTextChanged.connect(self.change_color)
        right.addWidget(self.color_box)

        self.btn_clear = QPushButton("Effacer")
        self.btn_clear.clicked.connect(self.canvas.clear)
        right.addWidget(self.btn_clear)

        self.btn_save = QPushButton("Enregistrer")
        self.btn_save.clicked.connect(self.save_image)
        right.addWidget(self.btn_save)

        self.btn_predict = QPushButton("Reconnaître")
        self.btn_predict.clicked.connect(self.do_predict)
        right.addWidget(self.btn_predict)

        self.result_label = QLabel("Prédiction : —")
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        right.addWidget(self.result_label)

        self.score_label = QLabel("Confiance : —")
        right.addWidget(self.score_label)

        right.addStretch()

        main_layout.addLayout(right, 1)

    def change_brush(self, value):
        self.canvas.brush_size = value

    def change_color(self, name):
        colors = {"Noir": "black", "Rouge": "red",
                  "Bleu": "blue", "Vert": "green"}
        self.canvas.brush_color = QColor(colors[name])

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le dessin", "dessin.png",
            "Images PNG (*.png);;JPEG (*.jpg)")
        if path:
            self.canvas.image.save(path)
            QMessageBox.information(self, "OK", f"Dessin enregistré :\n{path}")

    def do_predict(self):
        try:
            gray = self.canvas.get_numpy_gray()
            cls, conf = predict(gray)
            self.result_label.setText(f"Prédiction : {cls}")
            self.score_label.setText(f"Confiance : {conf*100:.2f} %")
        except FileNotFoundError:
            QMessageBox.warning(
                self, "Modèle manquant",
                "Le modèle n'est pas entraîné.\n"
                "Lance d'abord : python train_model.py")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
