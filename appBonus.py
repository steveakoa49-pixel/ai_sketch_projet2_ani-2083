import sys
import os
import datetime
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QMessageBox, QComboBox,
    QSizePolicy, QGroupBox, QListWidget, QListWidgetItem
)
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QPixmap, QIcon
from PyQt5.QtCore import Qt, QPoint, QSize

from predict import predict


# ============================================================
# BONUS 1 : Fonctions de filtres artistiques (OpenCV)
# ============================================================
def apply_pencil_sketch(img_gray):
    """Filtre crayon (pencil sketch) avec OpenCV."""
    inv = 255 - img_gray
    blurred = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(img_gray, 255 - blurred, scale=256)
    return sketch

def apply_cartoon(img_bgr):
    """Filtre cartoon avec OpenCV."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img_bgr, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def apply_pastel(img_bgr):
    """Filtre pastel avec OpenCV."""
    return cv2.stylization(img_bgr, sigma_s=60, sigma_r=0.45)


# ============================================================
# Canvas de dessin (inchangé, mais couleur par défaut modifiée)
# ============================================================
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

    def get_numpy_bgr(self):
        """Retourne l'image en BGR (pour les filtres OpenCV)."""
        img = self.image.convertToFormat(QImage.Format_RGB888)
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape(img.height(), img.width(), 3)
        return cv2.cvtColor(arr.copy(), cv2.COLOR_RGB2BGR)


# ============================================================
# Fenêtre principale avec tous les bonus
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Sketch — Reconnaissance de dessins (Bonus créativité)")
        self.setGeometry(100, 100, 1050, 580)

        self.gallery_dir = "galerie"
        os.makedirs(self.gallery_dir, exist_ok=True)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ===== Zone gauche : canvas + filtre =====
        left = QVBoxLayout()
        self.canvas = Canvas()
        left.addWidget(self.canvas)

        # BONUS 4 : second objet après reconnaissance
        self.second_obj_btn = QPushButton("➕ Ajouter un second objet")
        self.second_obj_btn.clicked.connect(self.prepare_second_object)
        left.addWidget(self.second_obj_btn)

        # BONUS 1 : filtres artistiques
        filter_group = QGroupBox("🎨 Filtres artistiques")
        filter_layout = QHBoxLayout()
        self.btn_pencil = QPushButton("Crayon")
        self.btn_pencil.clicked.connect(lambda: self.apply_filter("pencil"))
        self.btn_cartoon = QPushButton("Cartoon")
        self.btn_cartoon.clicked.connect(lambda: self.apply_filter("cartoon"))
        self.btn_pastel = QPushButton("Pastel")
        self.btn_pastel.clicked.connect(lambda: self.apply_filter("pastel"))
        self.btn_reset_filter = QPushButton("↺")
        self.btn_reset_filter.setToolTip("Restaurer le dessin original")
        self.btn_reset_filter.clicked.connect(self.reset_filter)
        filter_layout.addWidget(self.btn_pencil)
        filter_layout.addWidget(self.btn_cartoon)
        filter_layout.addWidget(self.btn_pastel)
        filter_layout.addWidget(self.btn_reset_filter)
        filter_group.setLayout(filter_layout)
        left.addWidget(filter_group)

        main_layout.addLayout(left, 0)

        # ===== Zone centrale : contrôles =====
        center = QVBoxLayout()

        center.addWidget(QLabel("Taille du pinceau :"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(40)
        self.slider.setValue(15)
        self.slider.valueChanged.connect(self.change_brush)
        center.addWidget(self.slider)

        center.addWidget(QLabel("Couleur :"))
        self.color_box = QComboBox()
        self.color_box.addItems([
            "Noir", "Rouge", "Bleu", "Vert",
            "Orange", "Violet", "Rose", "Marron"
        ])
        self.color_box.currentTextChanged.connect(self.change_color)
        center.addWidget(self.color_box)

        # BONUS 4 : palette étendue = déjà via color_box ci-dessus

        self.btn_clear = QPushButton("Effacer")
        self.btn_clear.clicked.connect(self.canvas.clear)
        center.addWidget(self.btn_clear)

        self.btn_save = QPushButton("Enregistrer")
        self.btn_save.clicked.connect(self.save_image)
        center.addWidget(self.btn_save)

        # BONUS 2 : sauvegarde automatique dans la galerie
        self.btn_save_gallery = QPushButton("💾 Ajouter à la galerie")
        self.btn_save_gallery.clicked.connect(self.save_to_gallery)
        center.addWidget(self.btn_save_gallery)

        self.btn_predict = QPushButton("Reconnaître")
        self.btn_predict.clicked.connect(self.do_predict)
        center.addWidget(self.btn_predict)

        self.result_label = QLabel("Prédiction : —")
        self.result_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        center.addWidget(self.result_label)

        self.score_label = QLabel("Confiance : —")
        center.addWidget(self.score_label)

        # BONUS 3 : historique des prédictions
        center.addWidget(QLabel("📜 Historique :"))
        self.history = QListWidget()
        self.history.setMaximumHeight(120)
        center.addWidget(self.history)

        center.addStretch()
        main_layout.addLayout(center, 1)

        # ===== Zone droite : galerie visuelle =====
        right = QVBoxLayout()
        right.addWidget(QLabel("🖼️ Galerie des dessins :"))
        self.gallery_list = QListWidget()
        self.gallery_list.setIconSize(QSize(80, 80))
        self.gallery_list.setMaximumWidth(180)
        self.gallery_list.itemDoubleClicked.connect(self.open_gallery_item)
        right.addWidget(self.gallery_list)

        self.btn_refresh_gallery = QPushButton("🔄 Rafraîchir")
        self.btn_refresh_gallery.clicked.connect(self.refresh_gallery)
        right.addWidget(self.btn_refresh_gallery)

        main_layout.addLayout(right, 0)

        # Charger la galerie au démarrage
        self.refresh_gallery()

        # Mémoriser l'image originale pour le reset de filtre
        self.original_image = None

    # ---------- Contrôles ----------
    def change_brush(self, value):
        self.canvas.brush_size = value

    def change_color(self, name):
        colors = {
            "Noir": "black", "Rouge": "red", "Bleu": "blue", "Vert": "green",
            "Orange": "orange", "Violet": "purple", "Rose": "pink", "Marron": "brown"
        }
        self.canvas.brush_color = QColor(colors[name])

    # ---------- Effacer / Enregistrer ----------
    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le dessin", "dessin.png",
            "Images PNG (*.png);;JPEG (*.jpg)")
        if path:
            self.canvas.image.save(path)
            QMessageBox.information(self, "OK", f"Dessin enregistré :\n{path}")

    # ---------- BONUS 2 : Galerie ----------
    def save_to_gallery(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dessin_{timestamp}.png"
        path = os.path.join(self.gallery_dir, filename)
        self.canvas.image.save(path)
        self.refresh_gallery()
        QMessageBox.information(self, "Galerie", f"Ajouté à la galerie :\n{filename}")

    def refresh_gallery(self):
        self.gallery_list.clear()
        if not os.path.exists(self.gallery_dir):
            return
        files = sorted(os.listdir(self.gallery_dir), reverse=True)
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(self.gallery_dir, fname)
            pix = QPixmap(path).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QListWidgetItem(fname)
            item.setIcon(QIcon(pix))          # ✅ QIcon au lieu de QPixmap
            item.setData(Qt.UserRole, path)
            self.gallery_list.addItem(item)
        self.gallery_list.clear()
        if not os.path.exists(self.gallery_dir):
            return
        files = sorted(os.listdir(self.gallery_dir), reverse=True)
        for fname in files:
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(self.gallery_dir, fname)
            pix = QPixmap(path).scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item = QListWidgetItem(fname)
            item.setIcon(QIcon(pix))          # ✅ QIcon au lieu de QPixmap
            item.setData(Qt.UserRole, path)
            self.gallery_list.addItem(item)

    def open_gallery_item(self, item):
        path = item.data(Qt.UserRole)
        if path and os.path.exists(path):
            img = QImage(path)
            if not img.isNull():
                self.canvas.image = img.scaled(400, 400, Qt.KeepAspectRatio,
                                               Qt.SmoothTransformation)
                self.canvas.image = self.canvas.image.convertToFormat(QImage.Format_RGB32)
                self.canvas.update()

    # ---------- BONUS 1 : Filtres ----------
    def apply_filter(self, kind):
        if self.original_image is None:
            self.original_image = self.canvas.image.copy()

        bgr = self.canvas.get_numpy_bgr()

        if kind == "pencil":
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            out = apply_pencil_sketch(gray)
        elif kind == "cartoon":
            out = apply_cartoon(bgr)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        elif kind == "pastel":
            out = apply_pastel(bgr)
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        else:
            return

        if len(out.shape) == 2:
            h, w = out.shape
            qimg = QImage(out.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, _ = out.shape
            qimg = QImage(out.data, w, h, 3 * w, QImage.Format_RGB888)

        self.canvas.image = qimg.copy().convertToFormat(QImage.Format_RGB32)
        self.canvas.update()

    def reset_filter(self):
        if self.original_image is not None:
            self.canvas.image = self.original_image.copy()
            self.canvas.update()
            self.original_image = None

    # ---------- BONUS 3 : Second objet ----------
    def prepare_second_object(self):
        if hasattr(self, "last_prediction"):
            self.history.addItem(
                f"✅ Objet 1 : {self.last_prediction[0]} ({self.last_prediction[1]*100:.1f} %)")
        self.canvas.clear()
        self.result_label.setText("Prédiction : —")
        self.score_label.setText("Confiance : —")
        QMessageBox.information(
            self, "Second objet",
            "Le canvas est effacé.\nDessinez un nouvel objet, puis cliquez sur « Reconnaître ».")

    # ---------- Prédiction ----------
    def do_predict(self):
        try:
            gray = self.canvas.get_numpy_gray()
            cls, conf = predict(gray)
            self.last_prediction = (cls, conf)
            self.result_label.setText(f"Prédiction : {cls}")
            self.score_label.setText(f"Confiance : {conf*100:.2f} %")
            self.history.addItem(f"🔍 {cls} — {conf*100:.2f} %")
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