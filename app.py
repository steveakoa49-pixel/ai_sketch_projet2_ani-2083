import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QMessageBox,
    QSizePolicy, QFrame, QButtonGroup
)
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QCursor
from PyQt5.QtCore import Qt, QPoint, QBuffer, QIODevice, QSize

from predict import predict

class Canvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(650, 550) 
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #ffffff; border: 2px solid #313244; border-radius: 12px;")
        
        self.image = QImage(self.width(), self.height(), QImage.Format_RGB32)
        self.image.fill(Qt.white) 
        
        self.last_point = QPoint()
        self.drawing = False
        self.brush_size = 12
        self.brush_color = QColor("#11111b") # Noir par défaut

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if self.drawing and (event.buttons() & Qt.LeftButton):
            painter = QPainter(self.image)
            if self.brush_color.alpha() == 0:
                painter.setPen(QPen(QColor(Qt.white), self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            else:
                painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            
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
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        self.image.save(buffer, "PNG")
        
        data = np.frombuffer(buffer.data(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        buffer.close()
        
        if img is None:
            return np.ones((self.height(), self.width()), dtype=np.uint8) * 255
        return img


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neon Sketch AI — Studio")
        self.setGeometry(100, 100, 1250, 700)
        self.setStyleSheet("background-color: #11111b; font-family: 'Segoe UI', Arial, sans-serif;")

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(25)

        # ===== ZONE GAUCHE (Panneau d'outils) =====
        left_frame = QFrame()
        left_frame.setStyleSheet("background-color: #1e1e2e; border-radius: 15px;")
        left_frame.setFixedWidth(340)
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(25, 25, 25, 25)
        left_layout.setSpacing(15)

        title = QLabel("NEON STUDIO")
        title.setStyleSheet("color: #cba6f7; font-size: 18px; font-weight: bold; margin-bottom: 5px;")
        left_layout.addWidget(title)

        lbl_outils = QLabel("OUTILS")
        lbl_outils.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold;")
        left_layout.addWidget(lbl_outils)

        tools_layout = QHBoxLayout()
        self.btn_crayon = QPushButton("✏️ Crayon")
        self.btn_gomme = QPushButton("🧽 Gomme")
        
        self.btn_crayon.clicked.connect(self.activate_pencil)
        self.btn_gomme.clicked.connect(self.activate_eraser)
        
        tools_layout.addWidget(self.btn_crayon)
        tools_layout.addWidget(self.btn_gomme)
        left_layout.addLayout(tools_layout)

        lbl_couleur = QLabel("PALETTE")
        lbl_couleur.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold; margin-top: 10px;")
        left_layout.addWidget(lbl_couleur)

        palette_layout = QHBoxLayout()
        palette_layout.setSpacing(5)
        
        # NOUVELLES COULEURS : Sombres et saturées pour que l'IA les voie bien !
        colors = [
            "#11111b", "#1e66f5", "#d20f39", "#40a02b", "#8839ef", # Ligne 1 (Noir, Bleu, Rouge, Vert, Violet)
            "#fe640b", "#179299", "#df8e1d", "#7287fd", "#4c4f69"  # Ligne 2 (Orange, Sarcelle, Moutarde, Bleu clair, Gris)
        ]
        
        palette_layout_1 = QHBoxLayout()
        palette_layout_2 = QHBoxLayout()
        palette_layout_1.setSpacing(6)
        palette_layout_2.setSpacing(6)

        self.color_group = QButtonGroup(self)
        for i, c in enumerate(colors):
            btn = QPushButton()
            btn.setFixedSize(26, 26)
            btn.setCursor(QCursor(Qt.PointingHandCursor))
            border_color = "white" if i == 0 else "transparent"
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {c}; 
                    border-radius: 13px; 
                    border: 2px solid {border_color};
                }}
                QPushButton:checked {{
                    border: 2px solid white;
                }}
            """)
            btn.setCheckable(True)
            self.color_group.addButton(btn)
            
            if i < 5:
                palette_layout_1.addWidget(btn)
            else:
                palette_layout_2.addWidget(btn)
            
            btn.clicked.connect(lambda checked, color=c: self.set_color(QColor(color)))
            
        self.color_group.buttons()[0].setChecked(True)
        
        palette_container = QVBoxLayout()
        palette_container.addLayout(palette_layout_1)
        palette_container.addLayout(palette_layout_2)
        left_layout.addLayout(palette_container)

        left_layout.addSpacing(10)
        slider_layout = QHBoxLayout()
        lbl_taille = QLabel("Taille :")
        lbl_taille.setStyleSheet("color: #a6adc8; font-size: 12px;")
        self.lbl_slider_val = QLabel("12")
        self.lbl_slider_val.setStyleSheet("color: #cdd6f4; font-size: 12px; font-weight: bold;")
        slider_layout.addWidget(lbl_taille)
        slider_layout.addStretch()
        slider_layout.addWidget(self.lbl_slider_val)
        left_layout.addLayout(slider_layout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(2)
        self.slider.setMaximum(40)
        self.slider.setValue(12)
        self.slider.valueChanged.connect(self.change_brush)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal { border: 1px solid #313244; height: 6px; background: #313244; border-radius: 3px; }
            QSlider::handle:horizontal { background: #cba6f7; width: 16px; margin: -5px 0; border-radius: 8px; }
        """)
        left_layout.addWidget(self.slider)

        left_layout.addSpacing(15)
        lbl_actions = QLabel("ACTIONS")
        lbl_actions.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold;")
        left_layout.addWidget(lbl_actions)

        self.btn_predict = QPushButton("✨ RECONNAÎTRE")
        self.btn_predict.setStyleSheet("""
            QPushButton { background-color: #cba6f7; color: #11111b; border-radius: 8px; padding: 12px; font-weight: bold; font-size: 14px; }
            QPushButton:hover { background-color: #b4befe; }
        """)
        self.btn_predict.clicked.connect(self.do_predict)
        left_layout.addWidget(self.btn_predict)

        self.btn_save = QPushButton("💾 ENREGISTRER")
        self.btn_save.setStyleSheet("""
            QPushButton { background-color: #a6e3a1; color: #11111b; border-radius: 8px; padding: 10px; font-weight: bold; font-size: 13px; margin-top: 5px; }
            QPushButton:hover { background-color: #94e2d5; }
        """)
        self.btn_save.clicked.connect(self.save_image)
        left_layout.addWidget(self.btn_save)

        self.btn_clear = QPushButton("Effacer le canvas")
        self.btn_clear.setStyleSheet("""
            QPushButton { background-color: #f38ba8; color: #11111b; border-radius: 6px; padding: 8px; font-weight: bold; margin-top: 5px;}
            QPushButton:hover { background-color: #eba0ac; }
        """)
        left_layout.addWidget(self.btn_clear)

        left_layout.addSpacing(15)
        lbl_resultat = QLabel("RÉSULTAT IA")
        lbl_resultat.setStyleSheet("color: #a6adc8; font-size: 11px; font-weight: bold;")
        left_layout.addWidget(lbl_resultat)

        self.result_label = QLabel("En attente...")
        self.result_label.setStyleSheet("color: #89dceb; font-size: 24px; font-weight: bold;")
        left_layout.addWidget(self.result_label)

        self.score_label = QLabel("Confiance : —")
        self.score_label.setStyleSheet("color: #a6adc8; font-size: 13px;")
        left_layout.addWidget(self.score_label)

        left_layout.addStretch()
        main_layout.addWidget(left_frame, 0)

        # ===== ZONE DROITE (Canvas) =====
        right_frame = QFrame()
        right_frame.setStyleSheet("background-color: #181825; border-radius: 15px;")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(20, 20, 20, 20)
        
        self.canvas = Canvas()
        right_layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        main_layout.addWidget(right_frame, 1)

        self.btn_clear.clicked.connect(self.canvas.clear)
        
        self.is_erasing = False
        self.update_tool_buttons()

    def change_brush(self, value):
        self.canvas.brush_size = value
        self.lbl_slider_val.setText(str(value))

    def update_tool_buttons(self):
        if self.is_erasing:
            self.btn_gomme.setStyleSheet("background-color: #89b4fa; color: #11111b; border-radius: 6px; padding: 8px; font-weight: bold;")
            self.btn_crayon.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 6px; padding: 8px;")
        else:
            self.btn_crayon.setStyleSheet("background-color: #89b4fa; color: #11111b; border-radius: 6px; padding: 8px; font-weight: bold;")
            self.btn_gomme.setStyleSheet("background-color: #313244; color: #cdd6f4; border-radius: 6px; padding: 8px;")

    def activate_pencil(self):
        self.is_erasing = False
        self.update_tool_buttons()
        checked_btn = self.color_group.checkedButton()
        if checked_btn:
            bg_color = checked_btn.styleSheet().split("background-color: ")[1].split(";")[0]
            self.canvas.brush_color = QColor(bg_color)

    def activate_eraser(self):
        self.is_erasing = True
        self.update_tool_buttons()
        self.canvas.brush_color = QColor(0, 0, 0, 0)

    def set_color(self, color):
        self.activate_pencil()
        self.canvas.brush_color = color

    def save_image(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le dessin", "mon_oeuvre_neon.png", 
            "Images PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            self.canvas.image.save(path)
            QMessageBox.information(self, "Succès", f"Ton dessin a été enregistré !\n\n{path}")

    def do_predict(self):
        try:
            gray = self.canvas.get_numpy_gray()
            cls, conf = predict(gray)
            self.result_label.setText(cls.capitalize())
            self.score_label.setText(f"Confiance : {conf*100:.2f} %")
            
            self.result_label.setStyleSheet("color: #89dceb; font-size: 24px; font-weight: bold;")
                
        except FileNotFoundError:
            QMessageBox.warning(self, "Modèle manquant", "Lance d'abord : python train_model.py")
        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())