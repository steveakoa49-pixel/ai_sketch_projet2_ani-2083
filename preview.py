import os, cv2, numpy as np
import matplotlib.pyplot as plt

classes = ["maison", "arbre", "voiture", "chat", "soleil"]
fig, axes = plt.subplots(5, 8, figsize=(12, 8))

for row, cls in enumerate(classes):
    folder = os.path.join("data", cls)
    files = sorted(os.listdir(folder))[:8]
    for col, fname in enumerate(files):
        img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_ylabel(cls)

plt.tight_layout()
plt.show()