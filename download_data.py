import os
import urllib.request
import numpy as np
import cv2

# Correspondance nom français -> nom Quick Draw
CLASSES = {
    "maison":  "house",
    "arbre":   "tree",
    "voiture": "car",
    "chat":    "cat",
    "soleil":  "sun",
}

NB_IMAGES = 500  # nombre d'images par classe
CACHE_DIR = "cache_npy"
DATA_DIR = "data"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"

for fr_name, en_name in CLASSES.items():
    print(f"\n=== {fr_name} ({en_name}) ===")
    npy_path = os.path.join(CACHE_DIR, f"{en_name}.npy")

    # 1) Télécharger si pas déjà fait
    if not os.path.exists(npy_path):
        url = BASE_URL.format(en_name)
        print(f"Téléchargement de {url} ...")
        try:
            urllib.request.urlretrieve(url, npy_path)
            print("  ✅ Téléchargé")
        except Exception as e:
            print(f"  ❌ Erreur : {e}")
            continue
    else:
        print("  ✅ Déjà en cache")

    # 2) Charger les données (format : N x 784 uint8)
    data = np.load(npy_path)
    print(f"  {len(data)} dessins disponibles, on garde les {NB_IMAGES} premiers")

    # 3) Sauvegarder en PNG
    out_dir = os.path.join(DATA_DIR, fr_name)
    os.makedirs(out_dir, exist_ok=True)

    for i, img in enumerate(data[:NB_IMAGES]):
        img_2d = img.reshape(28, 28)
        # Quick Draw : fond blanc (255), trait noir (0) → on garde tel quel
        cv2.imwrite(os.path.join(out_dir, f"{i:04d}.png"), img_2d)

    print(f"  ✅ {NB_IMAGES} images écrites dans data/{fr_name}/")

print("\n🎉 Terminé ! Toutes les images sont dans data/")