#!/usr/bin/env python3
"""
card_recognition.py
Punto de partida: detección de contornos de cartas sobre fondo verde,
warp a tamaño estándar y reconocimiento por template matching (rank + suit).

Requiere:
  pip install opencv-python numpy imutils

Uso:
  python card_recognition.py --templates templates/ [--image path/to/photo.jpg | --camera] [--camera-index N]
"""
from time import process_time
import cv2
import numpy as np
import os
import argparse
import imutils

def load_templates(tdir):
    ranks_dir = os.path.join(tdir, "ranks")
    suits_dir = os.path.join(tdir, "suits")

    print(f"Buscando plantillas en: {ranks_dir} y {suits_dir}")

    if not os.path.exists(ranks_dir) or not os.path.exists(suits_dir):
        raise FileNotFoundError(f"No se encontraron las carpetas 'ranks' o 'suits' en {tdir}")
    ranks = {}
    suits = {}
    for fname in os.listdir(ranks_dir):
        key, _ = os.path.splitext(fname)
        img = cv2.imread(os.path.join(ranks_dir, fname), cv2.IMREAD_GRAYSCALE)
        _, imgb = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        ranks[key] = cv2.resize(imgb, (70, 120))
    for fname in os.listdir(suits_dir):
        key, _ = os.path.splitext(fname)
        img = cv2.imread(os.path.join(suits_dir, fname), cv2.IMREAD_GRAYSCALE)
        _, imgb = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        suits[key] = cv2.resize(imgb, (40, 40))
    return ranks, suits

def segment_green(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 50, 30])   # calibrar según tapete
    upper = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def find_card_contours(mask, min_area_ratio=0.01):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    h, w = mask.shape
    min_area = h * w * min_area_ratio
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            candidates.append(approx.reshape(4, 2))
        else:
            # fallback: use minAreaRect to create quad
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int32(box)  # Cambiado de np.int0 a np.int32
            candidates.append(box)
    return candidates

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_card(img, pts, width=200, height=300):
    rect = order_points(pts)
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (width, height))
    return warp

def extract_corner(warp):
    h, w = warp.shape[:2]
    cx = warp[0:int(h*0.28), 0:int(w*0.33)]
    return cx

def preprocess_for_match(img_gray, size):
    img = cv2.resize(img_gray, size)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def match_templates(crop_gray, templates):
    best_key = None
    best_score = -1
    for key, tpl in templates.items():
        tpl_resized = tpl
        crop_r = preprocess_for_match(crop_gray, tpl.shape[::-1])
        res = cv2.matchTemplate(crop_r, tpl_resized, cv2.TM_CCOEFF_NORMED)
        _, mx, _, _ = cv2.minMaxLoc(res)
        if mx > best_score:
            best_score = mx
            best_key = key
    return best_key, best_score

def process_camera(tdir, camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: No se pudo acceder a la cámara con índice {camera_index}.")
        return

    ranks, suits = load_templates(tdir)
    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        orig = frame.copy()
        mask = segment_green(frame)
        cards = find_card_contours(mask)
        results = []

        for pts in cards:
            warp = warp_card(orig, pts)
            warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
            corner = extract_corner(warp)
            corner_gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
            h, w = corner_gray.shape
            rank_roi = corner_gray[0:int(h*0.65), 0:int(w*0.6)]
            suit_roi = corner_gray[int(h*0.55):int(h*0.95), 0:int(w*0.45)]
            rank_key, rank_score = match_templates(rank_roi, ranks)
            suit_key, suit_score = match_templates(suit_roi, suits)
            results.append({
                "pts": pts.tolist(),
                "rank": rank_key,
                "rank_score": float(rank_score),
                "suit": suit_key,
                "suit_score": float(suit_score),
            })

        for r in results:
            print(r)

        cv2.imshow("Original", orig)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def list_cameras():
    print("Buscando cámaras disponibles...")
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            print(f"Cámara encontrada en el índice {index}")
        cap.release()
        index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--templates", required=True, help="Directorio de plantillas (ranks y suits)")
    parser.add_argument("--image", help="Ruta de la imagen a procesar")
    parser.add_argument("--camera", action="store_true", help="Usar la cámara en lugar de una imagen")
    parser.add_argument("--camera-index", type=int, default=0, help="Índice de la cámara (por defecto 0)")
    parser.add_argument("--list-cameras", action="store_true", help="Lista las cámaras disponibles")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
    elif args.camera:
        process_camera(args.templates, args.camera_index)
    elif args.image:
        orig, mask, results = process_time(args.image, args.templates)
        print("Resultados:")
        for r in results:
            print(r)
    else:
        print("Error: Debes proporcionar --image, --camera o --list-cameras.")