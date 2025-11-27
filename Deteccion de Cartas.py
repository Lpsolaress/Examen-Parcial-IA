import os
import cv2
import numpy as np
import argparse
from typing import Tuple, Dict, Optional
import imutils

# --- Configurables ---
CARD_WIDTH = 200
CARD_HEIGHT = 300
MIN_CARD_AREA = 1500
CORNER_W = 70  # Ajusta según la posición del palo y el número en tus cartas
CORNER_H = 100  # Ajusta según la posición del palo y el número en tus cartas
RANK_TEMPLATE_SIZE = (50, 70)  # Tamaño ajustado para las plantillas de números
SUIT_TEMPLATE_SIZE = (50, 50)  # Tamaño ajustado para las plantillas de palos
MATCH_THRESHOLD_RANK = 0.4  # Umbral ajustado para números
MATCH_THRESHOLD_SUIT = 0.3  # Umbral ajustado para palos

# Variables globales para mantener las últimas detecciones
last_card = None
last_rank = None
last_suit = None

# --- Utilities ---

def load_templates(templates_root: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Carga imágenes de plantilla para ranks y suits.
    """
    ranks_dir = os.path.join(templates_root, "ranks")
    suits_dir = os.path.join(templates_root, "suits")

    ranks = {}
    suits = {}

    def load_folder(folder: str, target_dict: dict, resize_to: Optional[Tuple[int, int]] = None):
        print(f"Buscando plantillas en: {folder}")
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Plantillas no encontradas en {folder}")
        for fn in os.listdir(folder):
            fp = os.path.join(folder, fn)
            if not (fn.lower().endswith(".png") or fn.lower().endswith(".jpg") or fn.lower().endswith(".jpeg")):
                continue
            name = os.path.splitext(fn)[0]
            img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Advertencia: No se pudo cargar la imagen {fp}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            if resize_to:
                gray = cv2.resize(gray, resize_to, interpolation=cv2.INTER_AREA)
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            target_dict[name] = th
            print(f"Plantilla cargada: {name} desde {fp}")

    load_folder(ranks_dir, ranks, resize_to=RANK_TEMPLATE_SIZE)
    load_folder(suits_dir, suits, resize_to=SUIT_TEMPLATE_SIZE)

    return ranks, suits


def find_card_contours(mask, min_area_ratio=0.01):
    """
    Encuentra contornos que podrían ser cartas en la máscara binaria.
    """
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
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            candidates.append(box)
    return candidates


def four_point_transform(image, pts, width, height):
    """
    Realiza una transformación de perspectiva para obtener una vista superior de la carta.
    """
    rect = np.array(pts, dtype="float32")
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (width, height))
    return warp


def extract_corner_rank_suit(card_img):
    """
    Extrae la esquina superior izquierda de la carta para detectar el rank y el suit.
    """
    return card_img[:CORNER_H, :CORNER_W]


def annotate(image, contour, label):
    """
    Anota la imagen con el nombre de la carta detectada.
    """
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(contour)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def match_rank(corner_img: np.ndarray, rank_templates: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """
    Compara usando matchTemplate (TM_CCOEFF_NORMED) para el área del rank (texto).
    """
    if corner_img.ndim == 3:
        corner_img = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
    proc = cv2.resize(corner_img, RANK_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
    _, proc = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("Rank Recortado", proc)  # Mostrar el área recortada del número
    best_name = "unknown"
    best_score = -1.0
    for name, tpl in rank_templates.items():
        res = cv2.matchTemplate(proc, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_name = name
    return best_name, best_score


def match_suit(symbol_img: np.ndarray, suit_templates: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """
    Compara usando matchTemplate (TM_CCOEFF_NORMED) para el área del suit (símbolo).
    """
    if symbol_img.ndim == 3:
        symbol_img = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2GRAY)
    proc = cv2.resize(symbol_img, SUIT_TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)
    _, proc = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("Suit Recortado", proc)  # Mostrar el área recortada del palo
    best_name = "unknown"
    best_score = -1.0
    for name, tpl in suit_templates.items():
        res = cv2.matchTemplate(proc, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_name = name
    return best_name, best_score


def process_card(warp: np.ndarray, ranks_tpl: Dict[str, np.ndarray], suits_tpl: Dict[str, np.ndarray]) -> Tuple[str, str]:
    """
    Procesa la carta recortada para detectar el rank y el suit.
    """
    corner = extract_corner_rank_suit(warp)
    rank_name, _ = match_rank(corner[:, CORNER_W // 2:], ranks_tpl)
    suit_name, _ = match_suit(corner[:, :CORNER_W // 2], suits_tpl)
    return rank_name, suit_name


def process_frame(frame: np.ndarray, ranks_tpl: Dict[str, np.ndarray], suits_tpl: Dict[str, np.ndarray]) -> Tuple[np.ndarray, list]:
    """
    Procesa un frame, detecta cartas y devuelve frame anotado + lista de resultados.
    """
    global last_card, last_rank, last_suit

    orig = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    bin_img = 255 - mask_green

    card_contours = find_card_contours(bin_img)
    results = []

    for cnt in card_contours:
        try:
            warp = four_point_transform(orig, cnt, CARD_WIDTH, CARD_HEIGHT)
        except Exception:
            continue

        rank_name, suit_name = process_card(warp, ranks_tpl, suits_tpl)
        label = f"{rank_name} of {suit_name}"
        annotate(orig, cnt, label)

        # Actualizar las ventanas solo si se detecta una carta
        last_card = warp
        last_rank = warp[:CORNER_H, CORNER_W // 2:]  # Número
        last_suit = warp[:CORNER_H, :CORNER_W // 2]  # Palo

        results.append({
            "rank": rank_name,
            "suit": suit_name,
        })

    # Mostrar las ventanas fijas
    if last_card is not None:
        cv2.imshow("Primera Cámara - Carta Completa", last_card)
    if last_suit is not None:
        cv2.imshow("Segunda Cámara - Palo", last_suit)
    if last_rank is not None:
        cv2.imshow("Tercera Cámara - Número", last_rank)

    return orig, results


def main():
    parser = argparse.ArgumentParser(description="Detectar cartas y reconocer rank/suit con plantillas.")
    parser.add_argument("--templates", "-t", required=True, help="Carpeta root de templates (templates/)")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Índice de cámara (opcional)")
    args = parser.parse_args()

    ranks_tpl, suits_tpl = load_templates(args.templates)
    print(f"Templates cargadas: ranks={len(ranks_tpl)} suits={len(suits_tpl)}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: no se pudo abrir la cámara en el índice {args.camera}.")
        return

    print("Presione 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, _ = process_frame(frame, ranks_tpl, suits_tpl)
        cv2.imshow("Frame Principal", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()