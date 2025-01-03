import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def data_augmentation(image_path, save_dir):
    """
    Realiza Data Augmentation en la imagen original.
    
    Args:
        image_path (str): Ruta de la imagen original.
        save_dir (str): Directorio donde se guardarán las imágenes aumentadas.
    
    Genera imágenes con:
    - Rotación
    - Volteo
    - Cambio de brillo
    - Zoom
    """
    os.makedirs(save_dir, exist_ok=True)  # Asegurarse de que la carpeta existe

    # Cargar la imagen original
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo encontrar la imagen en {image_path}")
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 1. Rotación
    for angle in [15, -15, 30, -30]:
        rotated = rotate_image(image, angle)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_rotated_{angle}.jpg"), rotated)

    # 2. Volteo
    flipped_h = cv2.flip(image, 1)  # Volteo horizontal
    cv2.imwrite(os.path.join(save_dir, f"{base_name}_flipped_h.jpg"), flipped_h)

    # 3. Cambio de brillo
    for factor in [0.5, 1.5]:  # Menor brillo y mayor brillo
        brightness_changed = change_brightness(image, factor)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_brightness_{factor}.jpg"), brightness_changed)

    # 4. Zoom
    for zoom_factor in [1.2, 1.5]:  # Aumentar el zoom
        zoomed = apply_zoom(image, zoom_factor)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_zoom_{zoom_factor}.jpg"), zoomed)

    print(f"Data Augmentation completado. Imágenes guardadas en {save_dir}")

def rotate_image(image, angle):
    """
    Rota la imagen a un ángulo específico.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def change_brightness(image, factor):
    """
    Cambia el brillo de una imagen.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    brightened_image = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(brightened_image), cv2.COLOR_RGB2BGR)

def apply_zoom(image, zoom_factor):
    """
    Aplica un zoom a la imagen.
    """
    (h, w) = image.shape[:2]
    zoomed_w = int(w / zoom_factor)
    zoomed_h = int(h / zoom_factor)

    # Recortar al centro
    start_x = (w - zoomed_w) // 2
    start_y = (h - zoomed_h) // 2
    cropped = image[start_y:start_y + zoomed_h, start_x:start_x + zoomed_w]

    # Redimensionar a tamaño original
    return cv2.resize(cropped, (w, h))
