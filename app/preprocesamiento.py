import cv2
import os
import numpy as np

def suavizar_imagen(image):
    """
    Aplica un filtro de suavizado (filtro de media) a una imagen.
    """
    return cv2.blur(image, (5, 5))

def acentuar_imagen(image):
    """
    Aplica un filtro de acentuado a una imagen.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def procesar_imagenes(input_dir, output_dir):
    """
    Procesa todas las imágenes en un directorio (suavizado y acentuado).
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            
            # Suavizar y acentuar
            suavizada = suavizar_imagen(image)
            acentuada = acentuar_imagen(suavizada)
            
            # Guardar la imagen procesada
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, acentuada)

def preprocesar_imagen(image, target_size=(128, 128)):
    """
    Preprocesa la imagen para que sea compatible con el modelo CNN:
    1. Suaviza la imagen.
    2. Acentúa los bordes.
    3. Redimensiona la imagen.
    4. Normaliza los valores entre 0 y 1.
    
    Args:
        image (numpy.ndarray): Imagen original.
        target_size (tuple): Tamaño objetivo para el modelo (ancho, alto).
    
    Returns:
        numpy.ndarray: Imagen preprocesada.
    """
    # Suavizar y acentuar
    image = suavizar_imagen(image)
    image = acentuar_imagen(image)
    
    # Redimensionar y normalizar
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return image