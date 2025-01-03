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
    Aplica un filtro de acentuado de pase alto a una imagen.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def aplicar_sobel(image):
    """
    Aplica el filtro Sobel para acentuar bordes.
    Combina los gradientes en las direcciones X e Y.
    
    Args:
        image (numpy.ndarray): Imagen original.
    
    Returns:
        numpy.ndarray: Imagen con bordes resaltados.
    """
    # Convertir a escala de grises para aplicar Sobel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar Sobel en las direcciones X e Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradiente en X
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradiente en Y
    
    # Magnitud combinada
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    
    # Normalizar la magnitud para convertirla a 8 bits
    sobel_combined = np.uint8(cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX))
    
    # Convertir a BGR para mantener compatibilidad
    return cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)

def procesar_imagenes(input_dir, output_dir):
    """
    Procesa todas las imágenes en un directorio (suavizado, acentuado y filtro Sobel).
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            
            # Suavizar, acentuar y aplicar Sobel
            suavizada = suavizar_imagen(image)
            acentuada = acentuar_imagen(suavizada)
            sobel = aplicar_sobel(acentuada)
            
            # Guardar la imagen procesada
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, sobel)

def preprocesar_imagen(image, target_size=(128, 128)):
    """
    Preprocesa la imagen para que sea compatible con el modelo CNN:
    1. Suaviza la imagen.
    2. Acentúa los bordes.
    3. Aplica el filtro Sobel.
    4. Redimensiona la imagen.
    5. Normaliza los valores entre 0 y 1.
    
    Args:
        image (numpy.ndarray): Imagen original.
        target_size (tuple): Tamaño objetivo para el modelo (ancho, alto).
    
    Returns:
        numpy.ndarray: Imagen preprocesada.
    """
    # Suavizar, acentuar y aplicar Sobel
    image = suavizar_imagen(image)
    image = acentuar_imagen(image)
    image = aplicar_sobel(image)
    
    # Redimensionar y normalizar
    image = cv2.resize(image, target_size)
    image = image.astype("float32") / 255.0
    return image
