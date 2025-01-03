import cv2
import os
import numpy as np

def suavizar_imagen(image, kernel_size=5):
    """
    Aplica un filtro de suavizado (filtro de media) a una imagen de forma manual.
    
    Args:
        image (numpy.ndarray): Imagen original (en formato NumPy).
        kernel_size (int): Tamaño del kernel (debe ser impar).
    
    Returns:
        numpy.ndarray: Imagen suavizada.
    """
    if kernel_size % 2 == 0:
        raise ValueError("El tamaño del kernel debe ser impar.")

    # Asegurarse de que la imagen sea en escala de grises o RGB
    if len(image.shape) == 3:
        is_rgb = True
    else:
        is_rgb = False
        image = image[..., np.newaxis]  # Añadir un eje para manejar de manera uniforme

    # Dimensiones de la imagen
    h, w, c = image.shape
    offset = kernel_size // 2

    # Imagen con borde para manejar los bordes
    padded_image = np.pad(image, ((offset, offset), (offset, offset), (0, 0)), mode='reflect')

    # Crear una matriz para la imagen suavizada
    suavizada = np.zeros((h, w, c), dtype=np.uint8)

    # Aplicar filtro de media
    for i in range(h):
        for j in range(w):
            for channel in range(c):
                ventana = padded_image[i:i+kernel_size, j:j+kernel_size, channel]
                suavizada[i, j, channel] = np.mean(ventana)

    # Si la imagen original era en escala de grises, devolverla sin el tercer eje
    return suavizada if is_rgb else suavizada[..., 0]

def acentuar_imagen(image):
    """
    Aplica un filtro de acentuado de pase alto a una imagen de forma manual.
    
    Args:
        image (numpy.ndarray): Imagen original.
    
    Returns:
        numpy.ndarray: Imagen con el filtro de acentuado aplicado.
    """
    # Definir el kernel de pase alto
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    # Asegurarse de que la imagen sea en escala de grises o RGB
    if len(image.shape) == 3:
        is_rgb = True
    else:
        is_rgb = False
        image = image[..., np.newaxis]  # Añadir un eje para manejar de manera uniforme

    # Dimensiones de la imagen
    h, w, c = image.shape
    offset = kernel.shape[0] // 2  # Tamaño del desplazamiento (suponiendo kernel cuadrado)

    # Imagen con borde para manejar los bordes
    padded_image = np.pad(image, ((offset, offset), (offset, offset), (0, 0)), mode='reflect')

    # Crear una matriz para la imagen acentuada
    acentuada = np.zeros((h, w, c), dtype=np.uint8)

    # Aplicar convolución
    for i in range(h):
        for j in range(w):
            for channel in range(c):
                ventana = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1], channel]
                valor = np.sum(ventana * kernel)
                # Limitar el rango de valores entre 0 y 255
                acentuada[i, j, channel] = np.clip(valor, 0, 255)

    # Si la imagen original era en escala de grises, devolverla sin el tercer eje
    return acentuada if is_rgb else acentuada[..., 0]

def aplicar_sobel(image):
    """
    Aplica el filtro Sobel manualmente para detectar bordes en las direcciones X e Y.
    Combina ambos gradientes para obtener la magnitud de los bordes.
    
    Args:
        image (numpy.ndarray): Imagen original.
    
    Returns:
        numpy.ndarray: Imagen con bordes resaltados.
    """
    # Convertir a escala de grises si es necesario
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)  # Promedio de canales RGB
    
    # Definir los kernels de Sobel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    # Dimensiones de la imagen
    h, w = image.shape
    offset = sobel_x.shape[0] // 2

    # Imagen con borde para manejar los bordes
    padded_image = np.pad(image, ((offset, offset), (offset, offset)), mode='reflect')

    # Crear matrices para los gradientes en X, Y y la magnitud combinada
    grad_x = np.zeros((h, w), dtype=np.float32)
    grad_y = np.zeros((h, w), dtype=np.float32)
    sobel_combined = np.zeros((h, w), dtype=np.float32)

    # Aplicar convolución
    for i in range(h):
        for j in range(w):
            # Extraer ventana de la imagen
            ventana = padded_image[i:i+sobel_x.shape[0], j:j+sobel_x.shape[1]]
            # Gradiente en X
            grad_x[i, j] = np.sum(ventana * sobel_x)
            # Gradiente en Y
            grad_y[i, j] = np.sum(ventana * sobel_y)

    # Calcular la magnitud combinada
    sobel_combined = np.sqrt(grad_x**2 + grad_y**2)

    # Normalizar la magnitud combinada a rango [0, 255]
    sobel_combined = (sobel_combined / sobel_combined.max()) * 255.0
    sobel_combined = sobel_combined.astype(np.uint8)

    # Convertir a BGR para mantener compatibilidad
    return np.stack([sobel_combined]*3, axis=-1)

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
