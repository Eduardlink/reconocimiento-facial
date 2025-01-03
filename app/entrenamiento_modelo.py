import numpy as np
import os
import json
import h5py
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from app.activations import relu, relu_derivative, softmax  # Importar funciones de activación

def cargar_datos(base_dir, target_size=(128, 128)):
    """
    Carga imágenes y etiquetas desde el directorio base.
    """
    X, y, clases = [], [], []
    for label in os.listdir(base_dir):
        clase_dir = os.path.join(base_dir, label)
        if os.path.isdir(clase_dir):
            clases.append(label)
            for filename in os.listdir(clase_dir):
                filepath = os.path.join(clase_dir, filename)
                if filename.endswith((".jpg", ".png")):
                    img = Image.open(filepath).resize(target_size)
                    X.append(np.array(img))
                    y.append(label)
    X = np.array(X).astype("float32") / 255.0
    y = np.array(y)
    return X, y, clases

def inicializar_pesos(input_dim, output_dim):
    """
    Inicializa pesos para una capa densa.
    """
    return np.random.randn(input_dim, output_dim) * 0.01

def cross_entropy_loss(y_hat, y):
    """
    Calcula la pérdida de entropía cruzada.
    """
    m = y.shape[0]
    return -np.sum(y * np.log(y_hat + 1e-8)) / m

def cross_entropy_derivative(y_hat, y):
    """
    Calcula la derivada de la pérdida de entropía cruzada.
    """
    return y_hat - y

def guardar_modelo_h5(output_model_path, W1, b1, W2, b2, clases):
    """
    Guarda el modelo en formato .h5.
    """
    with h5py.File(output_model_path, "w") as f:
        f.create_dataset("W1", data=W1)
        f.create_dataset("b1", data=b1)
        f.create_dataset("W2", data=W2)
        f.create_dataset("b2", data=b2)
        class_indices = {clase: idx for idx, clase in enumerate(clases)}
        f.attrs["class_indices"] = json.dumps(class_indices)
    print(f"Modelo guardado en {output_model_path}")

def entrenar_modelo(base_dir, output_model_path):
    """
    Entrena un modelo CNN con las imágenes procesadas de forma manual y lo guarda en formato .h5.
    """
    # Cargar datos
    X, y, clases = cargar_datos(base_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar pesos
    input_dim = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    hidden_dim = 128
    output_dim = len(clases)

    W1 = inicializar_pesos(input_dim, hidden_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = inicializar_pesos(hidden_dim, output_dim)
    b2 = np.zeros((1, output_dim))

    # Codificar etiquetas
    lb = LabelBinarizer()
    y_train_encoded = lb.fit_transform(y_train)
    y_val_encoded = lb.transform(y_val)

    # Aplanar datos
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    epochs = 10
    learning_rate = 0.01

    for epoch in range(epochs):
        # Forward propagation
        Z1 = np.dot(X_train_flat, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)

        # Pérdida
        loss = cross_entropy_loss(A2, y_train_encoded)
        print(f"Época {epoch+1}/{epochs}, Pérdida: {loss:.4f}")

        # Backward propagation
        dZ2 = cross_entropy_derivative(A2, y_train_encoded)
        dW2 = np.dot(A1.T, dZ2) / X_train.shape[0]
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X_train.shape[0]

        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X_train_flat.T, dZ1) / X_train.shape[0]
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X_train.shape[0]

        # Actualización de pesos
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        # Validación
        Z1_val = np.dot(X_val_flat, W1) + b1
        A1_val = relu(Z1_val)
        Z2_val = np.dot(A1_val, W2) + b2
        A2_val = softmax(Z2_val)

        val_loss = cross_entropy_loss(A2_val, y_val_encoded)
        print(f"Época {epoch+1}/{epochs}, Pérdida de Validación: {val_loss:.4f}")

    # Guardar modelo en formato .h5
    guardar_modelo_h5(output_model_path, W1, b1, W2, b2, clases)

    # Guardar class_indices en JSON
    class_indices_path = os.path.join(os.path.dirname(output_model_path), "class_indices.json")
    class_indices = {clase: idx for idx, clase in enumerate(clases)}
    with open(class_indices_path, "w") as f:
        json.dump(class_indices, f)
    print(f"Mapeo de etiquetas guardado en {class_indices_path}")
