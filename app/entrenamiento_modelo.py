import numpy as np
import os
import json
import h5py
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from app.activations import relu, relu_derivative, softmax

def cargar_datos(base_dir, target_size=(128, 128)):
    X, y, clases = [], [], []
    for label in os.listdir(base_dir):
        clase_dir = os.path.join(base_dir, label)
        if os.path.isdir(clase_dir):
            clases.append(label)
            for filename in os.listdir(clase_dir):
                filepath = os.path.join(clase_dir, filename)
                if filename.endswith(('.jpg', '.png')):
                    img = Image.open(filepath).resize(target_size)
                    X.append(np.array(img))
                    y.append(label)
    X = np.array(X).astype('float32') / 255.0
    y = np.array(y)
    return X, y, clases

def inicializar_filtros(filtro_shape):
    return np.random.randn(*filtro_shape) * 0.01

def inicializar_pesos(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * 0.01

def convolucion(X, filtros, bias, stride=1, padding=0):
    n_samples, h_in, w_in, c_in = X.shape
    f_h, f_w, c_in, c_out = filtros.shape
    h_out = (h_in - f_h + 2 * padding) // stride + 1
    w_out = (w_in - f_w + 2 * padding) // stride + 1

    Z = np.zeros((n_samples, h_out, w_out, c_out))
    if padding > 0:
        X_padded = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        X_padded = X

    for i in range(h_out):
        for j in range(w_out):
            region = X_padded[:, i*stride:i*stride+f_h, j*stride:j*stride+f_w, :]
            for k in range(c_out):
                Z[:, i, j, k] = np.sum(region * filtros[:, :, :, k], axis=(1, 2, 3)) + bias[0, 0, 0, k]
    return Z

def max_pooling(X, pool_size=(2, 2), stride=2):
    n_samples, h_in, w_in, c_in = X.shape
    f_h, f_w = pool_size
    h_out = (h_in - f_h) // stride + 1
    w_out = (w_in - f_w) // stride + 1

    Z = np.zeros((n_samples, h_out, w_out, c_in))
    for i in range(h_out):
        for j in range(w_out):
            region = X[:, i*stride:i*stride+f_h, j*stride:j*stride+f_w, :]
            Z[:, i, j, :] = np.max(region, axis=(1, 2))
    return Z

def cross_entropy_loss(y_hat, y):
    m = y.shape[0]
    return -np.sum(y * np.log(y_hat + 1e-8)) / m

def cross_entropy_derivative(y_hat, y):
    return y_hat - y

def calcular_precision(y_hat, y):
    y_pred = np.argmax(y_hat, axis=1)
    y_true = np.argmax(y, axis=1)
    return np.mean(y_pred == y_true)

def guardar_modelo_h5(output_model_path, filtros, bias_conv, W_fc, b_fc, clases):
    with h5py.File(output_model_path, "w") as f:
        f.create_dataset("filtros", data=filtros)
        f.create_dataset("bias_conv", data=bias_conv)
        f.create_dataset("W_fc", data=W_fc)
        f.create_dataset("b_fc", data=b_fc)
        class_indices = {clase: idx for idx, clase in enumerate(clases)}
        f.attrs["class_indices"] = json.dumps(class_indices)
    print(f"Modelo guardado en {output_model_path}")

def entrenar_modelo_convolucional(base_dir, output_model_path):
    X, y, clases = cargar_datos(base_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    filtro_shape = (3, 3, X_train.shape[3], 16)
    filtros = inicializar_filtros(filtro_shape)
    bias_conv = np.zeros((1, 1, 1, 16))

    hidden_dim = 128
    output_dim = len(clases)

    input_dim_fc = (X_train.shape[1] // 2) * (X_train.shape[2] // 2) * 16
    W_fc = inicializar_pesos(input_dim_fc, hidden_dim)
    b_fc = np.zeros((1, hidden_dim))
    W_out = inicializar_pesos(hidden_dim, output_dim)
    b_out = np.zeros((1, output_dim))

    lb = LabelBinarizer()
    y_train_encoded = lb.fit_transform(y_train)
    y_val_encoded = lb.transform(y_val)

    epochs = 50
    learning_rate = 0.01

    for epoch in range(epochs):
        Z_conv = convolucion(X_train, filtros, bias_conv)
        A_conv = relu(Z_conv)
        A_pool = max_pooling(A_conv)
        A_flat = A_pool.reshape(A_pool.shape[0], -1)
        Z_fc = np.dot(A_flat, W_fc) + b_fc
        A_fc = relu(Z_fc)
        Z_out = np.dot(A_fc, W_out) + b_out
        A_out = softmax(Z_out)

        loss = cross_entropy_loss(A_out, y_train_encoded)
        print(f"Época {epoch+1}/{epochs}, Pérdida: {loss:.4f}")

        dZ_out = cross_entropy_derivative(A_out, y_train_encoded)
        dW_out = np.dot(A_fc.T, dZ_out) / X_train.shape[0]
        db_out = np.sum(dZ_out, axis=0, keepdims=True) / X_train.shape[0]

        dA_fc = np.dot(dZ_out, W_out.T)
        dZ_fc = dA_fc * relu_derivative(Z_fc)
        dW_fc = np.dot(A_flat.T, dZ_fc) / X_train.shape[0]
        db_fc = np.sum(dZ_fc, axis=0, keepdims=True) / X_train.shape[0]

        dA_pool = np.dot(dZ_fc, W_fc.T).reshape(A_pool.shape)
        dA_conv = np.zeros_like(A_conv)  # Simplificación para pooling

        filtros -= learning_rate * filtros
        W_fc -= learning_rate * dW_fc
        b_fc -= learning_rate * db_fc
        W_out -= learning_rate * dW_out
        b_out -= learning_rate * db_out

        Z_conv_val = convolucion(X_val, filtros, bias_conv)
        A_conv_val = relu(Z_conv_val)
        A_pool_val = max_pooling(A_conv_val)
        A_flat_val = A_pool_val.reshape(A_pool_val.shape[0], -1)
        Z_fc_val = np.dot(A_flat_val, W_fc) + b_fc
        A_fc_val = relu(Z_fc_val)
        Z_out_val = np.dot(A_fc_val, W_out) + b_out
        A_out_val = softmax(Z_out_val)

        val_loss = cross_entropy_loss(A_out_val, y_val_encoded)
        val_accuracy = calcular_precision(A_out_val, y_val_encoded)
        print(f"Época {epoch+1}/{epochs}, Pérdida de Validación: {val_loss:.4f}, Precisión de Validación: {val_accuracy * 100:.2f}%")

    guardar_modelo_h5(output_model_path, filtros, bias_conv, W_fc, b_fc, clases)
    return val_accuracy * 100
