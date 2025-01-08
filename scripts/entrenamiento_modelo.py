import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

def entrenar_modelo(base_dir, output_model_path):
    """
    Entrena un modelo CNN con las imágenes procesadas y devuelve la precisión en el conjunto de validación.
    
    Args:
        base_dir (str): Directorio base con carpetas por cada usuario.
        output_model_path (str): Ruta para guardar el modelo entrenado.
    
    Returns:
        float: Precisión del modelo en el conjunto de validación.
    """
    # Configuración de generadores de datos
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
    
    train_gen = datagen.flow_from_directory(
        base_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical",
        subset="training"
    )
    
    val_gen = datagen.flow_from_directory(
        base_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode="categorical",
        subset="validation"
    )
    
    # Guardar class_indices (mapeo de etiquetas)
    class_indices_path = os.path.join(os.path.dirname(output_model_path), "class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(train_gen.class_indices, f)
    print(f"Mapeo de etiquetas guardado en {class_indices_path}")
    
    # Modelo CNN
    # Modelo CNN con Dropout
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),  # Dropout después de la primera capa de pooling

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),  # Dropout después de la segunda capa de pooling

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),

        tf.keras.layers.Dense(len(train_gen.class_indices), activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Entrenar modelo
    model.fit(train_gen, validation_data=val_gen, epochs=50)
    
    # Evaluar modelo en el conjunto de validación
    loss, accuracy = model.evaluate(val_gen, verbose=1)
    print(f"Precisión en el conjunto de validación: {accuracy * 100:.2f}%")
    
    # Guardar modelo
    model.save(output_model_path)
    print(f"Modelo guardado en {output_model_path}")
    
    # Devolver la precisión en porcentaje
    return accuracy * 100
