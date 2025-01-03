import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

def generar_aumento_datos(user_dir, num_samples):
    """
    Genera imágenes aumentadas para el usuario registrado.
    """
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5]
    )

    images = [os.path.join(user_dir, img) for img in os.listdir(user_dir)]
    for img_path in images:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(150, 150))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=user_dir, save_prefix="aug", save_format="jpg"):
            i += 1
            if i >= num_samples:
                break

def entrenar_modelo():
    """
    Reentrena el modelo con los datos actualizados.
    """
    train_dir = "../users"

    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        print("Error: El directorio de imágenes está vacío o no existe.")
        return

    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical', subset='training'
    )

    val_generator = datagen.flow_from_directory(
        train_dir, target_size=(150, 150), batch_size=32, class_mode='categorical', subset='validation'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    model.fit(train_generator, validation_data=val_generator, epochs=30)

    model.save("scripts/modelo_reconocimiento_mejorado.h5")
    print("Modelo guardado como 'scripts/modelo_reconocimiento_mejorado.h5'")

    class_indices = train_generator.class_indices
    with open("scripts/class_indices.json", "w") as f:
        json.dump(class_indices, f)
