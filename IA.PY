import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np

# Configurando diretório principal
main_directory = r"C:\Users\Pichau\Desktop\IAGLOBAL\Faulty_solar_panel"

# Configurando geradores de dados
data_gen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

# Gerando conjuntos de treino e validação
train_data = data_gen.flow_from_directory(
    main_directory,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training',
)

val_data = data_gen.flow_from_directory(
    main_directory,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
)

# Criando o modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Compilando o modelo
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinando o modelo
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    validation_steps=val_data.samples // val_data.batch_size
)

# Salvando o modelo
model.save("solar_panel_classifier.h5")

# Avaliando o modelo
eval_results = model.evaluate(val_data)
print(f"Acurácia no conjunto de validação: {eval_results[1] * 100:.2f}%")

# Classes detectadas
print("Classes detectadas:", train_data.class_indices)

# Função para mostrar imagens de cada classe
def display_images_per_class(data, class_indices):
    class_names = list(class_indices.keys())
    fig, axes = plt.subplots(1, len(class_names), figsize=(20, 5))

    for i, class_name in enumerate(class_names):
        for images, labels in data:
            idx = np.where(labels[:, i] == 1)[0]
            if len(idx) > 0:
                axes[i].imshow(images[idx[0]])
                axes[i].set_title(class_name)
                axes[i].axis('off')
                break

    plt.show()

# Mostrar imagens de cada classe
display_images_per_class(val_data, train_data.class_indices)
