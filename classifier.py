import random
import numpy as NumPy
import tensorflow as TensorFlow
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator


def generatePlot(history):
    # Extracción de datos
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Creación del gráfico de accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training')
    plt.plot(epochs, val_acc, 'r', label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Creación del gráfico de loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training')
    plt.plot(epochs, val_loss, 'r', label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

seed_value = 31
random.seed(seed_value)                             # Semilla para Python
NumPy.random.seed(seed_value)                       # Semilla para NumPy
TensorFlow.random.set_seed(seed_value)              # Semilla para TensorFlow/Keras


number_of_images_per_category = 500                              # 100, 300 or 500
dataset_path = r".\Dataset with " + str(number_of_images_per_category) + " images"
batch_size = 16
image_size = (150, 150)
rescale_factor = 1. / 255
validation_split_value = 0.2

data_generator = ImageDataGenerator(
    rescale=rescale_factor,
    validation_split=validation_split_value
)

train_generator = data_generator.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = data_generator.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
model.add(Dropout(0.25))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.summary()


from keras.callbacks import EarlyStopping

# Configurar Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)  # 'patience' es el número de épocas sin mejora después de las cuales el


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

epochs = 30

history_of_train = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[ early_stopping ]
)

generatePlot(history_of_train)

score = model.evaluate(validation_generator, steps=50, verbose=0)  #'steps' es el número de lotes a evaluar

print('Test loss:', score[0])
print('Test accuracy:', score[1])


#x_batch, y_batch = next(train_generator)
#for i in range (0, 5):
#    image = x_batch[i]
#    plt.imshow(image)
#    plt.show()
#
