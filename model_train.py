import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from keras.optimizers import Adam

# Parameters
IMAGE_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 10
DATA_DIR = "Brain_Tumor_Dataset"

# Data preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')

# CNN Functional Model
input_layer = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = Conv2D(32, (3,3), activation='relu')(input_layer)
x = MaxPooling2D(2,2)(x)
x = Conv2D(64, (3,3), activation='relu', name='conv2d_1')(x)
x = MaxPooling2D(2,2)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(4, activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

loss, acc = model.evaluate(val_generator)
with open("metrics.txt", "w") as f:
    f.write(f"Validation Accuracy: {acc:.2f}")

model.save("brain_tumor_cnn_model.h5")
print("✅ Model trained and saved.")