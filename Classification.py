import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15

MODEL_PATH = r"C:\Users\gowthami\PycharmProjects\Brain Tumor\.venv\New_BrainTumorClassification_2025\ModelFiles\brain_tumor_cnn.h5"
CLASS_INDEX_PATH = r"C:\Users\gowthami\PycharmProjects\Brain Tumor\.venv\New_BrainTumorClassification_2025\ModelFiles\class_indices.json"

TRAIN_DIR = r"C:\Users\gowthami\PycharmProjects\Brain Tumor\.venv\New_BrainTumorClassification_2025\Dataset\Ctrain\Training"
TEST_DIR  = r"C:\Users\gowthami\PycharmProjects\Brain Tumor\.venv\New_BrainTumorClassification_2025\Dataset\Ctest\Testing"

# -----------------------------
# If model exists → load
# -----------------------------
if os.path.exists(MODEL_PATH):
    print("✅ Model found. Loading model...")
    model = load_model(MODEL_PATH)

    if os.path.exists(CLASS_INDEX_PATH):
        with open(CLASS_INDEX_PATH, "r") as f:
            class_indices = json.load(f)
        print("📌 Loaded class indices:", class_indices)

else:
    print("🚀 Model not found. Training new model...")

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    # 🔴 FORCE CLASS ORDER (IMPORTANT FIX)
    train_data = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_data = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    print("Class indices:", train_data.class_indices)

    # ✅ Print and save class indices
    print("📌 Class Indices:", train_data.class_indices)

    with open(CLASS_INDEX_PATH, "w") as f:
        json.dump(train_data.class_indices, f)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(4, activation='softmax')  # 🔥 4 classes
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS
    )

    model.save(MODEL_PATH)
    print("💾 Model saved successfully")

    # Accuracy graph
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.legend()
    plt.savefig(
        r"C:\Users\gowthami\PycharmProjects\Brain Tumor\.venv\New_BrainTumorClassification_2025\CReports\accuracy.png"
    )
    plt.show()

    # Loss graph
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend()
    plt.savefig(
        r"C:\Users\gowthami\PycharmProjects\Brain Tumor\.venv\New_BrainTumorClassification_2025\CReports\loss.png"
    )
    plt.show()
