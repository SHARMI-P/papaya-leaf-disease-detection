import tensorflow as tf
from tensorflow.keras import layers, models
import os

# =========================
# 1. Dataset Path
# =========================
dataset_path = r"E:\machine_learning\papaya main dataset"

print("Checking dataset path exists:", os.path.exists(dataset_path))
print("Folders inside dataset:", os.listdir(dataset_path))

# =========================
# 2. Load Dataset
# =========================
img_size = 128   # Reduced size (better for RAM)
batch_size = 16  # Reduced batch size

train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size
)

val_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_size, img_size),
    batch_size=batch_size
)

class_names = train_data.class_names
print("Classes found:", class_names)

# =========================
# 3. Performance Optimization
# =========================
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

# =========================
# 4. Data Augmentation
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# =========================
# 5. Build CNN Model
# =========================
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_size, img_size, 3)),

    data_augmentation,

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting

    layers.Dense(len(class_names), activation='softmax')
])

# =========================
# 6. Compile Model
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 7. Early Stopping
# =========================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =========================
# 8. Train Model
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop]
)

# =========================
# 9. Save Model
# =========================
model.save("papaya_leaf_model_improved.h5")
print("Model Saved Successfully!")
