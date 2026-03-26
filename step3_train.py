import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import json

TRAIN_DIR = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\train"
VALID_DIR = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\valid"

CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
AUTOTUNE    = tf.data.AUTOTUNE
NUM_CLASSES = 5

# Load datasets
print("Loading datasets...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, shuffle=True, seed=42
).cache().shuffle(1000).prefetch(AUTOTUNE)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, shuffle=False
).cache().prefetch(AUTOTUNE)

print(f"Train batches : {tf.data.experimental.cardinality(train_ds).numpy()}")
print(f"Valid batches : {tf.data.experimental.cardinality(valid_ds).numpy()}")

# Build model
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
], name="augmentation")

base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224,224,3))
base.trainable = False

inputs  = layers.Input(shape=(224,224,3))
x       = augment(inputs)
x       = base(x, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(128, activation="relu")(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
model   = models.Model(inputs, outputs, name="AcneDetector")

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

os.makedirs(r"C:\Users\Kartik\acne_project\models", exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=6,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(r"C:\Users\Kartik\acne_project\models\best_acne_model.keras",
                    monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3,
                      min_lr=1e-7, verbose=1),
]

print("\n" + "="*50)
print("  PHASE 1 — Training (Frozen Backbone)")
print("="*50)

history1 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=25,
    callbacks=callbacks,
    verbose=1
)

best_p1 = max(history1.history["val_accuracy"])
print(f"\nPhase 1 complete! Best val accuracy: {best_p1:.4f} ({best_p1*100:.1f}%)")

# Phase 2 — Fine-tune
print("\n" + "="*50)
print("  PHASE 2 — Fine-Tuning (Unfreeze top 20 layers)")
print("="*50)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
print(f"Trainable parameters now: {trainable_count:,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

best_p2 = max(history2.history["val_accuracy"])
print(f"\nPhase 2 complete! Best val accuracy: {best_p2:.4f} ({best_p2*100:.1f}%)")

# Save final model
model.save(r"C:\Users\Kartik\acne_project\models\best_acne_model.keras")
print("Model saved: models/best_acne_model.keras")

# Save training history
combined = {
    "accuracy":     history1.history["accuracy"]     + history2.history["accuracy"],
    "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
    "loss":         history1.history["loss"]         + history2.history["loss"],
    "val_loss":     history1.history["val_loss"]     + history2.history["val_loss"],
    "phase1_end":   len(history1.history["accuracy"]),
    "best_phase1":  best_p1,
    "best_phase2":  best_p2
}

with open(r"C:\Users\Kartik\acne_project\models\training_history.json", "w") as f:
    json.dump(combined, f)

print("History saved: models/training_history.json")
print("\nSTEP 3 COMPLETE")
