import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]
IMG_SIZE    = (224, 224)
NUM_CLASSES = 5

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
], name="augmentation")

base = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base.trainable = False  # Freeze for Phase 1
 
inputs  = layers.Input(shape=(224, 224, 3), name="input_image")
x       = augment(inputs)
x       = base(x, training=False)
x       = layers.GlobalAveragePooling2D(name="gap")(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation="relu", name="dense_1")(x)
x       = layers.Dropout(0.4, name="dropout_1")(x)
x       = layers.Dense(128, activation="relu", name="dense_2")(x)
x       = layers.Dropout(0.3, name="dropout_2")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32", name="predictions")(x)

model = models.Model(inputs, outputs, name="AcneDetector_5Class")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.summary()

total     = model.count_params()
trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
frozen    = total - trainable

print(f"\n{'='*45}")
print(f"  MODEL STATISTICS")
print(f"{'='*45}")
print(f"  Total parameters     : {total:,}")
print(f"  Trainable parameters : {trainable:,}")
print(f"  Frozen parameters    : {frozen:,}")
print(f"  Output classes       : {NUM_CLASSES}")
print(f"  Classes              : {CLASS_NAMES}")
print(f"{'='*45}")
print("\nSTEP 2 COMPLETE")
