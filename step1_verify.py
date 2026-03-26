import tensorflow as tf
import matplotlib.pyplot as plt
import os

TRAIN_DIR = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\train"
VALID_DIR = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\valid"
TEST_DIR  = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\test"

CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="categorical",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

train_batches = tf.data.experimental.cardinality(train_ds).numpy()
valid_batches = tf.data.experimental.cardinality(valid_ds).numpy()
test_batches  = tf.data.experimental.cardinality(test_ds).numpy()

print("\n" + "="*50)
print("  DATASET SUMMARY")
print("="*50)
print(f"  Classes    : {CLASS_NAMES}")
print(f"  Train      : ~{train_batches * BATCH_SIZE} images ({train_batches} batches)")
print(f"  Validation : ~{valid_batches * BATCH_SIZE} images ({valid_batches} batches)")
print(f"  Test       : ~{test_batches  * BATCH_SIZE} images ({test_batches}  batches)")
print("="*50)

# Count images per class in train
print("\n  Images per class (train):")
for cls in CLASS_NAMES:
    path  = os.path.join(TRAIN_DIR, cls)
    count = len([f for f in os.listdir(path) if f.lower().endswith((".jpg",".jpeg",".png"))])
    bar   = "█" * (count // 10)
    print(f"    {cls:<15}: {count:>4} images  {bar}")

# Show 12 sample images
os.makedirs(r"C:\Users\Kartik\acne_project\results", exist_ok=True)

plt.figure(figsize=(15, 9))
for images, labels in train_ds.take(1):
    for i in range(min(12, len(images))):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        label_idx = tf.argmax(labels[i]).numpy()
        plt.title(CLASS_NAMES[label_idx], fontsize=10, fontweight="bold", color="#2C3E50")
        plt.axis("off")

plt.suptitle("Real Acne Dataset — Sample Training Images", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(r"C:\Users\Kartik\acne_project\results\01_sample_images.png", dpi=150)
plt.show()
print("\nSaved: results/01_sample_images.png")
print("\nSTEP 1 COMPLETE")
