import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

TEST_DIR    = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\test"
CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]
IMG_SIZE    = (224, 224)

print("Loading model...")
model = tf.keras.models.load_model(r"C:\Users\Kartik\acne_project\models\best_acne_model.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=IMG_SIZE,
    batch_size=32, shuffle=False
).prefetch(tf.data.AUTOTUNE)

# Evaluate
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\nTest Accuracy : {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"Test Loss     : {test_loss:.4f}")

# Collect predictions
y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n" + "="*65)
print("  CLASSIFICATION REPORT")
print("="*65)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Confusion matrices
cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], linewidths=0.5, annot_kws={"size": 12})
axes[0].set_title(f"Confusion Matrix (Counts)\nTest Accuracy: {test_acc*100:.1f}%",
                  fontsize=13, fontweight="bold")
axes[0].set_xlabel("Predicted Class", fontsize=11)
axes[0].set_ylabel("True Class", fontsize=11)
axes[0].tick_params(axis="x", rotation=45)

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], linewidths=0.5, vmin=0, vmax=1,
            annot_kws={"size": 12})
axes[1].set_title("Confusion Matrix (Normalized %)",
                  fontsize=13, fontweight="bold")
axes[1].set_xlabel("Predicted Class", fontsize=11)
axes[1].set_ylabel("True Class", fontsize=11)
axes[1].tick_params(axis="x", rotation=45)

plt.suptitle(f"Model Evaluation on Test Set  |  Accuracy: {test_acc*100:.1f}%",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(r"C:\Users\Kartik\acne_project\results\03_confusion_matrix.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/03_confusion_matrix.png")
print("\nSTEP 5 COMPLETE")
