"""
generate_main_model_cm.py
Dedicated high-resolution confusion matrix for EfficientNetB0
(the main model used in the AcneAI project).
Outputs saved to both figures/ and model_graphs/EfficientNetB0/
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

PROJECT_DIR = r"C:/Users/Kartik/OneDrive/Desktop/acne_project"
MODELS_DIR  = PROJECT_DIR + "/models"
TEST_DIR    = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\test"
CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]
AUTOTUNE    = tf.data.AUTOTUNE

print("="*60)
print(" GENERATING MAIN MODEL CONFUSION MATRIX")
print(" Model: EfficientNetB0")
print("="*60)

# Load model
print("\nLoading EfficientNetB0...")
model_b0 = tf.keras.models.load_model(MODELS_DIR + "/best_acne_model.keras")

# Load test data
print("Loading test dataset...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=(224, 224),
    batch_size=32, shuffle=False
).prefetch(AUTOTUNE)

# Predictions
print("Running predictions...")
all_probs, all_labels = [], []
for imgs, lbls in test_ds:
    p = model_b0.predict(imgs, verbose=0)
    all_probs.append(p)
    all_labels.append(lbls.numpy())

probs_b0 = np.vstack(all_probs)
labels   = np.vstack(all_labels)
y_true   = np.argmax(labels, axis=1)
y_pred   = np.argmax(probs_b0, axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Overall accuracy
acc = np.mean(y_pred == y_true)
print(f"\nEfficientNetB0 Test Accuracy: {acc*100:.2f}%")

# ============================================================
# 1. Counts version
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=1, linecolor='white',
            annot_kws={"size": 14, "weight": "bold"},
            cbar_kws={"shrink": 0.8})
ax.set_title(f'EfficientNetB0 — Confusion Matrix\nTest Accuracy: {acc*100:.2f}%',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

save_counts = PROJECT_DIR + "/figures/confusion_matrix_efficientnetb0.png"
plt.savefig(save_counts, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved counts CM: {save_counts}")

# Also save to model_graphs folder
os.makedirs(PROJECT_DIR + "/model_graphs/EfficientNetB0", exist_ok=True)
save_counts2 = PROJECT_DIR + "/model_graphs/EfficientNetB0/confusion_matrix_main.png"
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=1, linecolor='white',
            annot_kws={"size": 14, "weight": "bold"},
            cbar_kws={"shrink": 0.8})
ax.set_title(f'EfficientNetB0 — Confusion Matrix\nTest Accuracy: {acc*100:.2f}%',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig(save_counts2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved counts CM: {save_counts2}")

# ============================================================
# 2. Normalized version
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=1, linecolor='white',
            annot_kws={"size": 14, "weight": "bold"},
            vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
ax.set_title(f'EfficientNetB0 — Normalized Confusion Matrix\nTest Accuracy: {acc*100:.2f}%',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

save_norm = PROJECT_DIR + "/figures/confusion_matrix_efficientnetb0_normalized.png"
plt.savefig(save_norm, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved normalized CM: {save_norm}")

# Also to model_graphs
save_norm2 = PROJECT_DIR + "/model_graphs/EfficientNetB0/confusion_matrix_main_normalized.png"
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=1, linecolor='white',
            annot_kws={"size": 14, "weight": "bold"},
            vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
ax.set_title(f'EfficientNetB0 — Normalized Confusion Matrix\nTest Accuracy: {acc*100:.2f}%',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig(save_norm2, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved normalized CM: {save_norm2}")

# ============================================================
# 3. Side-by-side (counts + normalized)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], linewidths=1, linecolor='white',
            annot_kws={"size": 13, "weight": "bold"})
axes[0].set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('True', fontsize=12)

sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], linewidths=1, linecolor='white',
            annot_kws={"size": 13, "weight": "bold"}, vmin=0, vmax=1)
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('True', fontsize=12)

fig.suptitle(f'EfficientNetB0 — Main Model Evaluation (Accuracy: {acc*100:.2f}%)',
             fontsize=15, fontweight='bold')
plt.tight_layout()

save_side = PROJECT_DIR + "/figures/confusion_matrix_efficientnetb0_combined.png"
plt.savefig(save_side, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved combined CM: {save_side}")

print("\n" + "="*60)
print(" MAIN MODEL CONFUSION MATRICES COMPLETE!")
print("="*60)


