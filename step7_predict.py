import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TEST_DIR    = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\test"
CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]

SEVERITY = {
    "Blackheads":"Mild",   "Whiteheads":"Mild",
    "Papules":"Moderate",  "Pustules":"Moderate",
    "Cyst":"Severe"
}
SEV_COLOR = {
    "Mild":"#27AE60", "Moderate":"#F39C12", "Severe":"#E74C3C"
}
ADVICE = {
    "Blackheads": "Use salicylic acid cleanser daily. Avoid heavy face oils.",
    "Whiteheads": "Non-comedogenic moisturizer. Gentle cleanser. Try retinoids.",
    "Papules":    "Apply benzoyl peroxide. Do NOT squeeze or pop.",
    "Pustules":   "Topical antibiotics. Keep hands off face. See dermatologist.",
    "Cyst":       "See dermatologist immediately. May need professional drainage."
}

model = tf.keras.models.load_model(r"C:\Users\Kartik\acne_project\models\best_acne_model.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=(224,224),
    batch_size=32, shuffle=True, seed=7
)

for images, labels in test_ds.take(1):
    img       = images[0]
    true_lbl  = CLASS_NAMES[tf.argmax(labels[0]).numpy()]
    probs     = model.predict(tf.expand_dims(img, 0), verbose=0)[0]
    pred_idx  = np.argmax(probs)
    pred_cls  = CLASS_NAMES[pred_idx]
    confidence= probs[pred_idx] * 100
    severity  = SEVERITY[pred_cls]
    sev_color = SEV_COLOR[severity]

    print("\n" + "="*50)
    print("  SINGLE IMAGE PREDICTION RESULT")
    print("="*50)
    print(f"  Predicted  : {pred_cls}")
    print(f"  True Label : {true_lbl}")
    print(f"  Confidence : {confidence:.1f}%")
    print(f"  Severity   : {severity}")
    print(f"  Advice     : {ADVICE[pred_cls]}")
    print(f"\n  All class probabilities:")
    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, probs)):
        bar    = "█" * int(prob * 30)
        marker = " ← PREDICTED" if i == pred_idx else ""
        print(f"    {cls:<15}: {prob*100:>5.1f}%  {bar}{marker}")
    print("="*50)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].imshow(img.numpy().astype("uint8"))
    match = "✓ CORRECT" if pred_cls == true_lbl else "✗ WRONG"
    axes[0].set_title(f"True: {true_lbl}  |  {match}",
                      fontsize=11, fontweight="bold",
                      color="#27AE60" if pred_cls == true_lbl else "#E74C3C")
    axes[0].axis("off")

    colors = [sev_color if i == pred_idx else "#BDC3C7" for i in range(len(CLASS_NAMES))]
    bars   = axes[1].barh(CLASS_NAMES, probs * 100, color=colors, height=0.55, edgecolor="white")
    for bar, p in zip(bars, probs * 100):
        axes[1].text(min(p + 0.5, 96), bar.get_y() + bar.get_height()/2,
                     f"{p:.1f}%", va="center", fontsize=10, fontweight="bold")

    axes[1].set_xlim(0, 112)
    axes[1].set_xlabel("Confidence (%)", fontsize=11)
    axes[1].set_title(
        f"Predicted: {pred_cls}  ({confidence:.1f}%)\nSeverity: {severity}",
        fontsize=11, fontweight="bold", color=sev_color
    )
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle("Single Image Prediction Demo", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(r"C:\Users\Kartik\acne_project\results\05_prediction_demo.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: results/05_prediction_demo.png")
    break

print("\nSTEP 7 COMPLETE")
