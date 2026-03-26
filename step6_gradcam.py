import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TEST_DIR    = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\test"
CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]

model = tf.keras.models.load_model(r"C:\Users\Kartik\acne_project\models\best_acne_model.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=(224,224),
    batch_size=32, shuffle=True, seed=99
).prefetch(tf.data.AUTOTUNE)

# Use the EfficientNetB0 top_conv layer
layer_name = "top_conv"
print(f"Using layer: {layer_name}")

def get_gradcam(model, img_array, layer_name):
    # Create a model that outputs both the conv layer and the final predictions
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        pred_idx = tf.argmax(preds[0])
        loss     = preds[:, pred_idx]

    grads       = tape.gradient(loss, conv_out)
    pooled      = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out    = conv_out[0]
    heatmap     = conv_out @ pooled[..., tf.newaxis]
    heatmap     = tf.squeeze(heatmap)
    heatmap     = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_idx), float(tf.reduce_max(preds[0]))

fig, axes = plt.subplots(4, 3, figsize=(13, 18))
fig.suptitle("Grad-CAM Explainability — Where the Model Looks",
             fontsize=15, fontweight="bold")

count = 0
for images, labels in test_ds:
    for i in range(len(images)):
        if count >= 4:
            break
        img       = images[i].numpy()
        img_batch = np.expand_dims(img, 0)
        true_cls  = CLASS_NAMES[tf.argmax(labels[i]).numpy()]

        try:
            heatmap, pred_idx, conf = get_gradcam(model, img_batch, layer_name)
            pred_cls = CLASS_NAMES[pred_idx]
            color    = "#27AE60" if pred_cls == true_cls else "#E74C3C"

            hmap_r   = tf.image.resize(heatmap[..., np.newaxis], [224,224]).numpy()[...,0]
            colored  = plt.cm.jet(hmap_r)[...,:3] * 255
            overlay  = np.clip(0.55 * img + 0.45 * colored, 0, 255).astype("uint8")

            axes[count,0].imshow(img.astype("uint8"))
            axes[count,0].set_title(f"Original\nTrue: {true_cls}", fontsize=9)
            axes[count,0].axis("off")

            axes[count,1].imshow(hmap_r, cmap="jet")
            axes[count,1].set_title("Attention Heatmap\n(Red = High focus)", fontsize=9)
            axes[count,1].axis("off")

            axes[count,2].imshow(overlay)
            axes[count,2].set_title(f"Overlay\nPred: {pred_cls} ({conf:.0%})",
                                    fontsize=9, color=color, fontweight="bold")
            axes[count,2].axis("off")
            count += 1
        except Exception as e:
            print(f"Error on image {i}: {e}")
            continue
    if count >= 4:
        break

plt.tight_layout()
plt.savefig(r"C:\Users\Kartik\acne_project\results\04_gradcam.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/04_gradcam.png")
print("\nSTEP 6 COMPLETE")
