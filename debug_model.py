import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load model
print("Loading model...")
model = tf.keras.models.load_model("models/best_acne_model.keras")
print("Model loaded successfully!")

# Check model summary
print("\nModel output shape:", model.output_shape)
print("Model output activation:", model.layers[-1].activation)

CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]

# Check the final dense layer weights
print("\n" + "="*50)
print("Checking final classification layer weights")
print("="*50)

final_layer = model.layers[-1]  # The final Dense layer
weights = final_layer.get_weights()
print(f"Final layer: {final_layer.name}")
print(f"Weights shape: {[w.shape for w in weights]}")
print(f"Weights (kernel): {weights[0]}")
print(f"Biases: {weights[1]}")

# Check if biases are all zeros
print(f"\nBias sum: {np.sum(weights[1])}")
print(f"Bias values: {weights[1]}")

# Try with different preprocessing
print("\n" + "="*50)
print("Test 1: Using [0,1] normalized images (current app method)")
print("="*50)
test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
test_img_01 = test_img.astype(np.float32) / 255.0
test_batch_01 = np.expand_dims(test_img_01, 0)
preds_01 = model.predict(test_batch_01, verbose=0)[0]
print("Prediction:", {CLASS_NAMES[i]: f"{p*100:.2f}%" for i, p in enumerate(preds_01)})

print("\n" + "="*50)
print("Test 2: Using raw [0,255] images (no normalization)")
print("="*50)
test_img_raw = test_img.astype(np.float32)
test_batch_raw = np.expand_dims(test_img_raw, 0)
preds_raw = model.predict(test_batch_raw, verbose=0)[0]
print("Prediction:", {CLASS_NAMES[i]: f"{p*100:.2f}%" for i, p in enumerate(preds_raw)})

print("\n" + "="*50)
print("Test 3: Using EfficientNet preprocessing [-1, 1]")
print("="*50)
from tensorflow.keras.applications.efficientnet import preprocess_input
test_img_en = test_img.astype(np.float32)
test_img_en = preprocess_input(test_img_en)
test_batch_en = np.expand_dims(test_img_en, 0)
preds_en = model.predict(test_batch_en, verbose=0)[0]
print("Prediction:", {CLASS_NAMES[i]: f"{p*100:.2f}%" for i, p in enumerate(preds_en)})

print("\n" + "="*50)
print("Test 4: Using all zeros image")
print("="*50)
test_img_zero = np.zeros((224, 224, 3), dtype=np.float32)
test_batch_zero = np.expand_dims(test_img_zero, 0)
preds_zero = model.predict(test_batch_zero, verbose=0)[0]
print("Prediction:", {CLASS_NAMES[i]: f"{p*100:.2f}%" for i, p in enumerate(preds_zero)})

print("\n" + "="*50)
print("Test 5: Using all ones image")
print("="*50)
test_img_one = np.ones((224, 224, 3), dtype=np.float32)
test_batch_one = np.expand_dims(test_img_one, 0)
preds_one = model.predict(test_batch_one, verbose=0)[0]
print("Prediction:", {CLASS_NAMES[i]: f"{p*100:.2f}%" for i, p in enumerate(preds_one)})

print("\n" + "="*50)
print("Test 6: Using all 128 (gray) image")
print("="*50)
test_img_gray = np.full((224, 224, 3), 128.0, dtype=np.float32)
test_batch_gray = np.expand_dims(test_img_gray, 0)
preds_gray = model.predict(test_batch_gray, verbose=0)[0]
print("Prediction:", {CLASS_NAMES[i]: f"{p*100:.2f}%" for i, p in enumerate(preds_gray)})

print("\n" + "="*50)
print("Summary: Are predictions varying?")
print("="*50)
print(f"Test 1 (0-1):   {CLASS_NAMES[np.argmax(preds_01)]}")
print(f"Test 2 (raw):   {CLASS_NAMES[np.argmax(preds_raw)]}")
print(f"Test 3 (EffNet): {CLASS_NAMES[np.argmax(preds_en)]}")
print(f"Test 4 (zeros): {CLASS_NAMES[np.argmax(preds_zero)]}")
print(f"Test 5 (ones):  {CLASS_NAMES[np.argmax(preds_one)]}")
print(f"Test 6 (gray):  {CLASS_NAMES[np.argmax(preds_gray)]}")

