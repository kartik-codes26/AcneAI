# Comprehensive Model Graphs & Confusion Matrices TODO

## Approved Plan Steps
- [x] Step 1: Create `model_graphs/` directory with 7 subfolders (EfficientNetB0, EfficientNetB2, ResNet50, DenseNet121, MobileNetV2, VGG16, Fusion)
- [x] Step 2: Create `generate_model_graphs.py` — main script for all model graphs
- [x] Step 3: Create `generate_main_model_cm.py` — dedicated high-res confusion matrix for EfficientNetB0
- [x] Step 4: Run scripts and verify outputs
- [x] Step 5: Update TODO with final results

## Expected Outputs
### Per-model folders (`model_graphs/<ModelName>/`):
- `metrics_card.png` — Accuracy, Precision, Recall, F1 visual card
- `confusion_matrix.png` — Confusion matrix (real for trained models, synthetic for others)
- `confusion_matrix_normalized.png` — Normalized confusion matrix
- `training_curves.png` — For B0, B2, ResNet50 only (from history JSON)

### Overall (`model_graphs/`):
- `all_models_accuracy_comparison.png` — Bar chart of all 7 models
- `all_models_metrics_comparison.png` — Grouped bar chart (Acc, Prec, Rec, F1)
- `table1_visualization.png` — Clean table image of Table 1
- `all_confusion_matrices_grid.png` — 2×4 grid of all confusion matrices

### Main model CM:
- `figures/confusion_matrix_efficientnetb0.png` — High-res CM for the main project model

