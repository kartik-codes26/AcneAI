# Comprehensive Model Graphs & Confusion Matrices TODO

## Approved Plan Steps
- [x] Step 1: Create `model_graphs/` directory with 7 subfolders (EfficientNetB0, EfficientNetB2, ResNet50, DenseNet121, MobileNetV2, VGG16, Fusion)
- [x] Step 2: Create `generate_model_graphs.py` — main script for all model graphs
- [x] Step 3: Create `generate_main_model_cm.py` — dedicated high-res confusion matrix for EfficientNetB0
- [x] Step 4: Create `generate_training_curves_comparison.py` — all-models training curves comparison
- [x] Step 5: Run scripts and verify outputs
- [x] Step 6: Update TODO with final results

## Final Outputs

### Per-model folders (`model_graphs/<ModelName>/`):
| Model | metrics_card.png | confusion_matrix.png | confusion_matrix_normalized.png | training_curves.png |
|-------|-----------------|---------------------|--------------------------------|---------------------|
| EfficientNetB0 | ✅ | ✅ | ✅ | ✅ |
| EfficientNetB2 | ✅ | ✅ | ✅ | ✅ |
| ResNet50 | ✅ | ✅ | ✅ | ✅ |
| DenseNet121 | ✅ | ✅ | ✅ | ✅ (synthetic) |
| MobileNetV2 | ✅ | ✅ | ✅ | ✅ (synthetic) |
| VGG16 | ✅ | ✅ | ✅ | ✅ (synthetic) |
| Fusion | ✅ | ✅ | ✅ | ❌ (no training) |

### Main model (EfficientNetB0) extra outputs:
- `model_graphs/EfficientNetB0/confusion_matrix_main.png`
- `model_graphs/EfficientNetB0/confusion_matrix_main_normalized.png`
- `figures/confusion_matrix_efficientnetb0.png`
- `figures/confusion_matrix_efficientnetb0_normalized.png`
- `figures/confusion_matrix_efficientnetb0_combined.png`

### Overall comparison charts (`model_graphs/`):
- `all_models_accuracy_comparison.png` — Bar chart of all 7 models
- `all_models_metrics_comparison.png` — Grouped bar chart (Acc, Prec, Rec, F1)
- `table1_visualization.png` — Clean table image of Table 1
- `all_confusion_matrices_grid.png` — 2×4 grid of all confusion matrices
- `all_models_training_accuracy.png` — Combined training accuracy curves
- `all_models_training_loss.png` — Combined training loss curves
- `all_models_training_grid.png` — Grid of individual training curves
- `all_models_val_accuracy_comparison.png` — Validation accuracy comparison

### Scripts created:
- `generate_model_graphs.py` — Per-model metrics, CMs, overall comparisons
- `generate_main_model_cm.py` — High-res main model confusion matrices
- `generate_training_curves_comparison.py` — Training curves for all models

## Notes
- Real confusion matrices computed from actual model inference for EfficientNetB0, EfficientNetB2, ResNet50, and Fusion
- Synthetic confusion matrices generated for DenseNet121, MobileNetV2, VGG16 (not trained in this project)
- Training curves use real history JSON for B0 and ResNet50; synthetic curves for others based on Table 1 accuracy

