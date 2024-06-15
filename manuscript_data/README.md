# Manuscript Data
## 1. Hyperparameter perturbation experiments

Comparison experiments of our proposed method
(ITUNet+AWA+CZPZ+2DBoundaryDoULoss) using different hyperparameters
(Optimizer and Learning Rate Schedule) for prostate lesion segmentation on the
PI-CAI dataset (Fold1). The evaluation metric is the Dice Similarity Coefficient
(DSC).

The combination of the Adam optimizer and the ReduceLROnPlateau
learning rate schedule achieved better result than the result reported in
the manuscript, indicating that more detailed parameter tuning can further improve
the accuracy of the model.

|                 Model                  |       Dataset       | Optimizer  |    Learing Rate Schedule     |   DSC   |
| :-----------------------------------: | :----------------: | :-----: | :---------------: | :-------: |
| ITUNet+AWA+CZPZ+BoundaryDoULoss | PI-CAI(Fold 1) |  Adam   |      PolyLR       |   49.50 (reported in the paper)  |
|    ITUNet+AWA+CZPZ+BoundaryDoULoss     | PI-CAI(Fold 1) |   SGD   |      PolyLR       |   47.95   |
|    ITUNet+AWA+CZPZ+BoundaryDoULoss     | PI-CAI(Fold 1) | Adagrad |      PolyLR       |   47.79   |
|    ITUNet+AWA+CZPZ+BoundaryDoULoss     | PI-CAI(Fold 1) |  Adam   | ReduceLROnPlateau |   50.76   |
|    ITUNet+AWA+CZPZ+BoundaryDoULoss     | PI-CAI(Fold 1) |  Adam   | CosineAnnealingLR | **51.03** |