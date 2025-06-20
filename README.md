A method for semi-supervised anomaly detection in SWaT and WADI datasets

The model utilizes unsupervised pre-training to predict time series values. The core prediction model incorporates a transformer encoder with RoPE embeddings and a graph structure layer from the paper Graph Neural Networks with Trainable Adjacency Matrices for Fault Diagnosis on Multivariate Sensor Data.
Training Process

    Pre-training: An initial model is trained using unsupervised techniques.
    Fine-tuning: The pre-trained model is then used to train a CatBoost model on a small, randomly selected dataset from the validation sample.
    Validation: The validation set is derived from the initial days of the test sample.
    Data Augmentation: The data undergoes augmentation to enhance training effectiveness.
    Label Usage: Less than 1% of the available labels are utilized in the training process.

Future Enhancements
    Active Learning: Implementation of active learning strategies to identify the most informative examples and reduce the reliance on large-scale labeling effor

Main file is ts_train.py witch contains forcasting model training.
Anomaly detection method in swat.ipynb and wadi.ipynb


| Method          | Precision (SWaT) | Recall (SWaT) | F1-score (SWaT) | Precision (WADI) | Recall (WADI) | F1-score (WADI) |
|-----------------|------------------|---------------|-----------------|------------------|---------------|-----------------|
| PCA             | 22.9             | 21.6          | 23.0            | 39.5             | 5.6           | 9.8             |
| KNN             | 7.8              | 7.8           | 7.8             | 7.6              | 7.7           | 7.6             |
| AE              | 72.6             | 52.6          | 61.0            | 34.3             | 34.3          | 34.3            |
| LSTM-VAE        | 96.2             | 52.6          | 68.0            | 87.7             | 14.4          | 24.7            |
| MAD-GAN         | 98.7             | 63.7          | 77.4            | 41.4             | 33.9          | 37.2            |
| LSTM-NDT        | 86.7             | 67.2          | 75.7            | 42.2             | 27.7          | 33.4            |
| FuGLAD          | 89.9             | 83.9          | 86.8            | 74.3             | 43.5          | 54.9            |
| GBAD            | 96.8             | 71.6          | 82.3            | 79.9             | 43.4          | 56.3            |
| iADCPS          | 94.6             | 85.3          | 89.7            | 92.7             | 65.2          | 76.6            |
| Proposed method | 98.4             | 87.8          | 92.7            | 99.2             | 69.1          | 81.5            |

Most of the baselines are taken from iADCPS: Time Series Anomaly Detection for Evolving Cyber-physical Systems via Incremental Meta-learning

Ablation study:

|             Method           |      SWaT    |              |      WADI     |             |
|:---------------------------:|:-------------:|:------------:|:-------------:|:-------------:|
|                             |       MAE     |     SMAPE    |       MAE     |      SMAPE    |
|        Method               |     0.0154    |     3.05%    |     0.0354    |     22.44%    |
|        W/o transformer      |     0.0397    |     3.50%    |     0.1308    |     24.66%    |
|        W/o graphs           |     0.0224    |     3.17%    |     0.0389    |     22.50%    |
|        W/o RoPE             |     0.0169    |     3.60%    |     0.0372    |     23.55%    |
