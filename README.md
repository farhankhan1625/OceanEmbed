# 🌊 OceanEmbed

## Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature

OceanEmbed is a deep-learning prototype designed to estimate subsurface ocean temperature profiles from multi-source ocean surface observations. It combines surface ocean variables with spatial and temporal deep-learning features to reconstruct temperature at multiple depths.

**Live Dashboard:** [OceanEmbed Explorer](https://oceanembed-jyjavxv7yvenk9s4brb63m.streamlit.app/)

---

## 🎯 Objective

To develop a data-driven framework that estimates subsurface ocean temperature at multiple depth levels using surface ocean observations.

The project aims to explore how spatial patterns and temporal changes in surface conditions can support subsurface temperature reconstruction.

## 🌍 Prototype Region

North Indian Ocean / Bay of Bengal prototype.

The current dashboard operates on the uploaded OceanEmbed NetCDF dataset. Geographic coverage and prediction availability depend on the selected dataset.

## 🚀 Key Features

- PyTorch-based subsurface temperature reconstruction.
- Seven surface input variables.
- Seven consecutive daily input frames.
- Predictions at 15 depth levels.
- Interactive date and geographic location selection.
- Temperature-depth profile visualization.
- Spatial temperature prediction maps.
- CSV export of predicted temperature profiles.
- Compact learned representation with a 128-dimensional embedding.

## 🛰️ Input Variables

The model uses seven surface variables:

| Variable | Description |
|---|---|
| SST | Sea Surface Temperature |
| SSS | Sea Surface Salinity |
| SLA | Sea Level Anomaly |
| UO | Eastward Surface Current |
| VO | Northward Surface Current |
| Uwind | Eastward Wind Component |
| Vwind | Northward Wind Component |

The model consumes seven consecutive daily frames of these variables.

## 🌡️ Prediction Depths

The model generates estimates at 15 depth levels:

`0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m`

These values represent the output levels configured for the current prototype.

## 🧠 Model Architecture

OceanEmbed uses a PyTorch neural network with convolutional residual feature extraction, temporal attention and spatial decoding.

The model processes a seven-day sequence of seven surface variables and produces a subsurface temperature estimate for each configured depth.

**Architecture overview:**

1. Surface-variable input preparation.
2. Normalization using saved training statistics.
3. Spatial feature extraction through convolutional residual blocks.
4. Temporal feature integration.
5. Spatial decoding.
6. Reconstruction of temperatures at 15 depth levels.
7. Conversion of normalized predictions back to temperature units.

The trained checkpoint is `oceanembed_best.pth`.

## 📊 Evaluation Results

The following results are from the project's previously recorded evaluation reports.

| Metric | 2020 Validation | 2021 Test |
|---|---:|---:|
| Pooled RMSE | 1.406 °C | 1.736 °C |
| Pooled MAE | 1.010 °C | 1.280 °C |
| Pooled Bias | -0.041 °C | -0.253 °C |

### Interpretation

The recorded pooled RMSE was approximately 1.41 °C on the 2020 validation set and 1.74 °C on the 2021 test set.

Performance varies by depth. Some intermediate and deeper levels show weak predictive skill, so the pooled metrics should not be interpreted as uniform accuracy at every depth.

These values are historical evaluation results; they are not a guarantee of accuracy for every dashboard prediction.

## 📈 Visualizations

The repository includes the following figures:

- `figures/OceanEmbed_depthwise_RMSE.png`
- `figures/OceanEmbed_embedding_PCA.png`
- `figures/OceanEmbed_surface_prediction_map.png`
- `figures/OceanEmbed_uncertainty_map.png`
- `figures/actual_vs_predicted_profile.png`
- `figures/model_comparison_R2.png`
- `figures/model_comparison_RMSE.png`

The provenance and consistency of these figures with the current PyTorch model should be verified before interpreting them as current model results.

## 🖥️ Interactive Dashboard

Open the deployed application:

**[Launch OceanEmbed Explorer](https://oceanembed-jyjavxv7yvenk9s4brb63m.streamlit.app/)**

### Dashboard workflow

1. Upload the trained PyTorch checkpoint.
2. Upload `surface_normalization.npz`.
3. Upload `target_temperature_stats.npz`.
4. Upload a monthly OceanEmbed NetCDF file.
5. Select a target date and geographic location.
6. Run temperature reconstruction.
7. Explore the temperature-depth profile and spatial prediction map.
8. Download the predicted profile as a CSV file.

The target date requires the preceding six daily frames.

Uploaded files may need to be supplied again after the application session resets.

## 📁 Repository Structure

```text
OceanEmbed/
├── README.md
├── app/
│   └── streamlit_app.py
├── figures/
│   ├── OceanEmbed_depthwise_RMSE.png
│   ├── OceanEmbed_embedding_PCA.png
│   ├── OceanEmbed_surface_prediction_map.png
│   ├── OceanEmbed_uncertainty_map.png
│   ├── actual_vs_predicted_profile.png
│   ├── model_comparison_R2.png
│   └── model_comparison_RMSE.png
├── models/
│   ├── X_patch_scaler.pkl
│   ├── Y_cnn_scaler.pkl
│   └── oceanembed_cnn.keras
└── requirements.txt
