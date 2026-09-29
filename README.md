# 🌊 OceanEmbed

## Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature

OceanEmbed is a deep-learning prototype for reconstructing subsurface ocean temperature profiles from multi-source ocean surface observations.

## 🎯 Objective

Estimate ocean temperature at multiple subsurface depths using surface ocean variables.

## 🌐 Prototype Region

North Indian Ocean / Bay of Bengal prototype.

- Latitude: 10–12°N
- Longitude: 88–90°E
- Grid resolution: 0.25°

## 📥 Input Variables

The prototype uses 7 surface variables:

1. SST — Sea Surface Temperature
2. SSS — Sea Surface Salinity
3. SLA — Sea Level Anomaly
4. U Current
5. V Current
6. U Wind
7. V Wind

## 🎯 Target Depths

0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700 and 1000 m.

GLORYS is used as the reference temperature dataset.

## 🧠 Model Architecture

Surface Ocean Variables  
↓  
3×3 Spatial Patch  
↓  
CNN Feature Extraction  
↓  
128-D Ocean Embedding  
↓  
Dense Decoder  
↓  
15-Level Temperature Profile

## 📊 Prototype Results

### OceanEmbed CNN

- RMSE: 2.043 °C
- MAE: 0.979 °C
- R²: 0.918

### MLP Baseline

- RMSE: 1.374 °C
- MAE: 1.110 °C
- R²: 0.962

These results are from a small one-day PoC dataset and should not be interpreted as final scientific validation.

## 📈 Visualizations

The project includes:

- Depth-wise RMSE
- Model comparison
- Actual vs predicted temperature profile
- Surface prediction map
- Prototype uncertainty map
- Ocean Embedding visualization

## 🖥️ Streamlit Explorer

A prototype Streamlit interface is provided in `app/streamlit_app.py`.

It accepts surface ocean variables and reconstructs a 15-depth temperature profile.

## 📁 Project Structure

```text
OceanEmbed_GitHub/
├── README.md
├── requirements.txt
├── models/
├── data/
├── notebooks/
├── src/
├── app/
│   └── streamlit_app.py
└── figures/
