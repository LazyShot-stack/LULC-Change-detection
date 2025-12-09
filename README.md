# Urban LULC Change Detection System (2018-2025)

## Overview
This project implements a robust Land Use/Land Cover (LULC) change detection system using **Sentinel-2 satellite imagery** and **Machine Learning (Random Forest)**. It analyzes urban expansion over an 8-year period (2018-2025) for the Bangalore region.

## Methodology
1.  **Data Acquisition**: Automated retrieval of cloud-free Sentinel-2 imagery and Dynamic World ground truth labels via Google Earth Engine (GEE).
2.  **Classification**: A Random Forest classifier is trained on spectral signatures to categorize land cover into 9 classes (Water, Trees, Built Area, etc.).
3.  **Change Detection**: Post-classification comparison is performed to quantify and visualize urban growth trends.

## Outputs
All results are saved in the `results/` directory:
-   **`map_{year}.png`**: Visualized LULC maps for each year.
-   **`change_map.png`**: A heatmap highlighting new urban areas formed between 2018 and 2025.
-   **`analysis_report.txt`**: A text report containing year-by-year urban statistics and growth metrics.
-   **`classification_{year}.tif`**: Raw GeoTIFF classification outputs for GIS software.
