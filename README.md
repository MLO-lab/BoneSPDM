# BoneSPDM
This repository contains a structured pipeline for analyzing bone data through multiple stages, including data import, visualization, alignment, KDE calculation, and clustering analysis. The pipeline is designed for CSV and IMS image files and processes spatial data for KDE and hierarchical clustering visualization.

## Table of Contents
1. [Requirements](#requirements)
2. [Workflow](#workflow)
   - [Data Import](#data-import)
   - [Data Inspection and Adjustments](#data-inspection-and-adjustments)
   - [Bone Alignment and Transformation](#bone-alignment-and-transformation)
   - [KDE Calculation and Visualization](#kde-calculation-and-visualization)
   - [Hierarchical Clustering and Heatmap](#hierarchical-clustering-and-heatmap)
3. [Usage](#usage)

---

## Requirements

This pipeline relies on several Python packages. Install them using the following command:

```bash
pip install pandas numpy matplotlib scipy scikit-image seaborn
```

## Workflow

### 1. Data Import
The first step is to import the necessary data files, which include:
- **CSV Files**: Contains quantitative data associated with the bone samples.
- **IMS Image Files**: High-resolution images for bone samples.


### 2. Data Inspection and Adjustments
After loading the data, this step visually inspects the image data and performs adjustments, such as flipping the image on the x or y axis to ensure proper orientation.


### 3. Bone Alignment and Transformation
In this step, each bone sample is aligned to a reference bone. All samples are transformed to the reference space, ensuring consistency across data.


### 4. Spatial Probability Density Map (SPDM) Generation and Visualization
The pipeline calculates the Kernel Density Estimate (KDE) to determine spatial probability density for the transformed data.


### 5. Hierarchical Clustering and Heatmap
The final step clusters the transformed data points (e.g., HSCs and random dots) using hierarchical clustering. A heatmap of the clustered data is also generated to visualize spatial relationships.

---

## Usage

1. **Run each section of the pipeline sequentially**, ensuring all paths and parameters are correctly set for your specific dataset.
2. **Inspect intermediate visualizations** to verify data quality and make adjustments as needed.
3. **Save output figures** and clustering data as needed for further analysis or reporting.

For further details on each stage, refer to the code comments and adjust parameters based on your analysis needs.
