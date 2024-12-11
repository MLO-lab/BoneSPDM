# BoneSPDM
This repository contains a structured pipeline for analyzing bone data through multiple stages, including data import, visualization, alignment, KDE calculation, and clustering analysis. The pipeline is designed for CSV and IMS image files and processes spatial data for KDE and hierarchical clustering visualization.

## Table of Contents
-  [Requirements](#requirements)
-  [Workflow](#workflow)
   - [1. Data Import](#1-data-import)
   - [2. Data Inspection and Adjustments](#2-data-inspection-and-adjustments)
   - [3. Bone Alignment and Transformation](#3-bone-alignment-and-transformation)
   - [4. KDE Calculation and Visualization](#4-kde-calculation-and-visualization)
   - [5. Clustering, Prediction and Visualization](#5-clustering-prediction-and-visualization)
- [Usage](#usage)

---

## Requirements

This pipeline relies on several Python packages, which can be installed automatically via the provided Conda environment file. 

### Installation

1. Ensure you have [Conda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.
2. Clone the repository and navigate to the directory
3. Create the Conda environment using the environment.yml file:
   ```bash
   conda env create -f environment.yml
4. Activate the environment:
   ```bash
   conda activate BoneSPDM

These commands will set up the necessary environment with all required dependencies.
### Data Arrangement

To ensure proper functionality, organize your data in the following folder structure within the project's root directory:

- **Data folders (without inhibitor)**: Organize other data files by time points in subdirectories named as follows:
  - **`d0/`**: For day 0 data.
  - **`d5/`**: For day 5 data.
  - **`d10/`**: For day 10 data.
  - **`d15/`**: For day 15 data.
  - **`d30/`**: For day 30 data.
  - **`reference_bone/`**: Place the reference bone image in this folder.
- **Data Folders (with inhibitor)**: If inhibitor data is available, organize it in a separate directory from the data without inhibitors. Subdivide this directory by time points as follows:
   - **`d15`**: For inhibitor data for day 15. For example, `inhibitor/d15/` would store day 15 inhibitor data.

Each folder should contain the relevant data files needed for analysis at that time point. This structured arrangement ensures smooth data processing within the pipeline.

## Workflow

### 1. Data Import
The first step is to import the necessary data files, which include:
- **CSV Files**: Contains quantitative data (positions) associated with the bone samples.
- **IMS Image Files**: Images for bone samples.


### 2. Data Inspection and Adjustments
After loading the data, this step visually inspects the image data and performs adjustments, such as flipping the image on the x or y axis to ensure proper orientation.


### 3. Bone Alignment and Transformation
In this step, each bone sample is aligned to a reference bone. All samples are transformed to the reference space, ensuring consistency across data.


### 4. Spatial Probability Density Map (SPDM) Generation and Visualization
The pipeline calculates the Kernel Density Estimate (KDE) to determine spatial probability density for the transformed data.


### 5. Clustering, Prediction and Visualization
The final step clusters the transformed data points (e.g., HSCs and random dots) using consensus clustering and decide the optimal cluster numbers based on the Silhouette Score. Random forest is applied to predict the cluster labels for data with inhibitos. Heatmaps of the clustered data are also generated to visualize spatial relationships. Scatter plots, and stacked bar plots are also available. Finally, the Mann-Whitney U test is performed to compare the cluster compositions between different groups.

---

## Usage

1. **Run each section of the pipeline sequentially**, ensuring all paths and parameters are correctly set for your specific dataset.
2. **Inspect intermediate visualizations** to verify data quality and make adjustments as needed.
3. **Save output figures** and clustering data as needed for further analysis or reporting.
