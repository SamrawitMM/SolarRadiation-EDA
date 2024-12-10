# Data Preprocessing and Visualization Toolkit

## Overview
It includes functionality for cleaning, transforming, and visualizing data, with robust error handling and flexible configuration.

---

## Key Features

### 1. Data Reading
- Handles CSV file loading with support for custom missing value handling using the `na_values` parameter.

### 2. Datetime Extraction
- Extracts components such as date, time, year, month, and day from a timestamp column.
- Reorders columns to ensure a structured dataset for further analysis.

### 3. Data Cleaning
- Detects and replaces negative values in numerical columns with medians or zeros.
- Identifies outliers using the Interquartile Range (IQR) method for skewed distributions.
- Provides a modular framework for detecting and visualizing outliers across datasets.

### 4. Visualization
- **Box Plots**: Quickly identify outliers in selected columns.
- **Line Plots**: Visualize monthly and daily trends to understand seasonal and temporal variations.
- **Hourly Trends**: Analyze hourly averages using group-by operations.
- Visualize the impact of dichotomous variables (e.g., Cleaning vs. Non-Cleaning) on parameters.

### 5. Error Handling
- Incorporates `try-except` blocks to manage common exceptions during file reading and column operations.

### 6. Flexible Configuration
- Supports customization for:
  - Median replacements for negative values.
  - Selection of columns to plot.
  - Titles for visualizations.


## Usage

### Requirements
Ensure the following Python packages are installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

### Running the Script
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```
2. Place your dataset in the appropriate folder.
3. Import the script to your notebook:
   

### Example Dataset
The script assumes a dataset with a timestamp column and numerical data for analysis. Replace placeholder column names in the script with your dataset's column names.



