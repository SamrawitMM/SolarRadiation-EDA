# Solar Radiation and Environmental Data Analysis

This consists of analyzing, visualizing, and cleaning a dataset containing solar radiation, environmental, and meteorological measurements. The workflow includes summary statistics, time series analysis, correlation analysis, and advanced visualizations to uncover patterns, trends, and anomalies in the data.

## Key Features

### 1. Data Exploration and Cleaning
- **Summary Statistics**: Calculate measures like mean, median, and standard deviation for numeric columns.
- **Data Quality Checks**: Identify and handle missing values, outliers, and incorrect entries (e.g., negative values in irradiance or temperature columns).
- **Data Cleaning**: 
  - Replace invalid or null entries in critical columns.

### 2. Time Series Analysis
- **Trend Identification**: Plot GHI, DNI, DHI, and temperature (Tamb) over time to observe:
  - Monthly and daily patterns.
  - Anomalies such as extreme peaks or drops in irradiance and temperature.
- **Impact of Cleaning**: Evaluate the effect of the 'Cleaning' process on sensor readings (ModA, ModB).

### 3. Correlation and Relationships
- **Correlation Analysis**: 
  - Correlation matrices and pair plots to explore relationships between GHI, DNI, DHI, and temperature.
  - Investigate wind speed (WS, WSgust, WD) impact on solar irradiance.
- **Scatter and Bubble Charts**:
  - Multi-variable relationships (e.g., GHI vs. Tamb vs. WS) using bubble size to represent additional variables like RH or BP.

### 4. Advanced Visualization
- **Wind Analysis**: Radial bar plots and wind roses to visualize wind speed and direction distribution.
- **Histograms**: Frequency distributions of variables such as GHI, DNI, DHI, WS, and temperatures.
- **Line and Bar Charts**: Temporal trends for solar radiation components and ambient conditions.

### 5. Statistical Insights
- **Outlier Detection**: Use IQR to flag significant deviations in sensor and environmental data.
- **Humidity and Temperature**: Analyze the influence of relative humidity (RH) on temperature and irradiance values.

## Key Objectives
- Uncover patterns and seasonal variations in solar radiation and environmental metrics.
- Evaluate the impact of external factors like cleaning and wind speed on sensor data quality.
- Provide a cleaner, structured dataset for further modeling or operational use.

## Dataset Overview
The dataset includes:
- **Solar Irradiance Components**: GHI, DNI, DHI.
- **Sensor Readings**: ModA, ModB, TModA, TModB.
- **Meteorological Data**: Wind speed (WS), gust speed (WSgust), direction (WD), temperature (Tamb), relative humidity (RH), and barometric pressure (BP).
- **Meta Information**: 'Cleaning' status and 'Comments' column for annotations.

## Tools and Libraries
- Python with pandas, numpy, matplotlib, seaborn, and scipy for data analysis and visualization.

## Future Work
- Integrating with streamlit

## Usage
1. Load the dataset using the provided notebook or scripts.
2. Perform exploratory data analysis (EDA) using the included visualizations and statistical methods implemented in scripts/utility.py
3. Apply the data cleaning techniques to prepare the dataset for modeling or operational use.
4. Generate visual reports to document findings.

---
