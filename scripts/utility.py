import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import cm
from windrose import WindroseAxes
from scipy.stats import shapiro, kruskal, levene, mannwhitneyu, f_oneway
from scipy import stats



# Reads a CSV file and return the data
def read_and_display(file_path, na_values=["NAN", "?", " "]):
    
    try:
        data = pd.read_csv(file_path, na_values=na_values)
        return data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


# Extract date, time, year, month and day from Timestamp
# and return updated dataframe with the new extracted features
def extract_datetime_components(data, timestamp_column, columns_to_order):
    
    try:
        # Ensure the column is in datetime format
        data[timestamp_column] = pd.to_datetime(data[timestamp_column])
        
        # Extract and add new columns
        data["Date"] = data[timestamp_column].dt.date
        data["Time"] = data[timestamp_column].dt.time
        data["Year"] = data[timestamp_column].dt.year
        data["Month"] = data[timestamp_column].dt.month
        data["Day"] = data[timestamp_column].dt.day

        data.drop(columns=[timestamp_column], inplace=True)

        reordered_data = data[columns_to_order]

        return reordered_data
    except KeyError as ke:
        print(f"Error: Column not found in the data - {ke}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Check for values less than 0
# Precipitation (mm/min), Wind speed (WS, WSgust, WSstdev)
# Barometric Pressure (BP)
# Relative Humidity (RH)
# Wind Direction (WD, WDstdev)
# can't be negative
# I replace it with median or 0 assuming this might be erroneous data entry or problem with the measuring device
def detect_and_replace_negative_values(data, median_dict):

    for col in data.select_dtypes(include=['number']).columns:  # Check only numeric columns
        # Count negative values
        count = (data[col] < 0).sum()
        print(f"{col} : {count} negative values")
        
        # Get the median for the column from the dictionary passed
        median = median_dict.get(col, 0)  
        
        # Replace negative values with the median if it's non-negative; otherwise, replace with 0
        replacement_value = median if median >= 0 else 0
        data[col] = data[col].where(data[col] >= 0, replacement_value)
    
    return data

# For skewed data distriubtion it uses IQR to detect for outliers
# Detected outliers displayed
def detect_outliers_iqr(data, skewed_cols):
    
    for col in skewed_cols:
        if col in data.columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = 0
            upper_bound = Q3 + 1.5 * IQR

            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            # data_cleaned = data_cleaned[~((data_cleaned[col] < lower_bound) | (data_cleaned[col] > upper_bound))]

            print(f'Column: {col} - Number of outliers: {outliers.shape[0]}')
            print(f'Lower bound: {lower_bound} & Upper bound: {upper_bound}')

            print('/n')
            print(outliers)
        else:
            print(f'Column {col} not found in the DataFrame.')

    return data


# Displays box plot, oulier visualization
def plot_boxplots(data, columns_to_plot):
    
    plt.figure(figsize=(20, 15))

    for i, column in enumerate(columns_to_plot, 1):
        if column in data.columns:
            plt.subplot(4, 4, i)
            sns.boxplot(x=data[column])
            plt.title(f'Box plot for {column}')
            plt.xlabel(column)
        else:
            print(f'Column {column} not found in the DataFrame.')

    plt.tight_layout()
    plt.show()


# Use z-score for normal distriubtions to detect outliers
# Uses a threshold of 3
# def detect_outliers_z_score(data, columns_to_check):


#     data_cleaned = data.copy()  

#     for col in columns_to_check:
#         if col in data.columns:
#             mean = data[col].mean()
#             std_dev = data[col].std()

#             # Calculate Z-score
#             data_cleaned[f'{col}_z_score'] = (data_cleaned[col] - mean) / std_dev

#             # Identify outliers based on Z-score threshold
#             outliers = data_cleaned[abs(data_cleaned[f'{col}_z_score']) > 3]

#             print(f'Total outliers in {col}: {outliers.shape[0]}')
#             print(f'Outliers in {col}:')
#             print(outliers)
#             print('\n')

#             # Remove outliers from the DataFrame
#             data_cleaned = data_cleaned[abs(data_cleaned[f'{col}_z_score']) <= 3]
#         else:
#             print(f'Column {col} not found in the DataFrame.')

#     # Drop the Z-score columns after filtering
#     data_cleaned = data_cleaned.drop(columns=[f'{col}_z_score' for col in columns_to_check if f'{col}_z_score' in data_cleaned.columns])

#     return data_cleaned
    

# Creating datetime to make it index which will be usable for graphs to do the trend
def preprocess_and_filter_data(data, date_col, time_col, years_to_include):
    
    # Ensure the date and time columns are strings
    data[date_col] = data[date_col].astype(str)
    data[time_col] = data[time_col].astype(str)
    
    # Create a datetime column and set it as the index
    data['Datetime'] = pd.to_datetime(data[date_col] + ' ' + data[time_col])
    data.set_index('Datetime', inplace=True)

    # Ensure 'Date' column is in datetime format
    data[date_col] = pd.to_datetime(data[date_col])

    # Set the 'Date' column as the index for filtering
    data.set_index(date_col, inplace=True)

    # Filter the data to include only the specified years
    data = data[data.index.year.isin(years_to_include)]

    return data


# Averages the month for the two years and display the monthly trend
def plot_monthly_trend(data, columns_to_plot, name):

    title='Monthly Trend of ' + name + ' Solar Radiation'
    
    # Resample the data to monthly averages
    monthly_avg = data.resample('M').mean(numeric_only=True)
    
    # Reset the index to extract the month and year for plotting
    monthly_avg['Month'] = monthly_avg.index.month
    monthly_avg['Year'] = monthly_avg.index.year

    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")

    # Plot each column specified in columns_to_plot
    for col in columns_to_plot:
        sns.lineplot(data=monthly_avg, x='Month', y=col, marker='o', label=f'{col}', linewidth=2)

    plt.title(title, fontsize=15, color='purple')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Value', fontsize=12)
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Averages the day found with in 365 days and display the daily trend
def plot_daily_trends(data, columns_to_plot, title_prefix='Daily Trend of'):

    # Resample the data to daily averages
    daily_avg = data.resample('D').mean(numeric_only=True)
    
    # Create a figure with subplots for each column
    plt.figure(figsize=(18, 16))
    sns.set_style("whitegrid")
    colors = ['orange', 'blue', 'green', 'red']

    for i, col in enumerate(columns_to_plot, 1):
        plt.subplot(len(columns_to_plot), 1, i)
        sns.lineplot(data=daily_avg, x=daily_avg.index, y=col, marker='o', color=colors[i % len(colors)])
        plt.title(f'{title_prefix} {col}', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel(f'Average {col}', fontsize=12)
        plt.grid(True)

    plt.tight_layout()
    plt.show()

# EXtract hour from time column uses group by to find the average and to display the hourly trend
def plot_hourly_trends(data, columns_to_plot, title_prefix='Hourly Trend of'):
    
    # Convert 'Time' column to datetime and extract the hour
    data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S')
    data['Hour'] = data['Time'].dt.hour

    # Group by hour and calculate the mean for each hour
    hourly_avg = data.groupby('Hour').mean(numeric_only=True)

    # Plot hourly trends for specified columns
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    color =['orange', 'blue','green', 'red']

    for i, col in enumerate(columns_to_plot, 1):
        plt.subplot(2, 2, i)
        hourly_avg[col].plot(kind='line', title=f'{title_prefix} {col}', marker='o', color=color[i % len(color)])
        plt.xlabel('Hour')
        plt.ylabel(f'{col} ({ "°C" if col == "Tamb" else "W/m²" })')

    plt.tight_layout()
    plt.show()


# Cleaning Vs ModA, Non Cleaned Vs ModA
# Cleaning Vs ModB, Non Cleaned Vs ModB
def plot_impact_of_cleaning(data, mod_a_col='ModA', mod_b_col='ModB'):

    # Filter data for cleaning and non-cleaning periods
    cleaning_data = data[data['Cleaning'] == 1]
    no_cleaning_data = data[data['Cleaning'] == 0]

    # Plot the impact of cleaning on ModA and ModB over time
    plt.figure(figsize=(18, 12))

    # Plot for ModA with and without cleaning
    plt.subplot(2, 1, 1)
    sns.lineplot(data=cleaning_data, x=cleaning_data.index, y=mod_a_col, label='With Cleaning', color='green', marker='o')
    sns.lineplot(data=no_cleaning_data, x=no_cleaning_data.index, y=mod_a_col, label='Without Cleaning', color='red', marker='o')
    plt.title(f'Impact of Cleaning on {mod_a_col} Sensor Readings', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'{mod_a_col} Sensor Readings', fontsize=12)
    plt.legend()
    plt.grid(True)

    # Plot for ModB with and without cleaning
    plt.subplot(2, 1, 2)
    sns.lineplot(data=cleaning_data, x=cleaning_data.index, y=mod_b_col, label='With Cleaning', color='blue', marker='o')
    sns.lineplot(data=no_cleaning_data, x=no_cleaning_data.index, y=mod_b_col, label='Without Cleaning', color='orange', marker='o')
    plt.title(f'Impact of Cleaning on {mod_b_col} Sensor Readings', fontsize=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(f'{mod_b_col} Sensor Readings', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Correlation analysis using heatmap
def plot_correlation_heatmap(data, numerical_cols):
    
    correlation_matrix = data[numerical_cols].corr()

    plt.figure(figsize=(16, 14))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Sensor Readings and Weather Data', fontsize=16)
    plt.show()


# Correlation Matrix & Scatter matrices of Solar radiation and temperature measures
# Crrelation Matrix & Scatter matrices of wind conditions and solar irraiance
def plot_correlation_and_scatter_matrices(data, cols_solar, cols_wind):

    # Set the style for seaborn plots
    sns.set_style("whitegrid")

    # Correlation Matrix for Solar Radiation and Temperature Measures
    plt.figure(figsize=(12, 8))
    correlation_matrix = data[cols_solar].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix for Solar Radiation and Temperature Measures')
    plt.show()

    # Pair Plot for Solar Radiation and Temperature Measures
    sns.pairplot(data[cols_solar], kind='scatter', plot_kws={'alpha': 0.6})
    plt.suptitle('Pair Plot for Solar Radiation and Temperature Measures', y=1.02)
    plt.show()

    # Scatter Matrix for Wind Conditions and Solar Irradiance
    sns.pairplot(data[cols_wind], kind='scatter', plot_kws={'alpha': 0.6})
    plt.suptitle('Scatter Matrix for Wind Conditions and Solar Irradiance', y=1.02)
    plt.show()

# Correlation and Scatter 
def plot_correlation_and_scatter(data, variables):

    # Correlation Analysis
    correlation_matrix = data[variables].corr()
    print("Correlation Matrix:")
    print(correlation_matrix)

    # Heatmap for correlation
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title('Correlation Matrix: ' + ', '.join(variables))
    plt.show()

    # Scatter Plots with Regression Lines
    plt.figure(figsize=(16, 12))
    
    for i, var in enumerate(variables[1:], 1):  # Start from 1 to create subplots for comparisons
        plt.subplot(2, 2, i)
        sns.scatterplot(data=data, x=variables[0], y=var, alpha=0.6)
        sns.regplot(data=data, x=variables[0], y=var, scatter=False, ci=None)
        plt.title(f'{variables[0]} vs {var}')
        plt.xlabel(variables[0])
        plt.ylabel(var)

    plt.tight_layout()
    plt.show()



# Relationship between two variables with bubble size and color representing additional variables
def plot_bubble_chart(data, x_col, y_col, size_col, color_col):
    
    plt.figure(figsize=(12, 8))

    bubble_size = data[size_col] / data[size_col].max() * 100  

    scatter = plt.scatter(
        x=data[x_col], 
        y=data[y_col], 
        s=bubble_size,  
        c=data[color_col], 
        cmap='viridis', 
        alpha=0.7, 
        edgecolor='k'
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label(f'{color_col} (Units)', fontsize=12)

    plt.title(f'Bubble Chart: {x_col} vs {y_col} vs {color_col} (Bubble Size = {size_col})', fontsize=15)
    plt.xlabel(f'{x_col} (Units)', fontsize=12)
    plt.ylabel(f'{y_col} (Units)', fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Filter highly correlated pairs
def find_highly_correlated_pairs(df, threshold=0.7):
    correlation_matrix = df.corr()
    
    highly_correlated_pairs = []
    
    # Iterate over the upper triangle of the matrix 
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:  # Check if the correlation is above the threshold
                highly_correlated_pairs.append(
                    (correlation_matrix.columns[i], correlation_matrix.columns[j], corr_value)
                )
    
    result_df = pd.DataFrame(highly_correlated_pairs, columns=['Variable 1', 'Variable 2', 'Correlation'])
    
    return result_df



# Plot showing the distribution of wind speed and direction
def plot_wind_rose(data, wind_direction_col, wind_speed_col):
 
    fig = plt.figure(figsize=(10, 10))
    ax = WindroseAxes.from_ax(fig=fig)
    
    ax.bar(
        data[wind_direction_col], 
        data[wind_speed_col], 
        normed=True,  
        bins=np.linspace(0, 30, 6), 
        edgecolor='black', 
        cmap=cm.viridis  
    )
    
    ax.set_title('Wind Rose Plot - Distribution of Wind Direction and Speed')
    ax.set_legend(loc='upper right', title='Wind Speed (m/s)')
    
    plt.show()

# Function for radial bar plot of wind speed by direction
def plot_radial_bar(data, wind_direction_col, wind_speed_col):
    # Group data by wind direction and calculate the mean wind speed
    wind_summary = data.groupby(wind_direction_col)[wind_speed_col].mean().reset_index()
    
    # Create a radial bar plot
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw={'projection': 'polar'})
    theta = np.deg2rad(wind_summary[wind_direction_col])
    bars = ax.bar(theta, wind_summary[wind_speed_col], width=2 * np.pi / len(wind_summary),
                  color='orange', alpha=0.7, edgecolor='black')
    
    ax.set_title('Radial Bar Plot - Average Wind Speed by Direction', va='bottom')
    ax.set_xlabel('Wind Direction')
    ax.set_ylabel('Average Wind Speed (m/s)')
    
    plt.show()


# Count unique values to check the relevance of the Column
def count_unique_values(data):
    
    for col in data.columns:
        unique_counts = data[col].value_counts(dropna=False)  # Count unique values, including NaNs
        print(f'Column: {col}')
        print(unique_counts)
        print('\n') 

# Get the median instead of replacing negative values with 0
def get_median_of_column(df, column_name):
   
    try:
        median_value = df.describe().T.loc[column_name, '50%']
        return median_value
    except KeyError:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")


# Perform statistical tests
def statistical_analysis(df1, df2, df3, variables, sample_frac=0.0038, random_state=42):
   
    # Sample the dataframes
    df1_sample = df1.sample(frac=sample_frac, random_state=random_state)
    df2_sample = df2.sample(frac=sample_frac, random_state=random_state)
    df3_sample = df3.sample(frac=sample_frac, random_state=random_state)

    results = {}

    # Loop through each variable to perform tests
    for var in variables:
        # Check normality for each sample using Shapiro-Wilk test
        shapiro_df1_p = shapiro(df1_sample[var])[1]
        shapiro_df2_p = shapiro(df2_sample[var])[1]
        shapiro_df3_p = shapiro(df3_sample[var])[1]

        results[var] = {
            'Shapiro-Wilk p-value (Location 1)': shapiro_df1_p,
            'Shapiro-Wilk p-value (Location 2)': shapiro_df2_p,
            'Shapiro-Wilk p-value (Location 3)': shapiro_df3_p,
        }

        # If all samples are non-normal (p-value < 0.05), use non-parametric tests
        if shapiro_df1_p < 0.05 and shapiro_df2_p < 0.05 and shapiro_df3_p < 0.05:
            # Kruskal-Wallis test for differences between the groups
            kruskal_test_p = kruskal(df1_sample[var], df2_sample[var], df3_sample[var]).pvalue

            # Mann-Whitney U tests for pairwise group comparisons
            mann_whitney_1_2_p = mannwhitneyu(df1_sample[var], df2_sample[var]).pvalue
            mann_whitney_1_3_p = mannwhitneyu(df1_sample[var], df3_sample[var]).pvalue
            mann_whitney_2_3_p = mannwhitneyu(df2_sample[var], df3_sample[var]).pvalue

            # Levene's test for equality of variances
            levene_test_p = levene(df1_sample[var], df2_sample[var], df3_sample[var]).pvalue

            results[var].update({
                'Kruskal-Wallis p-value': kruskal_test_p,
                'Mann-Whitney U p-value (Location 1 vs 2)': mann_whitney_1_2_p,
                'Mann-Whitney U p-value (Location 1 vs 3)': mann_whitney_1_3_p,
                'Mann-Whitney U p-value (Location 2 vs 3)': mann_whitney_2_3_p,
                'Levene\'s Test p-value': levene_test_p
            })
        else:
            # If at least one sample is normal, perform ANOVA for parametric comparison
            anova_test_p = f_oneway(df1_sample[var], df2_sample[var], df3_sample[var]).pvalue
            results[var]['ANOVA p-value'] = anova_test_p

    return results


# Make sure for normal distribution
def normality_analysis(dataframe, columns):

    # Run Shapiro-Wilk test for normality
    for col in columns:
        stat, p_value = stats.shapiro(dataframe[col])
        print(f'Normality test for {col}: Stat={stat:.4f}, p-value={p_value:.4g}')
        if p_value > 0.05:
            print(f'{col} appears to be normally distributed.')
        else:
            print(f'{col} does not appear to be normally distributed.')

    # Visual Inspection with Q-Q Plot
    for col in columns:
        plt.figure(figsize=(6, 4))
        stats.probplot(dataframe[col], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot for {col}')
        plt.show()

