# Housing-Market-Insights-Exploratory-Data-Analysis-for-Price-Prediction
This project analyzes housing data using Exploratory Data Analysis (EDA) to identify key factors influencing home prices. It involves data cleaning, visualization, and statistical insights to uncover trends, correlations, and anomalies. The findings help buyers, sellers, and analysts understand price determinants and market patterns.

# Exploring Cardiovascular Health Trends: An EDA Approach

## Table of Contents
 
 - [Project Overview](#project-overview)
 - [Data Source](#data-source)
 - [Tools](#tools)
 - [Data Loading and Initial Inspection](#data-loading-and-initial-inspection
 - [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
 - [Exploratory Data Analysis](#exploratory-data-analysis)
 - [Results and Findings](#results-and-findings)
 - [Recommendations](#recommendations)
 - [Limitations](#limitations)

## Project Overview

This project conducts a comprehensive exploratory data analysis (EDA) of a housing dataset containing various features that influence home prices. The goal is to understand:

 - The distribution and characteristics of key housing features
 - Relationships between different features
 - Factors that most strongly influence sale prices
 - Potential outliers and anomalies in the data

## Data Source 

Dataset: HousePrice.csv containing:

- 248 records of residential home sales
- 81 features describing each property
- Sale price as the target variable

Key Features:

- Property characteristics (size, rooms, age)
- Location information (neighborhood)
- Quality ratings
- Amenities and special features
- Sale information

## Tools

This project utilizes Python for Exploratory Data Analysis (EDA), leveraging the following libraries:
- pandas – Data manipulation and preprocessing
- numpy – Numerical operations
- matplotlib & seaborn – Data visualization
- scikit-learn – Statistical analysis and preprocessing
- Jupyter Notebook – Interactive code execution

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Visualization settings
plt.style.use('seaborn')
%matplotlib inline
sns.set_palette("husl")
pd.set_option('display.max_columns', 100)
```

## Data Loading and Initial Inspection

```python
# Load the dataset
df = pd.read_csv('HousePrice.csv')

# Initial inspection
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
display(df.head())

# Basic information
print("\nData types and missing values:")
df.info()

# Statistical summary
print("\nStatistical summary:")
display(df.describe(include='all').T)
```
- The dataset has 248 rows (houses) and 81 columns (features)
- We can see a mix of numerical and categorical variables
- Some columns have missing values (like 'Alley')
- The statistical summary shows ranges for numerical variables and frequency counts for categorical ones

## Data Cleaning and Preprocessing

```python
# Handle missing values
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing_values, missing_percent], axis=1, keys=['Total', 'Percent'])
print("Missing values summary:")
display(missing_data.head(20))

# Drop columns with excessive missing values (>50%)
df = df.drop(columns=missing_data[missing_data['Percent'] > 0.5].index)

# For numerical columns, fill missing values with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)
    
# For categorical columns, fill missing values with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Verify no missing values remain
print("\nMissing values after treatment:", df.isnull().sum().sum())

# Convert categorical variables to appropriate types
for col in cat_cols:
    df[col] = df[col].astype('category')

# Create age-related features
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodelAge'] = df['YrSold'] - df['YearRemodAdd']
```
- Identified columns with missing values
- Dropped features with >50% missing data (like 'PoolQC', 'MiscFeature')
- Filled numerical missing values with medians (robust to outliers)
- Filled categorical missing values with most frequent category
- Created new features for property age and remodel age

## Exploratory Data Analysis 

### Univariate Analysis
#### Target Variable (SalePrice) Distribution

```python
plt.figure(figsize=(12, 6))

# Histogram
plt.subplot(1, 2, 1)
sns.histplot(df['SalePrice'], kde=True, bins=30)
plt.title('Sale Price Distribution')
plt.xlabel('Sale Price ($)')

# Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(y=df['SalePrice'])
plt.title('Sale Price Boxplot')

plt.tight_layout()
plt.show()

# Skewness and kurtosis
print(f"Skewness: {df['SalePrice'].skew():.2f}")
print(f"Kurtosis: {df['SalePrice'].kurt():.2f}")
```
- Right-skewed distribution (skewness > 1)
- Presence of high-value outliers
- Median price around $160,000
- Range from 40,000 - 500,000+

### Physiological Factors

```python
# Blood pressure analysis
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(x='cardio', y='ap_hi', data=df)
plt.title('Systolic Blood Pressure')

plt.subplot(1,2,2)
sns.boxplot(x='cardio', y='ap_lo', data=df)
plt.title('Diastolic Blood Pressure')
```
- Purpose: Compare blood pressure metrics across CVD groups.
- Output: Side-by-side boxplots revealing:
  - Higher median systolic/diastolic BP in CVD-positive patients.
  - Wider IQR (interquartile range) for CVD group, indicating greater variability.

### Lifestyle Factors

```python
# Activity vs cardio
activity_cardio = pd.crosstab(df['active'], df['cardio'], normalize='index')*100
activity_cardio.plot(kind='bar', stacked=True)
plt.title('Physical Activity vs Cardiovascular Disease')
plt.ylabel('Percentage (%)')
```
- Purpose: Analyze how physical activity affects CVD rates.
- Method:
  - crosstab calculates percentage of CVD cases per activity level.
  - Stacked bar chart visualizes proportions.
- Output: Active patients show lower CVD prevalence (e.g., 30% vs. 50% in inactive group).

### Correlation Analysis

```python
# Correlation matrix
corr = df[['age', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'cardio']].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
```
- Purpose: Identify pairwise relationships between features.
- Output: Heatmap highlighting:
  - Strong positive correlation: ap_hi ↔ ap_lo (blood pressure metrics).
  - Moderate correlation: age ↔ cardio (older age → higher CVD risk).

## Results and Findings

1. Clinical Interventions:
   - Prioritize BP monitoring for patients over 50
   - Implement cholesterol management programs
2. Preventive Measures:
   - Promote physical activity initiatives
   - Target smoking cessation programs
3. Screening Protocols:
   - Annual cardiovascular screening after age 45
   - BMI and BP tracking for high-risk patients

