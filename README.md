# Quantitative Research Data Cleaning Documentation

## 1. INTRODUCTION

In quantitative research, accurate and reliable data is crucial for generating meaningful insights. However, the presence of null values and outliers often complicates the data analysis process:

- **Null Values**: These are missing or incomplete data points, which, if left unaddressed, can lead to biased results, incomplete analyses, and misinterpretations.
- **Outliers**: These are data points that deviate significantly from other observations. They can distort statistical analyses, impact models, and lead to unreliable conclusions if not properly managed.

This project focuses on developing efficient methods for detecting and treating such issues to enhance data integrity and research reliability.

## 2. PROBLEM STATEMENT

Data irregularities such as null values and outliers pose significant challenges in quantitative research. Key problems include:

- **Misleading Results**: Incomplete or inaccurate data can skew analyses.
- **Biased Conclusions**: Unaddressed anomalies can lead to incorrect inferences.
- **Analysis Complexity**: Manual detection and treatment of these issues can be time-consuming and error-prone.

Addressing these challenges is critical for ensuring data-driven decisions are based on accurate and complete information.

## 3. IMPORTANCE OF THIS PROJECT

- **Enhances Data Quality**: By identifying and treating null values and outliers, the project improves data reliability and validity.
- **Facilitates Accurate Analysis**: Ensures that statistical models and conclusions are based on clean and consistent datasets.
- **Saves Time**: Automating the cleaning process reduces the effort and time required for data preprocessing.
- **Broad Applicability**: Provides reusable methods for various research fields, including healthcare, economics, and social sciences.

## 4. MAIN OBJECTIVE

To develop and implement automated methods for detecting, visualizing, and treating null values and outliers, ensuring the integrity of quantitative datasets and enhancing research outcomes.

## 5. METHODOLOGY

### 5.1 Null Value Detection and Treatment

- **Detection**:
  - Use descriptive statistics to identify missing values.
  - Visualize missing data patterns using heatmaps or bar charts.

- **Treatment**:
  - **Imputation Techniques**: Replace null values with the mean, median, or mode of the dataset.
  - **Removal**: Drop records with null values if their occurrence is negligible.
  - **Domain-Specific Strategies**: Apply contextual rules based on the nature of the dataset.

### 5.2 Outlier Detection and Treatment

- **Detection**:
  - **Statistical Methods**: Use Z-scores or the Interquartile Range (IQR) method to identify anomalies.
  - **Visualization**: Employ box plots, scatter plots, or histograms to highlight outliers.

- **Treatment**:
  - **Winsorization**: Cap extreme values at predefined percentiles.
  - **Transformation**: Apply logarithmic or square root transformations to reduce variability.
  - **Removal**: Eliminate outliers that are proven to be data errors.
  - **Domain Expertise**: Collaborate with subject matter experts to validate and handle outliers appropriately.

### 5.3 Tools and Technologies

- **Programming Languages**: Python, R
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Statistical Techniques**: Z-score, IQR, Missing Data Heatmaps

### 5.4 Implementation Steps

1. **Load the Dataset**: Import the raw data into the environment.
2. **Identify Anomalies**: Use statistical and visualization techniques to detect null values and outliers.
3. **Apply Treatments**: Implement appropriate cleaning methods based on the dataset and research objectives.
4. **Validate Results**: Verify that the cleaned dataset is complete and free of anomalies.
5. **Document the Process**: Maintain a log of cleaning steps for reproducibility.

## 6. CONCLUSION

This project addresses critical issues in quantitative data analysis by providing a structured approach to null value and outlier detection and treatment. By employing automated techniques and leveraging programming tools, researchers can ensure their datasets are clean, accurate, and ready for analysis, ultimately leading to more reliable and impactful research outcomes.

# SECTION A:

## NULL VALUES / COLUMNS DETECTION AND TREATMENT TECHNIQUES

- Loading Dataset
- Finding Frequency of Clean and Null Values
- Visualizing Null Values with Graphs and Tables
- Null Value Treatment Strategies


```python 
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm, zscore
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import plotly.express as px
from sklearn.impute import SimpleImputer
```
### 1. NULL VALUES/COLUMNS DETECTION AND TREATMENT
```Load dataset```
```python 
df = pd.read_csv('datasets2.csv')
df.tail()
```
| ID   | firstname | age   | resident       |
|------|-----------|-------|----------------|
| 995  | Adama     | NaN   | Unguja North   |
| 996  | Kwame     | 45.0  | Morogoro       |
| 997  | Ayaan     | 150.0 | Pemba North    |
| 998  | Salif     | 0.0   | Iringa         |
| 999  | Adama     | 90.0  | Kagera         |


**1.2 FREQUENCY / PERCENT OF CLEAN AND NULL VALUES**

_Calculate null counts and clean data counts_

```python
null_counts = df['age'].isnull().sum()
clean_data_counts = df['age'].notnull().sum()
```
_Calculate percentages_
```python
total_count = len(df)
percent_null = (null_counts / total_count) * 100
percent_clean = (clean_data_counts / total_count) * 100
```
_Create a DataFrame for visualization_

```python
data = {
    'Type': ['Null', 'Clean'],
    'Count': [null_counts, clean_data_counts],
    'Percentage': [percent_null, percent_clean]
}
df_visualize = pd.DataFrame(data)
print(df_visualize)
```

| Type  | Count | Percentage |
|-------|-------|------------|
| Null  | 42    | 4.2        |
| Clean | 958   | 95.8       |


_create bar graph to visualize null_

```python
# Create a bar plot figure
fig = px.bar(df_visualize, x='Type', y='Count', text='Percentage',
             labels={'Type': 'Data Type', 'Count': 'Count', 'Percentage': 'Percentage (%)'},
             title='Frequency and Percentage of Null and Clean Data in Age Column',
             color='Type',
             hover_data={'Percentage': ':.2f%'})
​
# Update layout for customization
fig.update_layout(
    xaxis_title='',
    yaxis_title='Count',
    plot_bgcolor='rgba(0,0,0,0)',  # Remove background color
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='skyblue'),  # Add x-axis gridlines
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='skyblue'),  # Add y-axis gridlines
)
fig.show()
```
![1](https://github.com/user-attachments/assets/ca6fab34-db28-4022-ac53-195b5078697e)

_visualize null field ( age ) by table_

```python
# Before imputation (if there were missing values)
print(df[df['age'].isnull()])
```
| firstname | age   | resident                 |
|-----------|-------|--------------------------|
| Ayaan     | NaN   | Unguja North             |
| Chidinma  | NaN   | Kagera                   |
| Femi      | NaN   | Tabora                   |
| Jelani    | NaN   | Zanzibar Urban/West      |
| Chidinma  | NaN   | Kigoma                   |
| Odera     | NaN   | Ruvuma                   |
| Femi      | NaN   | Singida                  |
| Adama     | NaN   | Unguja North             |
| Penda     | NaN   | Kagera                   |
| Nia       | NaN   | Geita                    |
| Achieng   | NaN   | Zanzibar Central/South   |
| Salif     | NaN   | Songwe                   |

_null values treatment strategies_
```python
# Apply imputation
imputer = SimpleImputer(strategy='mean')

#fit: calculate the mean  mean =41.89  median =42.0   mode=33.0
#transform: Replaces the missing values with the calculated mean
df['age'] = imputer.fit_transform(df[['age']])

# After imputation (check if there are still any missing values)
print("After imputation:")
print(df[df['age'].isnull()])

# NB: The strategy can also be 'median', 'most_frequent', or 'constant' (with a user-specified value).
```
_print the dataframe after treatment_

```python
df.head()
```
| firstname | age       | resident     |
|-----------|-----------|--------------|
| Mosi      | 36.000000 | Dodoma       |
| Ayaan     | 41.890397 | Unguja North |
| Udo       | 21.000000 | Njombe       |
| Zola      | 57.000000 | Singida      |
| Chidinma  | 54.000000 | Mtwara       |

```here the firstname 'Ayaan` was null, now is replaced by mean average 41.89```
| Ayaan     | 41.890397 | Unguja North |

_other  treatment techniques_

```python
# Mean imputation
df['age'].fillna(df['age'].mean(), inplace=True)

# Median imputation
df['age'].fillna(df['age'].median(), inplace=True)

# Mode imputation (for categorical numeric data)
df['age'].fillna(df['age'].mode()[0], inplace=True)

```

_Visualize Null Fields after Treatment_
```python
null_counts = df.isnull().sum()
# Create bar plot figure
fig = px.bar(
    x=null_counts.index,
    y=null_counts.values,
    labels={'x': 'Columns', 'y': 'Null Count'},
    title='Count of Null Values by Column',
)
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',)
# Add gridlines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='green')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='green')
fig.show()
```
![2](https://github.com/user-attachments/assets/9449461a-4118-494a-8553-a9f863e81bcf)

# SECTION B

## OUTLIERS DETECTION AND TREATMENT TECHNIQUES
- Isolation Forest (Contamination)
- Local Outlier Factor (LOF)
- Normal Distributions and Z-scores (X ~ N(40,1))
- Quartiles
- Percentile-Based Method
- Outliers Treatment Techniques (Winsorization)

### OUTLIERS DETECTION TECHNIQUES

_Detection by Normal Distribution & Z scores_

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm, zscore

max_age = df['age'].max()
min_age = df['age'].min()
mean_age = df['age'].mean()
std_age = df['age'].std()

# Create figure for PDF
fig_pdf = go.Figure()

# Plot Normal distribution curve
x_values = np.linspace(min_age, max_age, 100)
normal_pdf = norm.pdf(x_values, mean_age, std_age)
fig_pdf.add_trace(go.Scatter(x=x_values, y=normal_pdf, mode='lines', name='Normal Distribution', line=dict(color='blue')))

# Highlight outliers using z-scores
z_scores = zscore(df['age'])
outliers = df[(z_scores < -3) | (z_scores > 3)]
fig_pdf.add_trace(go.Scatter(x=outliers['age'], y=[0]*len(outliers), mode='markers', name='Outliers', marker=dict(color='red', size=10)))

# Add vertical lines for mean and outlier thresholds
fig_pdf.add_vline(x=mean_age, line_dash="dash", line_color="red", name='Mean')
fig_pdf.add_vline(x=mean_age - 3*std_age, line_dash="dot", line_color="blue", name='Lower Threshold')
fig_pdf.add_vline(x=mean_age + 3*std_age, line_dash="dot", line_color="blue", name='Upper Threshold')

# Highlight typical range with a shaded rectangle
fig_pdf.add_vrect(
    x0=mean_age - 3*std_age, 
    x1=mean_age + 3*std_age, 
    fillcolor='rgba(0,100,80,0.2)', 
    line_width=0, 
    opacity=0.5,
    annotation_text='Typical Range'
)

# Add annotations for mean, thresholds, and any important points
fig_pdf.add_annotation(
    x=mean_age, 
    y=max(normal_pdf), 
    text="Mean", 
    showarrow=True, 
    arrowhead=1, 
    xshift=10
)
fig_pdf.add_annotation(
    x=mean_age - 3*std_age, 
    y=0.02, 
    text="Outlier Threshold (-3)", 
    showarrow=True, 
    arrowhead=1, 
    xshift=-10
)
fig_pdf.add_annotation(
    x=mean_age + 3*std_age, 
    y=0.02, 
    text="Outlier Threshold (3)", 
    showarrow=True, 
    arrowhead=1, 
    xshift=10
)

# Update layout for aesthetics
fig_pdf.update_layout(
    title='Probability Density Function (PDF) of Age',
    xaxis_title='Age',
    yaxis_title='Density',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='skyblue'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='skyblue')
)

fig_pdf.show()
```
![3](https://github.com/user-attachments/assets/4737ffa7-27bb-44c6-bc2f-b52523ba3d40)

```above figure  we have two outliers that are out or +3 Threshold to the right,  we have extreem two positive values out of range```

_Detection by Box plot graph_
```python
# Box Plot of Age
fig_box = go.Figure()
fig_box.add_trace(go.Box(y=df['age'], boxpoints='outliers', marker_color='blue', name='Age Distribution'))

# Add quartile annotations
quartiles = np.percentile(df['age'], [25, 50, 75])
quartile_text = [f"Q{i}: {quartile:.2f}" for i, quartile in enumerate(quartiles, start=1)]
fig_box.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color='white', size=0), name=', '.join(quartile_text)))
fig_box.update_layout(
    title='Box Plot of Age',
    xaxis_title='Age',
    yaxis_title='',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',  # Remove background color
     xaxis=dict(showgrid=True, gridwidth=1, gridcolor='skyblue'),  # Add x-axis gridlines
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='skyblue'),  # Add y-axis gridlines
    height=400,  # Adjust plot height
)
fig_box.show()
```
![4](https://github.com/user-attachments/assets/cf6854d2-2382-4f36-8443-9d5d8cf2dc7b)

_Detection by Isolation Forest_

```python
# Fit the Isolation Forest model
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(df[['age']])
df['anomaly'] = iso_forest.predict(df[['age']])

# Identify outliers
iso_outliers = df[df['anomaly'] == -1]

print(iso_outliers)
```
| firstname | age  | resident    | anomaly |
|-----------|------|-------------|---------|
| Ayaan     | 150.0 | Pemba North | -1      |
| Salif     | 0.0   | Iringa      | -1      |
| Adama     | 90.0  | Kagera      | -1      |


__visualize Isolation Forest by Scatter Plot_

```python
scatter_fig = px.scatter(df, x=df.index, y='age', color='anomaly',
                         color_discrete_map={1: 'blue', -1: 'red'},
                         labels={'color': 'Anomaly'})

scatter_fig.update_layout(
    title='Scatter Plot of Age with Anomalies Highlighted',
    xaxis_title='Index',
    yaxis_title='Age',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',  # Remove background color
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='skyblue'),  # Add x-axis gridlines
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='skyblue'),  # Add y-axis gridlines
)

scatter_fig.show()
```
![5](https://github.com/user-attachments/assets/894029c4-e5cc-4605-b5b1-2c07fa4559b4)


_visualize Isolation Forest by Box Plot_

```python
box_fig = px.box(df, x='anomaly', y='age', color='anomaly',
                 color_discrete_map={1: 'blue', -1: 'red'},
                 labels={'anomaly': 'Anomaly', 'age': 'Age'})

box_fig.update_layout(
    title='Box Plot of Age with Anomalies Highlighted',
    xaxis_title='Anomaly',
    yaxis_title='Age',
    showlegend=True,
    plot_bgcolor='rgba(0,0,0,0)',  # Remove background color
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='skyblue'),  # Add x-axis gridlines
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='skyblue'),  # Add y-axis gridlines
)
box_fig.show()
```






