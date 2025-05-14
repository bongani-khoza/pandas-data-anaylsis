# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Choose a dataset in CSV format (e.g., Iris dataset)
from sklearn.datasets import load_iris

# Load the dataset using pandas
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = iris.target_names[df['species']].str.replace(" ", "_")

# Display the first few rows of the dataset
print(df.head())

# Explore the structure of the dataset by checking data types and missing values
print(df.info())
print(df.isnull().sum())

# Clean the dataset by filling or dropping any missing values
df['sepal length (cm)'] = df['sepal length (cm)'].fillna(df['sepal length (cm)'].mean())
df['petal width (cm)'] = df['petal width (cm)'].fillna(df['petal width (cm)'].mean())

print(df.head())

# Compute the basic statistics of numerical columns using .describe()
num_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
df_num_stats = df[num_cols].describe()

print(df_num_stats)

# Perform groupings on categorical column and compute mean of numerical column
grouped_df = df.groupby('species_name')[num_cols[0]].mean()
print(grouped_df)

# Create different types of visualizations

# Line chart showing trends over time (e.g., a time-series of sales data)
plt.figure(figsize=(10,6))
sns.lineplot(x='sepal length (cm)', y='petal width (cm)', data=df)
plt.title('Trend in Sepal Length vs Petal Width')
plt.show()

# Bar chart showing the comparison of a numerical value across categories
species_counts = df['species_name'].value_counts()
plt.figure(figsize=(10,6))
sns.barplot(x=species_counts.index, y=species_counts.values)
plt.title('Species Counts')
plt.show()

# Histogram of a numerical column to understand its distribution
plt.figure(figsize=(10,6))
sns.histplot(df[num_cols[0]], kde=True)
plt.title(f'Histogram of {num_cols[0]}')
plt.show()

# Scatter plot to visualize the relationship between two numerical columns
plt.figure(figsize=(10,6))
sns.scatterplot(x='sepal length (cm)', y='petal width (cm)', data=df)
plt.title('Scatter Plot of Sepal Length vs Petal Width')
plt.show()
