import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv(r'D:\vscode\nyc_building_energy_efficiency_ratings\nyc_energy_rating\data\raw\energy_disclosure_2021_rows.csv')
df2 = pd.read_csv(r'D:\vscode\nyc_building_energy_efficiency_ratings\nyc_energy_rating\data\raw\geojson_lookup_rows.csv')
df.head()

# Merge datasets based on a common key
merged_data = pd.merge(df, df2, how='left', on='10_Digit_BBL')

filepath = os.getcwd()
def distribution_of_energy_efficiency(features1):
    # Distribution of Energy Efficiency Grades
    plt.figure(figsize=(10, 6))
    fig_1 = sns.countplot(x=features1, data=merged_data, palette='Set2')
    plt.title('Distribution of Energy Efficiency Grades')
    plt.xlabel('Energy Efficiency Grade')
    plt.ylabel('Count')
    plt.savefig(filepath)
    plt.close()

distribution_of_energy_efficiency(merged_data['Energy_Efficiency_Grade'])

def distribution_of_energy_star_score(features2):
    # Distribution of ENERGY STAR Scores
    plt.figure(figsize=(10, 6))
    fig_2 = sns.histplot(features2, bins=20, kde=True, color='skyblue')
    plt.title('Distribution of ENERGY STAR Scores')
    plt.xlabel('ENERGY STAR Score')
    plt.ylabel('Count')
    plt.savefig(filepath)
    plt.close()
distribution_of_energy_star_score(merged_data['Energy_Star_1-100_Score'])


def box_plot_energy_efficiency(features1,features2):
    # Box plot of Energy Efficiency Grade vs. DOF Gross Square Footage
    plt.figure(figsize=(10, 6))
    fig_3 = sns.boxplot(x=features1, y=features2, data=merged_data, palette='Set2')
    plt.title('Energy Efficiency Grade vs. DOF Gross Square Footage')
    plt.xlabel('Energy Efficiency Grade')
    plt.ylabel('DOF Gross Square Footage')
    plt.savefig(filepath)
    plt.close()
box_plot_energy_efficiency(merged_data['Energy_Efficiency_Grade'],merged_data['DOF_Gross_Square_Footage'])

def energy_efficiency_grade_star(feature1,feature2):
    # Box plot of Energy Efficiency Grade vs. ENERGY STAR Score
    plt.figure(figsize=(10, 6))
    fig_4 = sns.boxplot(x=feature1, y=feature2, data=merged_data, palette='Set2')
    plt.title('Energy Efficiency Grade vs. ENERGY STAR Score')
    plt.xlabel('Energy Efficiency Grade')
    plt.ylabel('ENERGY STAR Score')
    plt.savefig(filepath)
    plt.close()
energy_efficiency_grade_star(merged_data['Energy_Efficiency_Grade'],merged_data['Energy_Star_1-100_Score'])


def energy_efficiency_grade(feature1):
    # Pie chart of Energy Efficiency Grades
    grade_counts = merged_data[feature1].value_counts()
    plt.figure(figsize=(8, 8))
    fig_5 = plt.pie(grade_counts, labels=grade_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Energy Efficiency Grades')
    plt.axis('equal')
    plt.savefig(filepath)
    plt.close()
energy_efficiency_grade(merged_data['Energy_Efficiency_Grade'])

# Correlation between numeric features
numeric_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = merged_data[numeric_columns].corr()

plt.figure(figsize=(8, 6))
fig_6 = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig(filepath)
plt.close()