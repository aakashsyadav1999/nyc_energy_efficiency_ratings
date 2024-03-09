import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv(r'D:\vscode\nyc_building_energy_efficiency_ratings\nyc_energy_rating\data\raw\energy_disclosure_2021_rows.csv')
df2 = pd.read_csv(r'D:\vscode\nyc_building_energy_efficiency_ratings\nyc_energy_rating\data\raw\geojson_lookup_rows.csv')
df.head()

filepath = os.getcwd()

def merge_data(dataframe1,dataframe2):
    # Merge datasets based on a common key
    merged_data = pd.merge(dataframe1, dataframe2, how='left', on='10_Digit_BBL')
    merged_data.to_csv("merged_data")
merge_data(df,df2)

def load_dataset(filename):
    merged_data = pd.read_csv(r'D:\vscode\nyc_building_energy_efficiency_ratings\nyc_energy_rating\src\visualization\merged_data')
    return merge_data
    logging("Loading Merged Data")
load_dataset(merge_data)

merged_data = pd.read_csv(r'D:\vscode\nyc_building_energy_efficiency_ratings\nyc_energy_rating\src\visualization\merged_data')

def distribution_of_energy_efficiency(dataframe,features1):
    # Distribution of Energy Efficiency Grades
    plt.figure(figsize=(10, 6))
    fig_1 = sns.countplot(x=features1, data=dataframe, palette='Set2')
    plt.title('Distribution of Energy Efficiency Grades')
    plt.xlabel('Energy Efficiency Grade')
    plt.ylabel('Count')
    plt.savefig("Distribution of Energy Efficiency Grades")
    plt.close()
distribution_of_energy_efficiency(merged_data,merged_data['Energy_Efficiency_Grade'])

def distribution_of_energy_star_score(features2):
    # Distribution of ENERGY STAR Scores
    plt.figure(figsize=(10, 6))
    fig_2 = sns.histplot(features2, bins=20, kde=True, color='skyblue')
    plt.title('Distribution of ENERGY STAR Scores')
    plt.xlabel('ENERGY STAR Score')
    plt.ylabel('Count')
    plt.savefig("Distribution of ENERGY STAR Scores")
    plt.close()
distribution_of_energy_star_score(merged_data['Energy_Star_1-100_Score'])


def box_plot_energy_efficiency(dataframe,features1,features2):
    # Box plot of Energy Efficiency Grade vs. DOF Gross Square Footage
    plt.figure(figsize=(10, 6))
    fig_3 = sns.boxplot(x=features1, y=features2, data=dataframe, palette='Set2')
    plt.title('Energy Efficiency Grade vs. DOF Gross Square Footage')
    plt.xlabel('Energy Efficiency Grade')
    plt.ylabel('DOF Gross Square Footage')
    plt.savefig('Energy Efficiency Grade vs Gross Square Footage')
    plt.close()
box_plot_energy_efficiency(merged_data,merged_data['Energy_Efficiency_Grade'],merged_data['DOF_Gross_Square_Footage'])

def energy_efficiency_grade_star(dataframe,feature1,feature2):
    # Box plot of Energy Efficiency Grade vs. ENERGY STAR Score
    plt.figure(figsize=(10, 6))
    fig_4 = sns.boxplot(x=feature1, y=feature2, data=dataframe, palette='Set2')
    plt.title('Energy Efficiency Grade vs. ENERGY STAR Score')
    plt.xlabel('Energy Efficiency Grade')
    plt.ylabel('ENERGY STAR Score')
    plt.savefig('Energy Efficiency Grade vs ENERGY STAR Score')
    plt.close()
energy_efficiency_grade_star(merged_data,merged_data['Energy_Efficiency_Grade'],merged_data['Energy_Star_1-100_Score'])


def energy_efficiency_grade(dataframe,feature1):
    # Pie chart of Energy Efficiency Grades
    grade_counts = dataframe[feature1].value_counts()
    plt.figure(figsize=(8, 8))
    fig_5 = plt.pie(grade_counts, labels=grade_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Distribution of Energy Efficiency Grades')
    plt.axis('equal')
    plt.savefig('Distribution of Energy Efficiency Grades')
    plt.close()
energy_efficiency_grade(merged_data,'Energy_Efficiency_Grade')

def conver_to_numeric(dataframe):
    # Correlation between numeric features
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = dataframe[numeric_columns].corr()
    correlation_matrix.to_csv("correlation_matrix")
conver_to_numeric(merged_data)

correlation_matrix = pd.read_csv(r'D:\vscode\nyc_building_energy_efficiency_ratings\nyc_energy_rating\src\visualization\correlation_matrix',usecols=['10_Digit_BBL','Street_Number','DOF_Gross_Square_Footage','Energy_Star_1-100_Score','Latitude','Longitude'])

def correlations_matirx(dataframe):
    plt.figure(figsize=(8, 6))
    fig_6 = sns.heatmap(dataframe, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('Correlation Matrix')
    plt.close()
correlations_matirx(correlation_matrix)