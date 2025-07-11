import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pd.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pd.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pd.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pd.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pd.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total'] = pd.to_numeric(dataset_2['Total'].replace(',','', regex=True))
print(dataset_2.to_string()) #This line will print out your data
bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']
y = dataset_2['Total']

r2_vals =[]
for i,b in enumerate(bridges):
    X = dataset_2[[b]]
    model = LinearRegression()
    model.fit(X,y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    r2_vals.append((b,r2))
    print(f"R^2 value for {b}: {r2:.4f}")
    # Scatter plot for X and Y values
    plt.subplot(2, 2, i + 1)  # Create a 2x2 grid of subplots
    plt.scatter(X, y, color='blue', alpha=0.6, label='Actual Data')
    plt.plot(X, y_pred, color='red', label='Regression Line')
    plt.title(f"{b} (R²: {r2:.2f})")
    plt.xlabel(f"{b} Cyclist Count")
    plt.ylabel("Total Cyclists")
    plt.legend()
plt.tight_layout()
plt.show()
#Sorting Bridges based upon R^2 value
sorted_bridges = sorted(r2_vals,key=lambda x: x[1], reverse=True)
#Displaying top three bridges
print("\nTop 3 bridges based on R^2 value:")
for i,(bridge, r2) in enumerate(sorted_bridges[:3],start=1):
    print(f"{i}. {bridge} (R^2: {r2:.4f})")

#Average R^2 value
r2_vals = [r2 for _, r2 in r2_vals]
Sample_mean = np.mean(r2_vals)
mu = 0.25
Sample_std = np.std(r2_vals,ddof=1)
n = len(r2_vals)
Z = (Sample_mean - mu) / (Sample_std / np.sqrt(n))
p_value = (1-norm.cdf(abs(Z))) #one-tailed test
alpha = 0.05
# Null Hypothesis
print("\nHypothesis Test:")
print("H₀: Sample mean ≤ 0.25")
print("H₁: Sample mean > 0.25")
if p_value < alpha:
    print(f"Reject the null hypothesis (p-value: {p_value:.4f})")
else:
    print(f"Fail to reject the null hypothesis (p-value: {p_value:.4f})")
#Null Hypothesis

############################################################


#Question 2: Predciting cyclists based upon the weather data
weather_feat = ["High Temp", "Low Temp", "Precipitation"]
X_weather = dataset_2[weather_feat]
Y_weather = dataset_2['Total']
#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_weather, Y_weather, test_size=0.2, random_state=42)

weather_model = LinearRegression()
weather_model.fit(X_train, y_train)
Y_weather_pred = weather_model.predict(X_test)
r2_weather = r2_score(y_test, Y_weather_pred)
mse_weather = mean_squared_error(y_test, Y_weather_pred)

print(f"\nMean Squared Error for weather data: {mse_weather:.4f}")
print(f"R^2 value for weather data: {r2_weather:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, Y_weather_pred, color='blue', alpha=0.6, label='Predicted vs Actual Data')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Fit')
plt.title("Predicting total cyclists based on weather data")
plt.xlabel("Actual Total Cyclists")
plt.ylabel("Predicted Total Cyclists")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


################################################################


#Question 3: Analysing a pattern in the data
#Analyse the weekly pattern in the data

# Add the year to the Date column and convert to datetime
dataset_2['Date'] = dataset_2['Date'] + '-2016'
dataset_2['Date'] = pd.to_datetime(dataset_2['Date'], format='%d-%b-%Y', errors='coerce')
# Check for any NaT values in the Date column
dataset_2.dropna(subset=['Date'], inplace=True)
dataset_2['Day'] = dataset_2['Date'].dt.day_name()
# Check the unique values in the 'Day' column
weekly_avg = dataset_2.groupby('Day')['Total'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

#visualize the weekly pattern
plt.figure(figsize=(10, 6))
weekly_avg.plot(kind='bar', color='skyblue')
plt.title("Average Cyclists per Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Average Cyclists")  
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

day_map = {day:i for i, day in enumerate(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])}
dataset_2['Day Numeric'] = dataset_2['Day'].map(day_map)

X_day = dataset_2[bridges]
y_day = dataset_2['Day Numeric']

#Splitting the data into training and testing sets
X_train_day, X_test_day, y_train_day, y_test_day = train_test_split(X_day, y_day, test_size=0.2, random_state=42)

#Random Forest Classifier
day_model = RandomForestClassifier(random_state=42)
day_model.fit(X_train_day, y_train_day)
y_pred_day = day_model.predict(X_test_day)
accuracy = accuracy_score(y_test_day, y_pred_day)
print(f"\nAccuracy of Random Forest model for predicting day of the week: {accuracy:.4f}")