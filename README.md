# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
## Date:06.04.2024


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the AirLinePassengers data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
#### Import the packages
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```
#### Read the Airline Passengers dataset from a CSV file
```
data = pd.read_csv("/content/airline.csv")
```
#### Display the shape and the first 50 rows of the dataset
```
print("Shape of the dataset:", data.shape)
print("First 50 rows of the dataset:")
print(data.head(50))
```
#### Plot the first 50 values of the 'International' column
```
plt.plot(data['International '].head(50))
plt.title('First 50 values of the "International" column')
plt.xlabel('Index')
plt.ylabel('International Passengers')
plt.show()
```
#### Perform rolling average transformation with a window size of 5
```
rolling_mean_5 = data['International '].rolling(window=5).mean()
```
#### Display the first 10 values of the rolling mean
```
print("First 10 values of the rolling mean with window size 5:")
print(rolling_mean_5.head(10))
```
#### Perform rolling average transformation with a window size of 10
```
rolling_mean_10 = data['International '].rolling(window=10).mean()
```
#### Plot the original data and fitted value (rolling mean with window size 10)
```
plt.plot(data['International '], label='Original Data')
plt.plot(rolling_mean_10, label='Rolling Mean (window=10)')
plt.title('Original Data and Fitted Value (Rolling Mean)')
plt.xlabel('Index')
plt.ylabel('International Passengers')
plt.legend()
plt.show()
```
#### Fit an AutoRegressive (AR) model with 13 lags
```
lag_order = 13
model = AutoReg(data['International '], lags=lag_order)
model_fit = model.fit()
```
#### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
```
plot_acf(data['International '])
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(data['International '])
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```
#### Make predictions using the AR model
```
predictions = model_fit.predict(start=lag_order, end=len(data)-1)
```
#### Compare the predictions with the original data
```
mse = mean_squared_error(data['International '][lag_order:], predictions)
print('Mean Squared Error (MSE):', mse)
```
#### Plot the original data and predictions
```
plt.plot(data['International '][lag_order:], label='Original Data')
plt.plot(predictions, label='Predictions')
plt.title('AR Model Predictions vs Original Data')
plt.xlabel('Index')
plt.ylabel('International Passengers')
plt.legend()
plt.show()
```
### OUTPUT:
#### Plot the original data and fitted value
![image](https://github.com/manojvenaram/TSA_EXP8/assets/94165064/b4f4a99c-80df-42d6-9a32-75cd36d347c6)
#### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
![image](https://github.com/manojvenaram/TSA_EXP8/assets/94165064/60862b92-0bbd-428e-85c1-c7069a7d848f)
![image](https://github.com/manojvenaram/TSA_EXP8/assets/94165064/daea0877-a99a-491a-96ae-270f4b47f6eb)
#### Plot the original data and predictions
![image](https://github.com/manojvenaram/TSA_EXP8/assets/94165064/5cf8d0f9-13bc-49dd-af9f-32eafcd0f0a0)




### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
