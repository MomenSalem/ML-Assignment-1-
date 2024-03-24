# Name: Momen Salem
# ID: 1200034
# ML Assignment #1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # library for printing boxplot
import numpy as np
import random


def missing_value(null_data, data):
    null_feature = {}  # initiate the empty dictionary for null features (missing values)
    for i in null_data:  # loop for each feature
        for j in range(len(null_data[i])):  # loop for each example in each feature
            if null_data[i][j]:  # check if the feature is null (has true boolean value in the matrix)
                if data[i].dtype == "object":  # check if the type of data is not number
                    data[i].fillna(data[i].mode()[0], inplace=True)  # if the data not number find the mode of data
                else:
                    data[i].fillna(data[i].mean(), inplace=True)  # the data is number and we can find the mean
                if i in null_feature.keys():  # check if the feature exist in the dictionary and increment the number
                    # of missing value
                    null_feature[i] = null_feature[i] + 1
                else:  # else the feature is not in the dictionary so add it as a new key with one missing value
                    null_feature[i] = 1
    for i in null_feature:
        print("The feature", i, "has =", null_feature[i], "missing values")
    return null_feature


def skewnees(features_name, data_frame):
    S = 0
    for i in range(len(features_name)):
        mean = data_frame[features_name[i]].mean()
        median = data_frame[features_name[i]].median()
        sd = data_frame[features_name[i]].std()
        S = 3 * (mean - median) / sd
        print("The skewness of the feature (", features_name[i], ") is = ", S)


def function_y(para, matrix):
    y = []
    for i in range(len(matrix)):
        y.append(para[0] * matrix[i][0] + para[1] * matrix[i][1])
    return y


def SLR(parameters, x):  # simple linear regression function to print the line
    y = []
    for i in range(len(x)):
        y.append(parameters[0] + parameters[1] * x[i])
    return y


def NonLR(parameters, x):  # non-linear regression function using monomials (polynomial)
    y = []
    for i in range(len(x)):
        y.append(round(parameters[0] + parameters[1] * x[i] + parameters[2] * (x[i] ** 2), 2))
    return y


def Gradient_Descent(initial_parameters, x, y_actual):  # implement gradient descent function using iterations approach
    n = 0
    alpha = 0.05
    while n != 350:
        y_predicted = function_y(initial_parameters, x)  # calculate the function value by substitute
        # the parameters with data
        error = Error(y_actual, y_predicted)  # calculate the error using function
        g0 = 0  # initiate the first gradient value
        for i in range(len(x)):
            g0 += error[i] * x[i][
                0]  # update the gradiant value as multiplication between error and the first feature value
        g0 *= 2 / len(y_actual)  # multiply the gradient with (2/n) to have the min value of derivative of the mean
        # squared error (MSE)
        g1 = 0
        for j in range(len(x)):
            g1 += error[j] * x[j][1]
        g1 *= 2 / len(y_actual)
        grad = [g0, g1]
        for k in range(len(initial_parameters)):
            initial_parameters[k] = initial_parameters[k] - alpha * grad[k]
        n = n + 1
    return initial_parameters  # return the updated parameters which can be used as the final parameter


def Error(y_actual, y_predict):  # function to calculate the error by subtracting each expected value with its
    # predicted one
    error = []
    for m in range(len(y_actual)):
        error.append((y_predict[m] - y_actual[m]))
    return error


# question #1
print("----------------------------Question #1-----------------------------")
data = pd.read_csv('cars.csv')
print(data)  # print the data set (first 5 and last 5 rows)
print(data.info())  # print the number of columns and rows and information for dataset

# question #2 + #3
print("----------------------------Question #2 + #3-----------------------------")
null_data = data.isnull()  # create null data frame which is boolean table indicating if there is a value or not
print(null_data)  # dataframe containing true (is null) or false (is not null)
# (true mean the value is missing)
null_feature = missing_value(null_data, data)
print(data.info())  # print the data after filling the missing values

# question #4
data_grouped = data.groupby(data['origin'])['mpg']
plt.figure(figsize=(11, 6))
sns.boxplot(x='origin', y='mpg', data=data, color='c')
plt.title('Fuel Economy Comparison by Origin')
plt.xlabel('Country')
plt.ylabel('Miles per Gallon (mpg)')
plt.show()

# question #5
plt.figure(figsize=(13, 6))
plt.subplot(1, 3, 1)
sns.histplot(data['acceleration'], kde=True, color='r')
plt.title('Acceleration')

plt.subplot(1, 3, 2)
sns.histplot(data['horsepower'], kde=True)
plt.title('Horse Power')

plt.subplot(1, 3, 3)
sns.histplot(data['mpg'], kde=True, color='g')
plt.title('Mile Per Gallon')
plt.show()

# question #6
print("----------------------------Question #6-----------------------------")
features = ['acceleration', 'horsepower', 'mpg']
skewnees(features, data)

# question #7
print("----------------------------Question #7-----------------------------")
print(data[['horsepower', 'mpg']].corr())  # Correlation for two features
data.plot(kind='scatter', x='horsepower', y='mpg', label='Examples')
plt.show()

# question #8
print("----------------------------Question #8-----------------------------")
data.insert(loc=0, column='x0', value=1)  # adding the intercept feature
print(data[['x0', 'horsepower', 'mpg']])
matrix_X = data[['x0', 'horsepower']].to_numpy()
label = data['mpg'].to_numpy()
matrix_X_transpose = matrix_X.transpose()
inverse_matrix = np.linalg.inv(matrix_X_transpose.dot(matrix_X))
parameters = inverse_matrix.dot(matrix_X_transpose).dot(label)
print("w0 =", parameters[0], " & w1 =", parameters[1])
data.plot(kind='scatter', x='horsepower', y='mpg', label='Examples')
X = data['horsepower'].tolist()
Y = SLR(list(parameters), X)
plt.plot(X, Y, c='r', label='Linear Model')
plt.legend()
plt.title("Linear Model")
plt.show()

# question #9
print("----------------------------Question #9-----------------------------")
x_squared = []  # lest to square the value of x (horsepower =)
for i in range(len(data['horsepower'])):
    x_squared.append(data['horsepower'].tolist()[i] ** 2)
data['x^2'] = x_squared  # add the new column to my data frame
print(data[['x0', 'horsepower', 'x^2', 'mpg']])  # print the data after adding the x squared feature
# nonL stands for non-linear relation
matrix_X_nonL = data[['x0', 'horsepower', 'x^2']].to_numpy()  # same way as question #8 but now with non-linear relation
label_nonL = data['mpg']  # the label or actual output is saved in variable
matrix_X_transpose_nonL = matrix_X_nonL.transpose()  # calculate the transpose of matrix
inverse_matrix_nonL = np.linalg.inv(matrix_X_transpose_nonL.dot(matrix_X_nonL))  # calculate the invers of dot matrices
parameters_nonL = inverse_matrix_nonL.dot(matrix_X_transpose_nonL).dot(
    label_nonL)  # dot the invers result with label to obtain the parameters
print(list(parameters_nonL))  # print the parameters to see that they differ from liner regression

# print the result of non-linear model put i must sort the data to print just one line

data.plot(kind='scatter', x='horsepower', y='mpg', label='Examples')
X_NL = data['horsepower'].tolist()
X_NL.sort()  # must sort the numbers before find its model value
Y_NL = NonLR(list(parameters_nonL), X_NL)
plt.plot(X_NL, Y_NL, c='r', label='Model')
plt.legend()
plt.title("Non-Linear Model")
plt.show()

# question #10
print("----------------------------Question #10-----------------------------")
w0 = random.random()  # initiate the first parameter random number using the random library
w1 = random.random()
initial_parameters = [w0, w1]
print("The initial parameters are = ", w0, w1)
# first I want to normalize the data using z-score approach to converge fast to solution
X_mean = data['horsepower'].mean()
X_std = data['horsepower'].std()
data['horsepower_normalized'] = (data['horsepower'] - X_mean) / X_std  # add the normalized feature to data
matrix = data[['x0', 'horsepower_normalized']].to_numpy()  # save the matrix to be used in gradiant descent function
updated_parameters = Gradient_Descent(initial_parameters, matrix, label)
print("The updated parameters are = ", updated_parameters)
data.plot(kind='scatter', x='horsepower', y='mpg', label="Examples")
Y_grad = SLR(updated_parameters, data['horsepower_normalized'])  # find the function value using the updated parameters
plt.plot(data['horsepower'], Y_grad, c='r', label="Gradient Descent Model")
plt.legend()
plt.title('Model using Gradient Descent')
plt.show()

# print both lines to see if they identical to each other or no (question8 and question10 on same plot)
data.plot(kind='scatter', x='horsepower', y='mpg', label='Examples')
plt.plot(X, Y, c='r', linewidth=4, label='Linear Model')  # change the line width to see if the two lines lies on each
# other
plt.plot(data['horsepower'], Y_grad, c='b', label="Gradient Descent Model")
plt.legend()
plt.title("Both Linear Model using closed form and gradient descent")
plt.show()
