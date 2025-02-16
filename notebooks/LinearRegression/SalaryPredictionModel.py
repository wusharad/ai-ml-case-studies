# Simple Linear Regression to predict salary on the basis of year of experience

# Importing the libraries
import os
import logging as log
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Get the base project directory (parent of 'notebooks' directory)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Ensure 'log' directory exists
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
# Define logging file path inside 'log' directory
LOG_FILE_PATH = os.path.join(LOG_DIR, 'salary_prediction_model.log')
# Define dataset file path inside 'data' directory
DATASET_FILE_PATH = os.path.join(DATA_DIR, 'salary_dataset.csv')

# Configure logging
log.basicConfig(
    filename=LOG_FILE_PATH,
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

log.info("Starting prediction process...")

# Importing the dataset
dataset = pd.read_csv(DATASET_FILE_PATH)
X = dataset.iloc[:, :-1].values # input
y = dataset.iloc[:, 1].values # output

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)

# Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)

# Fitting Simple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')


# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()