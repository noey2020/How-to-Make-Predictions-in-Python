# How-to-Make-Predictions-in-Python

october 8, 2018

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me!

Example 1-1 shows the Python code that loads the data, prepares it, creates a
scatterplot for visualization, and then trains a linear model and makes a 
prediction.

Training and running a linear model using Scikit-Learn.

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

# Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands=',',delimiter='\t',
                             encoding='latin1', na_values="n/a")

# Prepare the data
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Visualize the data
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]

##############
If you had used an instance-based learning algorithm instead, you would have
found that Slovenia has the closest GDP per capita to that of Cyprus ($ 20,732),
and since the OECD data tells us that Slovenians’ life satisfaction is 5.7, you 
would have predicted a life satisfaction of 5.7 for Cyprus. If you zoom out a
bit and look at the two next closest countries, you will find Portugal and 
Spain with life satisfactions of 5.1 and 6.5, respectively. Averaging these 
three values, you get 5.77, which is pretty close to your model-based prediction. 
This simple algorithm is called k-Nearest Neighbors regression (in this example,
k = 3). Replacing the Linear Regression model with k-Nearest Neighbors regression
in the previous code is as simple as replacing this line: 

model = sklearn.linear_model.LinearRegression() 

with this one: 

model = sklearn.neighbors.KNeighborsRegressor( n_neighbors = 3)

##############
If all went well, your model will make good predictions. If not, you may need to
use more attributes (employment rate, health, air pollution, etc.), get more or
better quality training data, or perhaps select a more powerful model (e.g., a
Polynomial Regression model). In summary: You studied the data. You selected a
model. You trained it on the training data (i.e., the learning algorithm searched
for the model parameter values that minimize a cost function). Finally, you
applied the model to make predictions on new cases (this is called inference), 
hoping that this model will generalize well. This is what a typical Machine 
Learning project looks like.

##############
Concretize the Problem

I would start with asking the first question which is what exactly is the business 
objective; are we building a model? How do we expect to use and benefit from this 
model? This is important because it will determine how we will frame the problem,
what algorithms we will select, what performance measure we will use to evaluate
the model, and how much effort I will spend tweaking it.

We need to investigate what feature/s are relevant to include in our matrix. Single
feature means using univariate whereas multiple will be multivariate regression. 

The output of our model might be a classifier, prediction or output to another
module. Is it worth investing in the effort or will it affect revenues.. and so forth

##############
october 11, 2018

https://www.deeplearningbook.org/

I included some posts for reference.

https://github.com/noey2020/How-to-Talk-Hashing

https://github.com/noey2020/How-to-Talk-Algorithm-Analysis

https://github.com/noey2020/How-to-Talk-Recursion

https://github.com/noey2020/How-to-Talk-Linked-Lists

https://github.com/noey2020/How-to-Talk-Queues

https://github.com/noey2020/How-to-Talk-Stacks

https://github.com/noey2020/How-to-Talk-Lists-Stacks-and-Queues

https://github.com/noey2020/How-to-Talk-Linear-Regression

https://github.com/noey2020/How-to-Talk-Statistics-Pattern-Recognition-101

https://github.com/noey2020/How-to-Write-SPI-STM32

https://github.com/noey2020/How-to-Write-SysTick-Handler-for-STM32

https://github.com/noey2020/How-to-Write-Subroutines-in-C-Assembly-STM32

https://github.com/noey2020/How-to-Write-Multitasking-STM32

https://github.com/noey2020/How-to-Generate-Triangular-Wave-STM32-DAC

https://github.com/noey2020/How-to-Generate-Sine-Table-LUT

https://github.com/noey2020/How-to-Illustrate-Multiple-Exceptions-

https://github.com/noey2020/How-to-Blink-LED-using-Standard-Peripheral-Library

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me!

Have fun and happy coding!
