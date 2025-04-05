import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('linear-regression/data.csv')

# plt.scatter(data.studytime, data.score)
# plt.xlabel('Study Time')
# plt.ylabel('Score')
# plt.show()

def loss_function(m, b, points):
    """
    Calculates the mean squared error (MSE) for a given set of data points and a linear model.

    Args:
        m (float): The slope of the linear regression line.
        b (float): The y-intercept of the linear regression line.
        points (pandas.DataFrame): A DataFrame containing the data points with columns 'studytime' (independent variable) 
                                   and 'score' (dependent variable).

    Returns:
        float: The mean squared error (MSE) of the linear regression model on the given data points.
    """
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i]['studytime']
        y = points.iloc[i]['score']
        total_error += (y - (m * x + b)) ** 2
    total_error /= len(points)


def gradient_descent(m_now, b_now, points, alpha):
    """
    Performs one iteration of gradient descent to update the slope (m) and intercept (b) 
    of a linear regression model.
    Args:
        m_now (float): Current value of the slope (m).
        b_now (float): Current value of the intercept (b).
        points (pandas.DataFrame): A DataFrame containing the data points with columns 
                                   'studytime' (independent variable) and 'score' (dependent variable).
        alpha (float): The learning rate, which controls the step size of the updates.
    Returns:
        tuple: A tuple containing the updated values of the slope (m_new) and intercept (b_new).
    """
    m_gradient = 0
    b_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points.iloc[i]['studytime']
        y = points.iloc[i]['score']
        m_gradient += (-2 / N) * x * (y - (m_now * x + b_now))
        b_gradient += (-2 / N) * (y - (m_now * x + b_now))
        
    m_new = m_now - (alpha * m_gradient)
    b_new = b_now - (alpha * b_gradient)
    
    return m_new, b_new


m = 0
b = 0
alpha = 0.001
epochs = 300
for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch {i}: m = {m}, b = {b}")
    m, b = gradient_descent(m, b, data, alpha)
    

print(f"m: {m}, b: {b}")

plt.scatter(data.studytime, data.score, color='blue')
plt.plot(data.studytime, m * data.studytime + b, color='red')
plt.xlabel('Study Time')
plt.ylabel('Score')
plt.title('Linear Regression')
plt.show()