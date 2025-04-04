import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('data.csv')

plt.scatter(data.studytime, data.score)
plt.xlabel('Study Time')
plt.ylabel('Score')
plt.show()
