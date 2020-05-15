"""
Problem definition:
house Area: A=[70, 80, 90, 100, 110, 115, 120, 140] meter
house Price: Y=[770, 880, 990, 1100, 1220, 1265, 1320, 1540] Milion Toman

Derive the linear regression formula and solve it.
Hint:
Y(hat) = AW+b
what is the problem objective? Minimize the mean squared error
take MSE derivative w.r.t W and b and solve to find W and b
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.array([[70, 80, 90, 100, 110, 115, 120, 140], [790, 890, 980, 1170, 1260, 1305, 1320, 1540]], dtype=float)
data = pd.DataFrame(data=data.T, columns=['area', 'price'])

area_avg = np.mean(data['area'])
price_avg = np.mean(data['price'])

area_minus_avg = data['area'] - area_avg

W = np.dot(data['price'], area_minus_avg) / np.dot(area_minus_avg, area_minus_avg)
b = price_avg - W * area_avg

y_pred = data['area'] * W + b
plt.scatter(x=data['area'], y=data['price'])
plt.plot(data['area'], y_pred, color='red')
plt.xlabel("Area (meter squared)")
plt.ylabel("price (milion Toman)")
plt.legend()
plt.show()
