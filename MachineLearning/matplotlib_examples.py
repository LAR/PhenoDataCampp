# DataCampp Machine Learning for image data, Matplotlib examples

import matplotlib.pyplot as plt
import numpy as np


# Example 1 - a simple line plot

x = np.linspace(-5, 5, 1000)
y = np.cos(x)

plt.plot(x, y)

plt.savefig('my_figure_name.png')

plt.show()


# Example 2 - a plot with multiple lines

x = np.linspace(-5, 5, 1000)
y1 = np.cos(x)
y2 = np.sin(x)

plt.plot(x, y1,'r')
plt.plot(x, y2,'g')
plt.show()


# Example 3 - a scatter plot

x = np.linspace(0,10,15)
y1 = np.random.random(15)
y2 = np.random.random(15)

plt.plot(x,y1,'Xr') # scatter plot chosen by adding 'X' character in string
plt.scatter(x,y2,marker='o',color='b') # scatter plot chosen explicitly

plt.show()


# Example 4 - a fully annotated scatter plot

x = np.linspace(0,10,15)
y1 = np.random.random(15)
y2 = np.random.random(15)


# its often useful to explicitly make blank figures and axes
# so you can then refer to them directly

fig = plt.figure() # make a blank figure
ax = plt.axes() # add some axes


ax.plot(x,y1,'Xr',label='apples') # notice the keyword label 
ax.scatter(x,y2,marker='o',color='b', label='oranges')

ax.legend() # add legend (using the labels set above) 

ax.set_xlim([-5,15]) # set limits of x-axis
ax.set_ylim([-0.5,1.5]) # set limits of y-axis

ax.set_xlabel('some x values') # set label of x-axis
ax.set_ylabel('some y values') # set label of y-axis



ax.set_title('random fruit data')

plt.show()



