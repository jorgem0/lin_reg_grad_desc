import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Original Parabola Function
x = np.linspace(-1.25, 1.25, 100)
y = x ** 2  # parabola
# y = x ** 4 - x ** 2 # quadratic

n = len(x)  # Number of samples
iter = 0  # First iteration
itersteps = 200  # Total iterations

# Initial guess for x and vectors to keep track of certain values
xx = np.ones(itersteps) * -1
yy = np.zeros(itersteps)
J = np.zeros(itersteps)
dJdx = np.zeros(itersteps)

a = 0.01  # Step size

# While loop which stops after itersteps-1 due to the xx[iter+1]
while iter < itersteps - 1:
    J[iter] = xx[iter] ** 2  # Original function y=x^2
    # J[iter] = xx[iter] ** 4 -xx[iter] ** 2  # Original function y=x^4-x^2

    dJdx[iter] = 2 * xx[iter]  # Derivative of function y=x^2
    # dJdx[iter] = 4 * xx[iter]**3 -2 * xx[iter]  # Derivative of function y=x^4-x^2

    yy[iter] = xx[iter] ** 2  # Value of y at the current xx for y=x^2
    # yy[iter] = xx[iter] ** 4 - xx[iter] ** 2 # Value of y at the current xx for y=x^4-x^2

    print(xx[iter], yy[iter])
    xx[iter + 1] = xx[iter] - a * dJdx[iter]  # New xx value
    iter += 1

# Creating the initial figure/plot
fig1, ax, = plt.subplots(figsize=(10, 10))
ax.plot(x, y)  # Plot of parabola function that does not get

line, = ax.plot(xx[0], xx[0] ** 2, 'red')  # initial data y=x^2
dot = ax.scatter(xx[0], xx[0] ** 2, c='g', marker="o")  # initial point y=x^2
t1 = ax.text(xx[0], xx[0] ** 2, '%s' % (str([xx[0], xx[0] ** 2])), size=12, zorder=1, color='k', verticalalignment='top')  # initial point text y=x^2

# line, = ax.plot(xx[0], xx[0] ** 4 -xx[0] ** 2, 'red')  # initial data y=x^4-x^2
# dot = ax.scatter(xx[0], xx[0] ** 4 -xx[0] ** 2, c='g', marker="o")  # initial point y=x^4-x^2
# t1 = ax.text(xx[0], xx[0] ** 4 -xx[0] ** 2, '%s' % (str([xx[0], xx[0] ** 4 -xx[0] ** 2])), size=12, zorder=1, color='k', verticalalignment='top') # initial point text y=x^4-x^2

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')


# Function updates data for plot animation using animation.FuncAnimation() below
# The variable that is passed to this function from FuncAnimation is frames=itersteps-1
# This acquires the data at every iteration step
def update(iter):
    # Setting new values for plot
    ax.relim()  # resizing plot area
    ax.autoscale_view(True, True, True)  # resizing plot area
    line.set_data(xx[0:iter], yy[0:iter])
    dot.set_offsets([xx[iter], yy[iter]])
    t1.set_position((xx[iter], yy[iter]))

    t1.set_text(str([round(xx[iter], 8), round(xx[iter] ** 2, 8)]))  # point text y=x^2
    # t1.set_text(str([round(xx[iter], 8), round(xx[iter] ** 4 - xx[iter] ** 2 , 8)]))  # point text y=x^4-x^2

    ax.set_title(r'Gradient Descent $\alpha$=' + str(a) + ' iteration: ' + str(iter))
    ax.legend(bbox_to_anchor=(1, 1.2), fontsize='x-small')  # legend location and font
    return line, ax


# Animation function after all the variables at each iteration have been calculated
# Calls the function update and passes in the number of frames=itersteps-1 to get all the data at each iteration
ani = animation.FuncAnimation(fig1, update, frames=itersteps - 1, interval=10, blit=False, repeat_delay=100)

# Saves animations as .mp4
filename = 'parabola_example_alpha_' + str(a) + '.mp4'
# filename = 'quadratic_example_alpha_' + str(a) + '.mp4'
ani.save(filename, fps=30, extra_args=['-vcodec', 'libx264'])

# Shows plot which is animated
plt.show()