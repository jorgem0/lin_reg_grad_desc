import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Importing csv data as Pandas dataframe
df_test = pd.read_csv(r'C:\USERNAME\test.csv')

print('----------Test Data----------')
print(df_test.head())  # Printing first five rows of dataframe

# Converting data to array for later calculations
x_test = np.array(df_test['x'])
y_test = np.array(df_test['y'])

n = len(x_test)  # Number of samples in data

iter = 0  # Initial iteration counter
itersteps = 250  # Total number of iterations

# Initial guesses for linear regression model of y=mx+b and arrays to keep track of values over iterations
m = np.zeros(itersteps)
b = np.zeros(itersteps)
J = np.zeros(itersteps)
dJdm = np.zeros(itersteps)
dJdb = np.zeros(itersteps)
SSR = np.zeros(itersteps)
SSE = np.zeros(itersteps)
SSTO = np.zeros(itersteps)
R_sq = np.zeros(itersteps)

a = 1E-5  # Step size

# Gradient Descent implementation on the linear regression model
while iter < itersteps - 1:
    print('Iteration: ', iter)
    y_new = m[iter] * x_test + b[iter]

    J[iter] = 1 / n * np.sum((y_test - y_new) ** 2)  # Mean Squared Error

    dJdm[iter] = 1 / n * np.sum((y_test - y_new) * -2 * x_test)
    dJdb[iter] = 1 / n * np.sum((y_test - y_new) * -2)

    SSR[iter] = np.sum((y_new - np.mean(y_test)) ** 2)
    SSE[iter] = np.sum((y_test - y_new) ** 2)
    SSTO[iter] = np.sum((y_test - np.mean(y_test)) ** 2)
    R_sq[iter] = 1-SSE[iter]/SSTO[iter]

    m[iter + 1] = m[iter] - a * dJdm[iter]
    b[iter + 1] = b[iter] - a * dJdb[iter]
    iter += 1

# Data to create a smooth line for the linear regression model
x_min = min(x_test)
x_max = max(x_test)
x = np.linspace(x_min, x_max, n)
y = m[0] * x + b[0]

# Creating initial figure
fig1, ax, = plt.subplots(figsize=(10, 10))
ax.scatter(x_test, y_test)  # Initial scatter plot that does not get updated
line, = ax.plot(x, y, 'red')  # Initial linear fit
ax.set_xlabel('X')
ax.set_ylabel('Y')
t1 = ax.text(50, 25, 'Eqn: y=' + str(round(m[iter], 4)) + '*x+' + str(round(b[iter], 4)), fontsize=15) # Text displays y=mx+b on plot


# Function updates data for plot animation using animation.FuncAnimation() below
# The variable that is passed to this function from FuncAnimation is frames=itersteps-1
# This acquires the data at every iteration step
def update(iter):
    y = m[iter] * x + b[iter]  # The linear fit data at each iteration using m and b values at that iteration
    line.set_data(x, y)  # Updates linear fit data for plot
    t1.set_text('Eqn: y=' + str(round(m[iter], 4)) + '*x+' + str(round(b[iter], 4)))
    ax.set_title('Linear Regression with $R^2=$' + str(round(R_sq[iter], 4)) + ' iteration: ' + str(iter))
    ax.legend(bbox_to_anchor=(1, 1.2), fontsize='x-small')  # legend location and font
    return line, ax


# Animation function after all the variables at each iteration have been calculated
# Calls the function update and passes in the number of frames=itersteps-1 to get all the data at each iteration
ani = animation.FuncAnimation(fig1, update, frames=itersteps - 1, interval=10, blit=False, repeat_delay=100)
ani.save('lin_reg.mp4', fps=30, extra_args=['-vcodec', 'libx264'])  # Saves animation

# Figure of errors and coefficient of determination
R_sq0=np.where(R_sq>0) # Finds indices of where Rsq>0

fig2, ax2 = plt.subplots(figsize=(10, 10))

ax2.plot(np.arange(R_sq0[0][0], itersteps - 1), SSE[R_sq0[0][0]:-1], label='SSE', color='blue')
ax2.plot(np.arange(R_sq0[0][0], itersteps - 1), SSR[R_sq0[0][0]:-1], label='SSR', color='#0080ff')
ax2.plot(np.arange(R_sq0[0][0], itersteps - 1), SSTO[R_sq0[0][0]:-1], label='SSTO', color='#00c0ff')

ax2.set_title('Errors')
ax2.set_xlabel('Number of Iterations')
ax2.set_ylabel('Error', color='blue')
ax2.tick_params('y', colors='blue')

ax3 = ax2.twinx() # Second set of data on same x-axis
ax3.plot(np.arange(R_sq0[0][0], itersteps - 1), R_sq[R_sq0[0][0]:-1], color='red', label='$R^2$')
ax3.set_ylabel('Rsq', color='red')
ax3.tick_params('y', colors='red')

ax2.legend(bbox_to_anchor=(1, .60), fontsize='x-small')  # legend location and font size
ax3.legend(bbox_to_anchor=(1, .50), fontsize='x-small')  # legend location and font size
fig2.savefig('errors.png')  # saves figure

# Creating mesh for 3D Surface Plot of MSE
mm = np.linspace(-10, 10, 1000)
bb = np.linspace(-10, 10, 1000)
mmm, bbb = np.meshgrid(mm, bb)

JJ = np.zeros([len(mm), len(bb)])

# Calculating values of MSE for different combinations of m and b
for i in range(len(mm)):
    for j in range(len(bb)):
        print('i: ',i, 'j: ', j)
        yy = mmm[i, j] * x_test + bbb[i, j]
        JJ[i, j] = 1 / n * np.sum((y_test - yy) ** 2)

# 3D Surface Plot
fig3 = plt.figure()
ax4 = fig3.gca(projection='3d')
surf = ax4.plot_surface(mmm, bbb, JJ, cmap=plt.cm.jet)
fig3.colorbar(surf)

ax4.plot(m[:-1], b[:-1], J[:-1], 'red')
ax4.scatter(m[-2], b[-2], J[-2], c='b', marker="o",label='Gradient Descent')
ax4.text(m[-2], b[-2], J[-2], '%s' % ('MSE: '+str(round(J[-2],5))), color='k')
ax4.set_xlabel('m')
ax4.set_ylabel('b')
ax4.set_zlabel('MSE')
ax4.set_title('3D Surface Plot of MSE')

# Acquiring and plotting minimum value of MSE
i_min, j_min = np.unravel_index(JJ.argmin(), JJ.shape)
ax4.scatter(mmm[i_min, j_min], bbb[i_min, j_min], np.min(JJ), c='g', marker="o",label='3D Plot')
ax4.text(mmm[i_min, j_min], bbb[i_min, j_min], np.min(JJ), '%s' % ('MSE: '+str(round(np.min(JJ),5))), color='k')

ax4.legend(bbox_to_anchor=(0.15, .50), fontsize='x-small')

fig3.savefig('MSE_3D.png')  # saves figure



# Calculating linear fit and errors for both methods
# Gradient Descent
y = m[-1] * x_test + b[-1] # Predicted Values
y2 = m[-1] * x + b[-1] # Linear fit
SSR1 = np.sum((y - np.mean(y_test)) ** 2)
SSE1 = np.sum((y_test - y) ** 2)
SSTO1 = np.sum((y_test - np.mean(y_test)) ** 2)
R_sq1 = 1-SSE1/SSTO1

# 3D Plot
yyy = mmm[i_min, j_min] * x_test + bbb[i_min, j_min] # Predicted Values
yyy2 = mmm[i_min, j_min] * x + bbb[i_min, j_min] # Linear fit
SSR2 = np.sum((yyy - np.mean(y_test)) ** 2)
SSE2 = np.sum((y_test - yyy) ** 2)
SSTO2 = np.sum((y_test - np.mean(y_test)) ** 2)
R_sq2 = 1-SSE2/SSTO2


# Printing Min MSE, m, and b for both methods
print('Gradient Descent Min MSE: ', J[-2])
print('Gradient Descent m: ', m[-2], ' and b: ', b[-2])
print('Gradient Descent $R^2$: ', R_sq1)
print('3D Plot Min MSE: ', np.min(JJ))
print('3D Plot m: ', mmm[i_min, j_min], ' and b: ', bbb[i_min, j_min])
print('3D Plot $R^2$: ', R_sq2)


# Linear fit for both methods on scatter plot
fig4, ax5 = plt.subplots(figsize=(10, 10))
ax5.scatter(x_test, y_test)  # initial data that does not get updated
ax5.plot(x, y2, 'blue', label='Gradient Descent: y=' + str(round(m[-1], 5)) + '*x+' + str(round(b[-1], 5)) + ' $R^2=$' + str(round(R_sq1, 5)))  # initial linear fit
ax5.plot(x, yyy2, 'green', label='3D Plot: y=' + str(round(mmm[i_min, j_min], 5)) + '*x+' + str(round(bbb[i_min, j_min], 5)) + ' $R^2=$' + str(round(R_sq2, 5)))  # initial linear fit
ax5.set_title('Linear Regression')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.legend()

fig4.savefig('lin_reg_both.png')  # saves figure

# Residual plot for both methods
e1=y_test-y
e2=y_test-yyy

fig5, ax6 = plt.subplots(figsize=(10, 10))
ax6.scatter(x_test, e1, c='blue', label='Gradient Descent: y=' + str(round(m[-1], 5)) + '*x+' + str(round(b[-1], 5)) + ' $R^2=$' + str(round(R_sq1, 5)))  # initial linear fit
ax6.scatter(x_test, e2, c='green', label='3D Plot: y=' + str(round(mmm[i_min, j_min], 5)) + '*x+' + str(round(bbb[i_min, j_min], 5)) + ' $R^2=$' + str(round(R_sq2, 5)))  # initial linear fit
ax6.set_title('Residuals')
ax6.set_xlabel('X')
ax6.set_ylabel('Residual')
ax6.legend()

fig5.savefig('res_both.png')  # saves figure

# Contour plot of 3D plot
fig6, ax7 = plt.subplots(figsize=(10, 10))
ax7.contour(mmm, bbb, JJ, 100, cmap=plt.cm.jet);
ax7.plot(m[:-1], b[:-1], 'red')
ax7.scatter(m[-2], b[-2], c='b', marker="o",label='Gradient Descent: y=' + str(round(m[-1], 5)) + '*x+' + str(round(b[-1], 5)) + ' $R^2=$' + str(round(R_sq1, 5)))
ax7.text(m[-2], b[-2], '%s' % ('MSE: '+str(round(J[-2],5))), color='k')
ax7.scatter(mmm[i_min, j_min], bbb[i_min, j_min], c='g', marker="o",label='3D Plot: y=' + str(round(mmm[i_min, j_min], 5)) + '*x+' + str(round(bbb[i_min, j_min], 5)) + ' $R^2=$' + str(round(R_sq2, 5)))
ax7.text(mmm[i_min, j_min], bbb[i_min, j_min], '%s' % ('MSE: '+str(round(np.min(JJ),5))), color='k')
ax7.set_xlabel('m')
ax7.set_ylabel('b')
ax7.set_title('Contour Plot of MSE')
ax7.legend()

fig6.savefig('contour.png')  # saves figure

plt.show()  # Show all plots