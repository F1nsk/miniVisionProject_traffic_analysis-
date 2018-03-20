import math
import numpy as np
from numpy import matlib as matlib
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.models.widgets import Button
from bokeh.plotting import figure, output_file, show

class Kalman:

    def __init__(self, F, B, Q, H, R, init_x, init_mu, init_P, deltaT):
        np.set_printoptions(suppress=True)

        self.F = F
        self.B = B
        self.Q = Q
        self.H = H
        self.R = R
        self.K = np.matrix([[]])
        self.prevX = init_x
        self.prevP = init_P
        self.prevMu = init_mu
        self.nextX = init_x
        self.nextP = init_P
        self.deltaT = deltaT

        self.observations = np.array([0])
        self.predictions = np.array([[0]])

        self.iterations = 0

    def calcNextX(self):
        self.prevX = self.F * self.nextX + self.B * self.prevMu

    def calcNextP(self):
        self.prevP = self.F * self.nextP * self.F.T + self.Q

    def calcK(self):
        self.K = self.prevP * self.H.T * ((self.H * self.prevP * self.H.T + self.R)**-1)

    def calcPredictX(self, measurement):
        self.nextX = self.prevX + self.K * (measurement - self.H * self.prevX)

        temp = np.array(self.nextX)

        #print(self.predictions)
        #print(temp)
        self.predictions = np.vstack((self.predictions, temp))


    def calcPredictP(self):
        self.nextP = (np.matlib.identity(4) - self.K * self.H) * self.prevP

    def transformObservation(self, new_obs):
        temp = np.array(new_obs)
        self.observations = np.vstack((self.observations, temp))
        return self.H * new_obs

    def run_one_step(self, new_obs):
        self.calcNextP()
        self.calcK()
        self.calcNextX()
        new_obs = self.transformObservation(new_obs)
        self.calcPredictX(new_obs)
        self.calcPredictP()

        self.iterations = self.iterations + 1

        return self.nextX

    def draw_graph(self, var):
        plot = figure(plot_width=600, plot_height=600, title="Data from the tests")
        plot.grid.grid_line_alpha = 0.3
        plot.xaxis.axis_label = 'Iteration'
        plot.yaxis.axis_label = 'Value'

        self.observations = np.delete(self.observations, 0)
        self.predictions = np.delete(self.predictions, (1), axis=1)
        self.predictions = np.delete(self.predictions, 0)
        self.predictions = self.predictions[var::4]
        self.observations = self.observations[var::4]

        t = np.linspace(0, (len(self.observations) - 1) * self.deltaT, len(self.observations))

        print(self.observations)

        print(self.predictions)

        plot.line(t, self.observations, color='red', legend='CW_Small_1')
        plot.line(t, self.predictions, color='green', legend='CW_Small_2')
        plot.legend.location = "top_left"

        show(plot)

    def get_number_of_iterations(self):
        return self.iterations

    def get_nextX(self):
        return self.nextX


""" Example of how to use the kalman filter, using x and y direction with speed


deltaT = 1.

F = np.matrix([[1, 0, deltaT, 0], [0, 1, 0, deltaT], [0, 0, 1, 0], [0, 0, 0, 1]])
B = np.matrix([[deltaT**2 / 2, 0], [0, deltaT**2 / 2], [deltaT, 0], [0, deltaT]])
H = np.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

model_noise = 0.01
measurement_noise = 20

R = measurement_noise**2 * np.matlib.identity(4) / deltaT
Q = model_noise**2 * np.matlib.identity(4) * deltaT

init_x = np.matrix([[0.], [0.], [0], [0]])
init_mu = np.matrix([[0., 0], [0., 0]])  # This is the acceleration estimate
init_P = np.matlib.identity(4)
kalman = Kalman(F, B, Q, H, R, init_x, init_mu, init_P, deltaT)

for i in range(0, 150, 5):
    random_number = randint(-10, 10)
    measurement_x = i + random_number
    next_x = np.matrix([[measurement_x], [measurement_x], [random_number], [random_number]])
    kalman.run_one_step(next_x)

kalman.draw_graph(2)
"""