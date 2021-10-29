import math

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Main(object):

    def plotGraf(self, x_scale, Euler_scale, ImprovedEuler_scale, RungeKutta_scale, Exact_scale):
        fig = make_subplots()
        fig.add_trace(go.Scatter(x=x_scale, y=Euler_scale, name='Euler Method'))
        fig.add_trace(go.Scatter(x=x_scale, y=ImprovedEuler_scale, name='Improved Euler Method'))
        fig.add_trace(go.Scatter(x=x_scale, y=RungeKutta_scale, name='Runge-Kutta Method'))
        fig.add_trace(go.Scatter(x=x_scale, y=Exact_scale, name='Exact Solution'))

        fig.update_layout(title_text='Approximation Methods')

        fig.update_xaxes(title_text='X')

        fig.update_yaxes(title_text='Y')

        st.write(fig)

    def plotLTE(self, x_scale, Euler_scale, ImprovedEuler_scale, RungeKutta_scale):

        fig = make_subplots()

        fig.add_trace(go.Scatter(x=x_scale, y=Euler_scale, name='Euler Method'))
        fig.add_trace(go.Scatter(x=x_scale, y=ImprovedEuler_scale, name='Improved Euler Method'))
        fig.add_trace(go.Scatter(x=x_scale, y=RungeKutta_scale, name='Runge-Kutta Method'))

        fig.update_layout(title_text='LTE')

        fig.update_xaxes(title_text='X')

        fig.update_yaxes(title_text='Y')

        st.write(fig)

    def plotGTE(self, x_scale, Euler_scale, ImprovedEuler_scale, RungeKutta_scale):

        fig = make_subplots()

        fig.add_trace(go.Scatter(x=x_scale, y=Euler_scale, name='Euler Method'))
        fig.add_trace(go.Scatter(x=x_scale, y=ImprovedEuler_scale, name='Improved Euler Method'))
        fig.add_trace(go.Scatter(x=x_scale, y=RungeKutta_scale, name='Runge-Kutta Method'))

        fig.update_layout(title_text='GTE')

        fig.update_xaxes(title_text='X')

        fig.update_yaxes(title_text='Y')

        st.write(fig)

    def inputX0(self):
        return st.number_input(label='x0', value=0)

    def inputY0(self):
        return st.number_input(label='y0', value=math.sqrt(0.5))

    def inputX(self):
        return st.number_input(label='X', value=3)

    def inputN(self):
        return st.number_input(label='N', min_value=1, max_value=1000, value=15)

    def inputn0(self):
        return st.number_input(label='n0', value=1)

    def inputN1(self):
        return st.number_input(label='N', value=10)

    def call_error(self, text):
        st.error(text)


class Grid(Main):
    setvalue = Main()
    x0 = setvalue.inputX0()
    y0 = setvalue.inputY0()
    X = setvalue.inputX()
    N = setvalue.inputN()
    h = (X - x0) / N
    x_scale = [x0]

    def fillXArray(self):
        for i in range(1, self.N + 1):
            self.x_scale.append(self.x_scale[i - 1] + self.h)


class NumericalMethod(Grid):
    def function(self, x_param, y_param):
        return x_param * y_param - x_param * y_param ** 3


class EulerMethod(NumericalMethod):
    y_scale = []

    def EulerCalculation(self, x_param, y_param, h_param):
        return y_param + h_param * self.function(x_param, y_param)

    def EulerArray(self):
        for i in range(self.N + 1):
            if i == 0:
                self.y_scale.append(self.y0)
            else:
                self.y_scale.append(self.EulerCalculation(self.x_scale[i - 1], self.y_scale[i - 1], self.h))


class ImprovedEulerMethod(NumericalMethod):
    y_scale = []

    def ImprovedEulerCalculation(self, x_param, y_param, h_param):
        return y_param + h_param * self.function(x_param + h_param / 2,
                                                 y_param + h_param / 2 * self.function(x_param, y_param))

    def ImprovedEulerArray(self):
        for i in range(self.N + 1):
            if i == 0:
                self.y_scale.append(self.y0)
            else:
                self.y_scale.append(self.ImprovedEulerCalculation(self.x_scale[i - 1], self.y_scale[i - 1], self.h))


class RungeKuttaMethod(NumericalMethod):
    y_scale = []

    def K1(self, x_param, y_param, h_param):
        return self.function(x_param, y_param)

    def K2(self, x_param, y_param, h_param):
        return self.function(x_param + h_param / 2, y_param + h_param * self.K1(x_param, y_param, h_param) / 2)

    def K3(self, x_param, y_param, h_param):
        return self.function(x_param + h_param / 2, y_param + h_param * self.K2(x_param, y_param, h_param) / 2)

    def K4(self, x_param, y_param, h_param):
        return self.function(x_param + h_param, y_param + h_param * self.K3(x_param, y_param, h_param))

    def RungeKuttaCalculation(self, x_param, y_param, h_param):
        return y_param + h_param / 6 * (self.K1(x_param, y_param, h_param) +
                                        self.K2(x_param, y_param, h_param) * 2 +
                                        self.K3(x_param, y_param, h_param) * 2 +
                                        self.K4(x_param, y_param, h_param))

    def RungeKuttaArray(self):
        for i in range(self.N + 1):
            if i == 0:
                self.y_scale.append(self.y0)
            else:
                self.y_scale.append(self.RungeKuttaCalculation(self.x_scale[i - 1], self.y_scale[i - 1], self.h))


class ExactSolutionMethod(Grid):
    y_scale = []
    C = 1

    def const(self):
        return self.y0 / (math.sqrt(1 - self.y0) * math.exp(self.x0 ** 2 / 2))

    def ExactSolutionCalculation(self, x_param):
        return math.sqrt(math.exp(x_param ** 2) / (1 + math.exp(x_param ** 2) * self.C))

    def ExactSolutionArray(self):
        for i in range(self.N + 1):
            self.y_scale.append(self.ExactSolutionCalculation(self.x_scale[i]))


class Master(object):
    def master(self, grid):
        grid.fillXArray()

        Euler = EulerMethod()
        ImprovedEuler = ImprovedEulerMethod()
        RungeKutta = RungeKuttaMethod()
        ExactSolution = ExactSolutionMethod()

        X = grid.x_scale
        Y_exact = ExactSolution.y_scale
        h = grid.h

        main = Main()

        Euler.EulerArray()
        ImprovedEuler.ImprovedEulerArray()
        RungeKutta.RungeKuttaArray()
        ExactSolution.ExactSolutionArray()

        main.plotGraf(grid.x_scale, Euler.y_scale, ImprovedEuler.y_scale, RungeKutta.y_scale, ExactSolution.y_scale)

        EulerLTE = [0]
        ImprovedEulerLTE = [0]
        RungeKuttaLTE = [0]

        for i in range(1, grid.N + 1):
            EulerLTE.append(abs(Y_exact[i] - Euler.EulerCalculation(X[i - 1], Y_exact[i - 1], h)))
            ImprovedEulerLTE.append(
                abs(Y_exact[i] - ImprovedEuler.ImprovedEulerCalculation(X[i - 1], Y_exact[i - 1], h)))
            RungeKuttaLTE.append(abs(Y_exact[i] - RungeKutta.RungeKuttaCalculation(X[i - 1], Y_exact[i - 1], h)))

        main.plotLTE(grid.x_scale, EulerLTE, ImprovedEulerLTE, RungeKuttaLTE)

        EulerGTE = []
        ImprovedEulerGTE = []
        RungeKuttaGTE = []

        n0 = main.inputn0()
        N = main.inputN1()

        if n0 > N:
            main.call_error('n0 > N')
        elif n0 == 0:
            main.call_error('n0 == 0')

        x_GTE = []
        for i in range(n0, N + 1):
            x_GTE.append(i)

        for nTmp in x_GTE:
            h = (grid.X - grid.x0) / nTmp
            x = grid.x0
            yArr = [grid.y0]

            EulerMaxLTE = 0
            ImprovedEulerMaxLTE = 0
            RungeKuttaMaxLTE = 0
            for i in range(1, nTmp):
                x += h
                yArr.append(ExactSolution.ExactSolutionCalculation(x))
                EulerMaxLTE = max(EulerMaxLTE, abs(yArr[i] - Euler.EulerCalculation(x - h, yArr[i - 1], h)))
                ImprovedEulerMaxLTE = max(ImprovedEulerMaxLTE, abs(yArr[i] - ImprovedEuler.ImprovedEulerCalculation(x - h, yArr[i - 1], h)))
                RungeKuttaMaxLTE = max(RungeKuttaMaxLTE, abs(yArr[i] - RungeKutta.RungeKuttaCalculation(x - h, yArr[i - 1], h)))

            EulerGTE.append(EulerMaxLTE)
            ImprovedEulerGTE.append(ImprovedEulerMaxLTE)
            RungeKuttaGTE.append(RungeKuttaMaxLTE)

        main.plotGTE(x_GTE, EulerGTE, ImprovedEulerGTE, RungeKuttaGTE)

grid = Grid()
master = Master()
master.master(grid)