import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import numpy as np

class Utils:
    def DrawDirect(u,a,b,x,u_left,u_right,M):
        fig = plt.figure(facecolor='white')
        ax = plt.axes(xlim=(a, b), ylim=(u_left, u_right))
        line, = ax.plot([], [], lw=1)

        def animate(i):
            line.set_xdata(x)
            line.set_ydata(u[i, :])
            return line

        anim = animation.FuncAnimation(fig, animate, frames=M, interval=100)
        plt.show()
    def DrawConjugate(S,x,q,init_q):

        def q_init(x):
            return 2 * x - 1 + 2 * np.sin(5 * x * np.pi) + 0.35

        plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg'
        # График зависимости функционала от номера итерации
        # plt.plot(J[0:S - 1])
        # plt.show()
        # plt.savefig('J.pdf')

        fig2 = plt.figure(facecolor='white')
        ax = plt.axes(xlim=(0, 1), ylim=(-2, 2))
        line, = ax.plot([], [], lw=1, color='red')
        line2, = ax.plot([], [], lw=1, color='green')

        # Анимация
        def animate(i):
            line.set_xdata(x)
            line.set_ydata(q[i, :])
            line2.set_xdata(x)
            line2.set_ydata(init_q)
            return line

        anim = animation.FuncAnimation(fig2, animate, frames=S, interval=100)
        FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
        anim.save(r'C:\Users\FS\Desktop\Main Mission\Conjucate_problem_solution.mp4', writer=FFwriter)
        plt.show()
    def DrawSolution(S,x,q,J,init_q):

        def q_init(x):
            return 2 * x - 1 + 2 * np.sin(5 * x * np.pi) + 0.35
        plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg'
        # График зависимости функционала от номера итерации
        plt.plot(J[0:S - 1])
        plt.show()
        plt.savefig('J.pdf')

        fig2 = plt.figure(facecolor='white')
        ax = plt.axes(xlim=(0, 1), ylim=(-2, 2))
        line, = ax.plot([], [], lw=1, color='red')
        line2, = ax.plot([], [], lw=1, color='green')

        # Анимация
        def animate(i):
            line.set_xdata(x)
            line.set_ydata(q[i, :])
            line2.set_xdata(x)
            line2.set_ydata(init_q)
            return line

        anim = animation.FuncAnimation(fig2, animate, frames=S, interval=100)
        FFwriter = animation.FFMpegWriter(fps=30, extra_args=['-vcodec', 'libx264'])
        anim.save(r'C:\Users\FS\Desktop\Main Mission\Conjucate_problem_solution.mp4', writer=FFwriter)
        plt.show()
    def DiagonalPreparationDirect(N, eps, tau, q, h, y,u_left,u_right):
        a_diag = np.zeros(N - 1, dtype=complex)
        b_diag = np.zeros(N - 1, dtype=complex)
        c_diag = np.zeros(N - 1, dtype=complex)

        a_diag[0] = 1 - (1 + 1j) / 2 * tau * (-2 * eps / h ** 2) + ((y[1] - u_left) / (2 * h)) - q[1]
        for n in range(1, N - 1):
            b_diag[n] = -(1 + 1j) / 2 * tau * (eps / h ** 2 - y[n] / (2 * h))
        for n in range(1, N - 2):
            a_diag[n] = 1 - (1 + 1j) / 2 * tau * (-2 * eps / h ** 2 + ((y[n + 1] - y[n - 1]) / (2 * h)) - q[n + 1])
        for n in range(0, N - 2):
            c_diag[n] = -(1 + 1j) / 2 * tau * (eps / h ** 2 + y[n] / (2 * h))
        a_diag[N - 2] = 1 - (1 + 1j) / 2 * tau * (-2 * eps / h ** 2 + ((u_right - y[N - 3]) / (2 * h)) - q[N - 1])
        return a_diag, b_diag, c_diag
    def DiagonalPreparationConjugate(eps, N, tau, q, h, u):
        a_diag = np.zeros(N - 1, dtype=complex)
        b_diag = np.zeros(N - 1, dtype=complex)
        c_diag = np.zeros(N - 1, dtype=complex)

        for n in range(1, N - 1):
            b_diag[n] = (1 + 1j) / 2 * tau * (-eps / h ** 2 - u[n + 1] / (2 * h))
        for n in range(0, N - 1):
            a_diag[n] = 1 + (1 + 1j) / 2 * tau * (2 * eps / h ** 2 + q[n + 1])
        for n in range(0, N - 2):
            c_diag[n] = (1 + 1j) / 2 * tau * ((-eps / h ** 2) + (u[n + 1] / (2 * h)))

        return a_diag, b_diag, c_diag
    def TridiagonalMatrixAlgorithm(a_diag, b_diag, c_diag, B):
        n = len(B)
        Ab = np.zeros((3, n), dtype=complex)
        Ab[0, 1:] = c_diag[:-1]
        Ab[1, :] = a_diag
        Ab[2, :-1] = b_diag[1:]
        X = solve_banded((1, 1), Ab, B)
        return X
    def gradient_calculation(u, psi, tau, M, N):
        result = np.zeros(N + 1)
        for n in range(N + 1):
            for m in range(1, M + 1):
                result[n] += (u[m, n] * psi[m, n] + u[m - 1, n] * psi[m - 1, n]) * tau / 2
        return resul
    def functional_calculation(u, f_obs, h, N):
        result = 0
        for n in range(1, N + 1):
            result += ((u[n] - f_obs[n]) ** 2 + (u[n - 1] - f_obs[n - 1]) ** 2) * h / 2
        return result