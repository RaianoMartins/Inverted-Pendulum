import numpy as np
import cv2
from invertedPendulumGraphics import InvertedPendulum
from scipy.integrate import solve_ivp
import control
import matplotlib.pyplot as plt

class MyLinearizedSystem:
    def __init__(self):
        g = 9.8
        L = 1.5
        m = 1.0
        M = 5.0
        b = 1

        self.A = np.array([
            [0, 1, 0, 0],
            [0, -b / M, m * g / M, 0],
            [0, 0, 0, 1],
            [0, -b / (M * L), (m + M) * g / (M * L), 0]
        ])

        self.B = np.expand_dims(np.array([0, 1.0 / M, 0., 1 / (M * L)]), 1)  # 4x1

    def compute_K(self, desired_eigs=[-0.1, -0.2, -0.3, -0.4]):
        self.K = control.place(self.A, self.B, desired_eigs)

    def get_K(self):
        return self.K

def u(y):
    u_ = -np.matmul(ss.K, y)
    return u_[0]

def y_dot(t, y):
    g = 9.8
    L = 1.5
    m = 1.0
    M = 5.0
    b = 1

    x_ddot = m * g * np.cos(y[2]) * np.sin(y[2]) - m * L * (y[3] ** 2) * np.sin(y[2]) + u(y) - b * y[1]
    x_ddot = x_ddot / (M + m * (1 - np.cos(y[2]) ** 2))

    theta_ddot = (M + m) * g * np.sin(y[2]) - m * L * (y[3] ** 2) * np.sin(y[2]) * np.cos(y[2]) + (
            u(y) - b * y[1]) * np.cos(y[2])
    theta_ddot = theta_ddot / (L * (M + m * (1 - np.cos(y[2]) ** 2)))

    return [y[1], x_ddot, y[3], theta_ddot]

if __name__ == "__main__":
    ss = MyLinearizedSystem()

    # Cenário 1
    Q = np.diag([25, 1, 1, 1])
    R = np.diag([0.025])

    K, S, E = control.lqr(ss.A, ss.B, Q, R)
    ss.compute_K(E)

    sol = solve_ivp(y_dot, [0, 20], [-1, 0, -0.4, 0], t_eval=np.linspace(0, 20, 200))
    syst = InvertedPendulum()

    recalculated_u = [u(sol.y[:, i]) for i in range(len(sol.t))]

    plt.ion()  # Ativa o modo interativo do matplotlib

    # Cria uma figura para a lei de controle
    fig, ax = plt.subplots()

    # Configuração do gráfico
    line, = ax.plot([], [], label='u(t)')
    ax.set_xlim(0, sol.t[-1])
    ax.set_ylim(min(recalculated_u) - 1, max(recalculated_u) + 1)
    ax.legend()
    ax.grid(True)  # Adiciona grades ao gráfico
    ax.set_title('Lei de Controle u(t)')

    plt.show()

    # Inicializa a janela do OpenCV
    initial_frame = syst.step([sol.y[0, 0], sol.y[1, 0], sol.y[2, 0], sol.y[3, 0]], sol.t[0])
    cv2.imshow('im', initial_frame)
    cv2.moveWindow('im', 100, 100)  # Muda a posição da janela do OpenCV

    # Pausa para iniciar a simulação
    print("Pressione 'q' na janela do OpenCV para iniciar a simulação.")
    while True:
        if cv2.waitKey(30) == ord('q'):
            break

    # Simulação
    for i, t in enumerate(sol.t):
        rendered = syst.step([sol.y[0, i], sol.y[1, i], sol.y[2, i], sol.y[3, i]], t)
        cv2.imshow('im', rendered)

        line.set_data(sol.t[:i+1], recalculated_u[:i+1])

        ax.relim()
        ax.autoscale_view()

        plt.draw()
        plt.pause(0.001)  # Atualiza o gráfico

        if cv2.waitKey(30) == ord('q'):
            break

    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()
