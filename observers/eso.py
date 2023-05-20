from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)))
        self.Tp = Tp
        self.states = []
        
        if len(state) < 3:
            self.x_hat = state[0]
        else:
            self.x_hat = state[0:2]

    def set_B(self, B):
        self.B = B

    def update(self, q, u):
        self.states.append(copy(self.state))
        ### TODO implement ESO update
        
        # Equation 42
        z_hat_dot = np.array([self.A @ np.array([self.state]).T]).flatten() + \
                            np.array([self.B @ [u]]).flatten() + \
                            np.array([self.L @ np.array([q - self.x_hat]).T]).flatten()

        z_hat = self.state + (z_hat_dot * self.Tp)

        if len(z_hat) < 4:
            self.x_hat = z_hat[0]
        else:
            self.x_hat = z_hat[0:2]
        self.state = z_hat

    def get_state(self):
        return self.state
