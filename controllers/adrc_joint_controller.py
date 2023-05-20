import numpy as np
from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        # Equations 48 - 51
        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]], dtype=np.float32)
        B = np.array([[0],
                      [self.b],
                      [0]], dtype=np.float32)
        L = np.array([[3 * (-p)],
                      [3 * ((-p)** 2)],
                      [(-p) ** 3]], dtype=np.float32)
        W = np.array([1, 0, 0])
        self.eso = ESO(A, B, W, L, q0, Tp)
        
        self.model = ManipulatorModel(Tp)
        self.init_model()
        

    def init_model(self):
        self.model.l1 = 0.5
        self.model.r1 = 0.04
        self.model.m1 = 3.
        self.model.l2 = 0.4
        self.model.r2 = 0.04
        self.model.m2 = 2.4
        self.model.I_1 = 1 / 12 * self.model.m1 * (3 * self.model.r1 ** 2 + self.model.l1 ** 2)
        self.model.I_2 = 1 / 12 * self.model.m2 * (3 * self.model.r2 ** 2 + self.model.l2 ** 2)
        self.model.m3 = 0.5
        self.model.r3 = 0.05
        self.model.I_3 = 2. / 5 * self.model.m3 * self.model.r3 ** 2
        self.model.alfa = (self.model.m1 * ((self.model.l1 / 2) ** 2)) + self.model.I_1 + \
                    (self.model.m2 * ((self.model.l1 ** 2) + ((self.model.l2 / 2) ** 2))) + \
                    self.model.I_2 + (self.model.m3 * ((self.model.l1 ** 2) +  \
                    (self.model.l2 ** 2))) + self.model.I_3
        self.model.beta = (self.model.m2 * self.model.l1 * (self.model.l2 / 2)) + \
                    (self.model.m3 * self.model.l1 * self.model.l2)
        self.model.gamma = (self.model.m2 * ((self.model.l2 / 2) ** 2)) + \
                        self.model.I_2 + (self.model.m3 * (self.model.l2 ** 2)) + self.model.I_3
        
    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.b = b
        B = np.array([[0],
                      [b],
                      [0]], dtype=np.float32)
        
        self.eso.set_B(B)
    
    def calculate_b_hat(self, i, x_hat):
        M = np.array([[self.model.alfa + (2 * self.model.beta * np.cos(x_hat)), 
                       self.model.gamma + (self.model.beta * np.cos(x_hat))],
                      [self.model.gamma + (self.model.beta * np.cos(x_hat)), 
                       self.model.gamma]])

        # equation 53
        M_inv = np.linalg.inv(M)
        return M_inv[i, i]

    def calculate_control(self, i, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement ADRC
        x_hat, x_hat_dot, f = self.eso.get_state()
        v = q_d_ddot + self.kd * (q_d_dot - x_hat_dot) + self.kp * (q_d - x[0])

        # Equation 37
        u = (v - f)/self.b
        self.eso.update(x[0], u)
        self.set_b(self.calculate_b_hat(i, x_hat))

        return u
