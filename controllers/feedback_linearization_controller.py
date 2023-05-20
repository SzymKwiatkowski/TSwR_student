import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp, m3=0.0, r3=0.01, Kd=100.0, Kp= 100.0):
        self.Tp = Tp
        self.model = ManipulatorModel(Tp, m3=m3, r3=r3)
        self.Kd = Kd
        self.Kp = Kp

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        # q, q_dot = np.hsplit(x[np.newaxis,:], 2)
        # q = np.transpose(np.array(q))
        # q_dot = np.transpose(np.array(q_dot))
        q = np.transpose(np.array([x[:2]]))
        q_dot = np.transpose(np.array([x[2:]]))
        M = self.model.M(x)
        C = self.model.C(x)
        q_r_ddot = np.transpose(np.array([q_r_ddot]))
        q_r_dot = np.transpose(np.array([q_r_dot]))
        q_r = np.transpose(np.array([q_r]))

        v = q_r_ddot - self.Kd * ( q_dot - q_r_dot ) - self.Kp * (q - q_r)

        tau = M @ v + C @ q_dot
        return tau
