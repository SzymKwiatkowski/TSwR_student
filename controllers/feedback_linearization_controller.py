import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManipulatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """

        M = self.model.M(x)
        C = self.model.C(x)
        q_r_ddot = np.array([q_r_ddot])
        q_r_dot = np.array([q_r_dot])

        tau = M @ np.transpose(q_r_ddot) + C @ np.transpose(q_r_dot)
        return tau
