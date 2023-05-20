import numpy as np
from .controller import Controller
from models.manipulator_model import ManipulatorModel
from controllers.feedback_linearization_controller import FeedbackLinearizationController
import utils.maths as mts
import logging


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.current_model_id = 0
        
        self.controllers = [FeedbackLinearizationController(Tp=Tp, m3=0.1, r3=0.05, Kd=100.0, Kp=100.0),
                            FeedbackLinearizationController(Tp=Tp, m3=0.01, r3=0.01, Kd=100.0, Kp=100.0),
                            FeedbackLinearizationController(Tp=Tp, m3=1.0, r3=0.3, Kd=40.0, Kp=40.0)]
        self.i = 0
        self.Kd = np.array([[10.0, 0]])
        self.Kp = np.array([[.2, 0]])

    def choose_model(self, errors):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        self.i = 0 # Default model is that with ID = 0
        minimum_error_id = np.argmin(errors)
        if minimum_error_id < len(self.controllers) and self.i != minimum_error_id:
            self.i = minimum_error_id
        
        if self.current_model_id != self.i:
            print('Model switched to: ', str(self.i))
            # print("{0:{fill}<10}{1}{0:{fill}<10}".format("#"," Model Selector ", fill='#'))
            # print("{0:{fill}<10}{1}{2} {0:{fill}<10}".format("#","Choose model:=", self.i, fill='#'))
            self.current_model_id = self.i

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        errors = []
        control_signals = []
        for i in range(len(self.controllers)):
            u = self.controllers[i].calculate_control(x, q_r, q_r_dot, q_r_ddot)
            y_hat = mts.x_dot(self.controllers[i].model, x ,u)
            errors.append( x[:, np.newaxis] - y_hat )
            control_signals.append(u)
        
        self.choose_model(errors)
        
        return control_signals[self.i]
