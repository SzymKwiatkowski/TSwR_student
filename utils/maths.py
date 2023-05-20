import numpy as np

from models.manipulator_model import ManipulatorModel

def x_dot(model : ManipulatorModel, x, u):
        invM = np.linalg.inv(model.M(x))
        zeros = np.zeros((2, 2), dtype=np.float32)
        A = np.concatenate([np.concatenate([zeros, np.eye(2)], 1), np.concatenate([zeros, -invM @ model.C(x)], 1)], 0)
        b = np.concatenate([zeros, invM], 0)
        return A @ x[:, np.newaxis] + b @ u