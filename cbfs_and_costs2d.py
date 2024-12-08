import numpy as np

class CBF:
    """
    Control Barrier Function
    """
    def __init__(self):
        """
        Initialize CBF parameters.
        """
        self.beta = 0.5
        self.gamma = 0.9
        self.P = np.diag([1.0, 1.0, 0.0, 0.0])
        self.c = np.zeros((4, 1))
        self.c[0 , 0] = 1.125
        # self.P = np.diag([1.31, 4.00])
        # self.c = np.zeros((2, 1))
        # self.c[0 , 0] = 1.125

    def eval(self, state_x):
        """ 
        Evaluate CBF from one dimensional state array.
        """
        assert state_x.shape == (4, )
        state_x_m = state_x[:, np.newaxis]
        return (state_x_m - self.c).T @ self.P @ (state_x_m - self.c) - self.beta

    def dhdx(self, state_x):
        """
        Return CBF derivative.
        """
        assert state_x.shape == (4, )
        state_x_m = state_x.ravel() 
        state_x_m = state_x_m[:, np.newaxis]
        return 2*(state_x_m - self.c).T @ self.P
    
    def dhdx2(self, state_x):
        """
        Return CBF second derivative
        """
        return 2*self.P

