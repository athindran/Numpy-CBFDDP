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
        self.P = np.diag([1.0, 1.5, 0.0, 0.0])
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

class MultiCBF_b:
    """
    Control Barrier Function
    """
    def __init__(self, kappa=2.0, use_smoothening=False):
        """
        Initialize CBF parameters.
        """
        self.beta = 0.5
        self.shift = np.array([[0.2], [1.4], [0.0], [0.0]])
        self.gamma = 0.9
        self.P = np.diag([1.0, 1.5, 0.0, 0.0])
        self.c = np.zeros((4, 1))
        self.c[0 , 0] = 1.0

    def eval(self, state_x):
        """ 
        Evaluate CBF from one dimensional state array.
        """
        assert state_x.shape == (4, )
        state_x_m = state_x[:, np.newaxis]

        h1 = (state_x_m - self.c).T @ self.P @ (state_x_m - self.c) - self.beta
        h2 = (state_x_m - (self.c + self.shift)).T @ self.P @ (state_x_m - (self.c + self.shift)) - self.beta

        if self.use_smoothening:
            hsm = (-np.log(np.exp(-self.kappa*h1) + np.exp(-self.kappa*h2)))/self.kappa
            return hsm
        else:
            return np.minimum(h1, h2)

    def eval_idx(self, state_x):
        """ 
        Evaluate CBF from one dimensional state array.
        """
        assert state_x.shape == (4, )
        state_x_m = state_x.ravel() 
        state_x_m = state_x_m[:, np.newaxis]
        h1 = (state_x_m - self.c).T @ self.P @ (state_x_m - self.c) - self.beta
        h2 = (state_x_m - (self.c + self.shift)).T @ self.P @ (state_x_m - (self.c + self.shift)) - self.beta
        if h1<h2:
            return 0
        else:
            return 1

    def dhdx(self, state_x):
        """
        Return CBF derivative.
        """
        assert state_x.shape == (4, )
        state_x_m = state_x.ravel() 
        state_x_m = state_x_m[:, np.newaxis]
        
        dh1dx = 2*(state_x_m - self.c).T @ self.P
        dh2dx = 2*(state_x_m - self.c - self.shift).T @ self.P
        if self.use_smoothening:
            h1 = (state_x_m - self.c).T @ self.P @ (state_x_m - self.c) - self.beta
            h2 = (state_x_m - (self.c + self.shift)).T @ self.P @ (state_x_m - (self.c + self.shift)) - self.beta

            denominator = (np.exp(-self.kappa*h1) + np.exp(-self.kappa*h2))
            dhsmoothdh1 = np.exp(-self.kappa*h1) / denominator
            dhsmoothdx = dhsmoothdh1 * dh1dx 
            dhsmoothdh2 = np.exp(-self.kappa*h2) / denominator
            dhsmoothdx += dhsmoothdh2 * dh2dx

            return dhsmoothdx

        idx = self.eval_idx(state_x)
    
        if idx==0:
            return dh1dx
        elif idx==1:
            return dh2dx

    def dhdx2(self, state_x):
        """
        Return CBF second derivative
        """
        assert state_x.shape == (4, )
        state_x_m = state_x.ravel() 
        state_x_m = state_x_m[:, np.newaxis]

        if self.use_smoothening:
            dh1dx = 2*(state_x_m - self.c).T @ self.P
            dh2dx = 2*(state_x_m - self.c - self.shift).T @ self.P
            h1 = (state_x_m - self.c).T @ self.P @ (state_x_m - self.c) - self.beta
            h2 = (state_x_m - (self.c + self.shift)).T @ self.P @ (state_x_m - (self.c + self.shift)) - self.beta
            
            denominator = (np.exp(-self.kappa*h1) + np.exp(-self.kappa*h2))
            dhsmoothdh1 = np.exp(-self.kappa*h1) / denominator
            dhsmoothdh2 = np.exp(-self.kappa*h2) / denominator

            d2hsmoothdh1dx = self.kappa * dhsmoothdh1 * dh1dx - self.kappa * dhsmoothdh1 * dh1dx / denominator
            d2hsmoothdh2dx = self.kappa * dhsmoothdh2 * dh2dx - self.kappa * dhsmoothdh2 * dh2dx / denominator

            d2hsmoothdx2 = 2 * self.P * (dhsmoothdh1 + dhsmoothdh2) 
            d2hsmoothdx2 += np.outer(d2hsmoothdh1dx, dh1dx) + np.outer(d2hsmoothdh2dx, dh2dx)

            return d2hsmoothdx2

        return 2*self.P

