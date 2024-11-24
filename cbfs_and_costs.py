import numpy as np

class CBF:
    """
    Control Barrier Function
    """
    def __init__(self):
        """
        Initialize CBF parameters.
        """
        self.beta = 1.0
        self.gamma = 0.9
        self.P = np.diag([1.0, 1.0])
        self.c = np.zeros((2, 1))
        self.c[0 , 0] = 1.125
        # self.P = np.diag([1.31, 4.00])
        # self.c = np.zeros((2, 1))
        # self.c[0 , 0] = 1.125

    def eval(self, state_x):
        """ 
        Evaluate CBF from one dimensional state array.
        """
        assert state_x.shape == (2, )
        state_x_m = state_x[:, np.newaxis]
        return self.beta - (state_x_m - self.c).T @ self.P @ (state_x_m - self.c)

    def dhdx(self, state_x):
        """
        Return CBF derivative.
        """
        assert state_x.shape == (2, )
        state_x_m = state_x.ravel() 
        state_x_m = state_x_m[:, np.newaxis]
        return -2*(state_x_m - self.c).T @ self.P
    
    def dhdx2(self, state_x):
        """
        Return CBF second derivative
        """
        return -2*self.P
    
    def lie_f(self, state_x, linear_sys):
        """
        Return lie derivative of CBF wrt control-independent dynamics.
        """
        return self.dhdx(state_x) @ linear_sys.A @ state_x
    
    def lie_g(self, state_x, linear_sys):
        """
        Return lie derivative of CBF wrt control-dependent dynamics.
        """
        return self.dhdx(state_x) @ linear_sys.B
    
    def apply_filter(self, state_x, u_perf, linear_sys):
        """
        Apply CBF filter.
        """
        print("State", state_x, state_x.shape)
        assert state_x.shape == (2,)
        lie_f_coeff = self.lie_f(state_x.ravel(), linear_sys).ravel()
        lie_g_coeff = self.lie_g(state_x.ravel(), linear_sys).ravel()
        if(abs(lie_g_coeff[0])<1e-2):
            print("Lie derivatives", lie_f_coeff, lie_g_coeff)
            #sys.exit(0)
        lie_f_coeff += self.gamma*self.eval(state_x.ravel()).ravel()
        if ((lie_g_coeff @ u_perf + lie_f_coeff)>0):
            return np.array(u_perf), lie_f_coeff, lie_g_coeff
        else:
            return np.array([-lie_f_coeff[0]/(lie_g_coeff[0] + 1e-6)]), lie_f_coeff, lie_g_coeff
