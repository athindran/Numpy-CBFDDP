from scipy.integrate import solve_ivp

import numpy as np
import gym

class LinearSys(gym.Env):
    """
    Linear System Environment
    """
    def __init__(self):
        """
        Initializes the state dimension, action dimension.
        A matrix, B matrix and sampling time.
        """
        super(LinearSys, self).__init__()
        self.state_dim = 2
        self.action_dim = 1
        self.A = np.array([[0.0, 1.0], [-0.09, 0.1]])
        self.B = np.array([[0.0],[18.09]])
        self.dt = 0.001

    def step(self, obs, action=np.zeros((1, ))):
        """
        Integrates for one ODE step.
        Action should be a one dimensional array of size action_dim.
        Function additionally integrates internal state.
        """
        assert action.shape[0]==self.action_dim
        # ODE solver
        ode_out = solve_ivp(fun=self.deriv_vec, y0=obs.ravel(), args=(action[np.newaxis, :]), t_span=(0, self.dt))
        new_obs = ode_out.y[:, -1]
        self.state_x = np.array(new_obs)
        return new_obs, action
    
    def get_jacobian(self, obs, action):
        return np.eye(self.state_dim)+self.dt*self.A, self.dt*self.B, self.A, self.B
    
    def get_obs(self):
        """
        Return internal state.
        """
        return self.state_x
    
    def deriv_vec(self, t, state_x, u):
        """
        Return system forward propagation derivative.
        """
        return (self.A @ state_x + self.B @ u)
    
    def reset(self):
        """
        Reset to initial state.
        """
        self.state_x = np.array([[0.5, -0.1]])
