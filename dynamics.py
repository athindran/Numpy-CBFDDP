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
        self.A = np.array([[0.0, 1.0], [0.0, 0.0]])
        self.B = np.array([[0.0],[1.0]])
        self.dt = 0.01

    def step(self, obs, action=np.zeros((1, ))):
        """
        Integrates for one ODE step.
        Action should be a one dimensional array of size action_dim.
        Function additionally integrates internal state.
        """
        assert action.shape[0]==self.action_dim
        action = np.clip(action, a_min=-2.0, a_max=2.0)
        # ODE solver
        ode_out = solve_ivp(fun=self.deriv_vec, y0=obs.ravel(), args=(action[np.newaxis, :]), t_span=(0, self.dt))
        new_obs = ode_out.y[:, -1]
        return new_obs, action
    
    def get_jacobian(self, obs, action):
        return np.eye(self.state_dim)+self.dt*self.A, self.dt*self.B, self.A, self.B
    
    def deriv_vec(self, t, state_x, u):
        """
        Return system forward propagation derivative.
        """
        return (self.A @ state_x + self.B @ u)
    
    def reset(self, cbf_type):
        """
        Reset to initial state.
        """
        if cbf_type == 'A':
            return np.array([[1.0, 0.0]])
        elif cbf_type == 'B' or cbf_type=='C':
            return np.array([[-0.8, 1.5]])


class DoubleIntegrator2D(gym.Env):
    """
    Linear System Environment
    """
    def __init__(self):
        """
        Initializes the state dimension, action dimension.
        A matrix, B matrix and sampling time.
        """
        super(DoubleIntegrator2D, self).__init__()
        self.state_dim = 4
        self.action_dim = 2
        self.A = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        self.B = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        self.dt = 0.01

    def step(self, obs, action=np.zeros((2, ))):
        """
        Integrates for one ODE step.
        Action should be a one dimensional array of size action_dim.
        Function additionally integrates internal state.
        """
        assert action.shape[0]==self.action_dim
        action[0] = np.clip(action[0], a_min=-1.0, a_max=1.0)
        action[1] = np.clip(action[1], a_min=-3.0, a_max=3.0)
        # ODE solver
        #print(obs, action)
        ode_out = solve_ivp(fun=self.deriv_vec, y0=obs.ravel(), args=(action[np.newaxis, :]), t_span=(0, self.dt))
        new_obs = ode_out.y[:, -1]
        return new_obs, action
    
    def get_jacobian(self, obs, action):
        return np.eye(self.state_dim)+self.dt*self.A, self.dt*self.B, self.A, self.B
    
    def deriv_vec(self, t, state_x, u):
        """
        Return system forward propagation derivative.
        """
        #print(self.B.shape, u.shape)
        return (self.A @ state_x + self.B @ u)
    
    def reset(self, cbf_type):
        """
        Reset to initial state.
        """
        if cbf_type == 'A':
            return np.array([[-1.0, 0.1, 1.5, 0.0]])
        elif cbf_type == 'B' or cbf_type=='C':
            return np.array([[-1.0, 0.1, 1.5, 0.0]])
