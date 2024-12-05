import numpy as np
import time
from typing import Tuple
import copy

import gym
from cbfs_and_costs import CBF
import cvxpy as cp
from cvxpy import SolverError

def barrier_filter_quadratic_one(P, p, c):
    def is_neg_def(x):
        # Check if a matrix is PSD
        return np.all(np.real(np.linalg.eigvals(x)) <= 0)

    # CVX faces numerical difficulties otherwise
    check_nd = (P < 0)
    # Check if P is PD
    if (check_nd):
        u = cp.Variable((1))
        P = np.array(P)
        p = np.array(p)

        prob = cp.Problem(cp.Minimize(1.0 * cp.square(u[0])),
                          [cp.quad_form(u, P) + p.T @ u + c >= 0])
        try:
            prob.solve(verbose=False)
        except SolverError:
            pass

    if (not check_nd or u[0] is None or prob.status not in [
            "optimal", "optimal_inaccurate"]):
        u = cp.Variable((1))
        prob = cp.Problem(cp.Minimize(1.0 * cp.square(u[0])),
                          [p @ u + c >= 0])
        try:
            prob.solve(verbose=False)
        except SolverError:
            pass

    if prob.status not in ["optimal", "optimal_inaccurate"] or u[0] is None:
        return np.array([0.])
    return np.array([u[0].value])


class ReachabilityLQPolicy:
    def __init__(self, state_dim: int, action_dim:int, marginFunc: CBF, env: gym.Env, horizon = 50, Rc=1e-5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.marginFunc =  marginFunc
        self.horizon = horizon
        self.env = env
        
        self.R = np.diag(Rc* np.ones((env.action_dim, )))

        self.tol = 1e-6
        self.eps = 1e-5
        self.max_iters = 30
        self.line_search = "baseline"
    
    def initialize_trajectory(self, obs: np.ndarray, nominal_controls:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nominal_states = np.zeros((self.horizon + 1, self.state_dim))
        nominal_states[0] = np.array( obs )
        
        for t in range(self.horizon):
            obs, nominal_controls[t] = self.env.step( obs, nominal_controls[t] )
            nominal_states[t+1] = np.array( obs )
        
        return nominal_states, nominal_controls
    
    def forward_pass(self, nominal_states: np.ndarray, nominal_controls: np.ndarray, 
                     K_closed_loop:np.ndarray, k_open_loop:np.ndarray, alpha=1.0) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray, int]:
        new_states = np.array(nominal_states)
        new_controls = np.array(nominal_controls)

        obs = np.array( nominal_states[0] )
        for t in range(self.horizon):
            new_controls[t] = new_controls[t] + K_closed_loop[t] @ (obs - nominal_states[t]) + alpha*k_open_loop[t]
            obs, new_controls[t] = self.env.step( obs, new_controls[t] )
            new_states[t+1] = np.array( obs )
        
        reachable_margin, critical_margin, state_margins, critical_index = self.get_margin(new_states, new_controls)
        
        return new_states, new_controls, reachable_margin, critical_margin, state_margins, critical_index
    
    def get_margin(self, nominal_states, nominal_controls):        
        reachable_margin = np.inf
        critical_margin = np.inf

        state_margins = np.zeros((self.horizon, ))
        critical_index = -1
        
        t = self.horizon - 1
        # Find margin over truncated horizon
        while t>=0:
            obs_curr = nominal_states[t:t+1]
            
            failure_margin = self.marginFunc.eval(obs_curr.ravel())
            state_margins[t] = failure_margin

            if(failure_margin<reachable_margin):
                critical_index = t
                reachable_margin = failure_margin
                critical_margin = failure_margin

            reachable_margin = reachable_margin - 0.5*nominal_controls[t] @ self.R @ nominal_controls[t]
            t = t - 1
        return reachable_margin, critical_margin, state_margins, critical_index
        
    def backward_pass(self, nominal_states, nominal_controls):
        # Perform an ILQ backward pass
        V_x = np.zeros((self.horizon + 1, self.state_dim))
        V_xx = np.zeros((self.horizon + 1, self.state_dim, self.state_dim))
        
        Q_x = np.zeros((self.horizon, self.state_dim))
        Q_u = np.zeros((self.horizon, self.action_dim))
        Q_xx = np.zeros((self.horizon, self.state_dim, self.state_dim))
        Q_uu = np.zeros((self.horizon, self.action_dim, self.action_dim))
        Q_ux = np.zeros((self.horizon, self.action_dim, self.state_dim))
        
        margins = np.zeros((self.horizon, ))
        state_margins = np.zeros((self.horizon, ))
                
        k_open_loop = np.zeros((self.horizon, self.action_dim))
        K_closed_loop = np.zeros((self.horizon, self.action_dim, self.state_dim))
        
        index_lists = []
        
        # Initialize reachability margin to infinity
        reachable_margin = np.inf
        
        # Backward pass
        t = self.horizon - 1
        index_lists = np.zeros((self.horizon,), dtype=np.int32)
        while t>=0:
            obs_curr = nominal_states[t:t+1]
            action_curr = nominal_controls[t:t+1]
            
            failure_margin = self.marginFunc.eval(obs_curr.ravel()) 
            
            if(failure_margin<reachable_margin):
                index_lists[self.horizon - 1 - t] = -1
                reachable_margin = failure_margin
            else:
                index_lists[self.horizon - 1 - t] = 0
            reachable_margin = reachable_margin - 0.5*nominal_controls[t] @ self.R @ nominal_controls[t]
            margins[t] = reachable_margin
            state_margins[t] = failure_margin
            t = t - 1
        
        t = self.horizon - 1
        while t>=0:
            current_index = index_lists[self.horizon - 1 - t]
            obs_curr = nominal_states[t:t+1]
            action_curr = nominal_controls[t:t+1]

            # Failure derivatives            
            c_x_failure = self.marginFunc.dhdx(obs_curr.ravel())
            c_xx_failure = self.marginFunc.dhdx2(obs_curr.ravel())
            Ad, Bd, _, _ = self.env.get_jacobian(obs_curr.ravel(), action_curr.ravel())
            
            if(current_index == -1):
                Q_x[t] = c_x_failure
                Q_xx[t] = c_xx_failure
                Q_u[t] = -self.R@nominal_controls[t] + Bd.T @ V_x[t+1]
                Q_ux[t] = Bd.T @ V_xx[t+1] @ Ad 
                Q_uu[t] = -self.R@np.eye(action_curr.size) + Bd.T @ V_xx[t+1] @ Bd
                Q_uu_delta = -self.R@np.eye(action_curr.size) + Bd.T @ ( V_xx[t+1] - self.eps*np.eye(self.state_dim) ) @ Bd
                Q_ux_delta = Bd.T @ ( V_xx[t+1] - self.eps*np.eye(self.state_dim) ) @ Ad
            else:
                Q_x[t] = Ad.T @ V_x[t+1]
                Q_xx[t] = Ad.T @ V_xx[t+1] @ Ad
                Q_u[t] =  -self.R@nominal_controls[t] + Bd.T @ V_x[t+1]
                Q_ux[t] = Bd.T @ V_xx[t+1] @ Ad  
                Q_uu[t] = -self.R@np.eye(action_curr.size) + Bd.T @ V_xx[t+1] @ Bd
                Q_uu_delta = -self.R@np.eye(action_curr.size) + Bd.T @ ( V_xx[t+1] - self.eps*np.eye(self.state_dim) ) @ Bd
                Q_ux_delta = Bd.T @ ( V_xx[t+1] - self.eps*np.eye(self.state_dim) ) @ Ad

            Q_uu_inv = np.linalg.inv( Q_uu_delta )
            
            # Signs for maximization
            k_open_loop[t] =  - Q_uu_inv @ Q_u[t]
            K_closed_loop[t] =  - Q_uu_inv @ Q_ux_delta
            
            # Update value function derivative for the previous time step
            if(current_index == -1):
                V_x_critical = c_x_failure
                V_xx_critical = c_xx_failure
                V_x[t] = Q_x[t] + Q_ux[t].T @ k_open_loop[t]
                V_xx[t] = Q_xx[t] + Q_ux[t].T @ K_closed_loop[t]
            else:
                V_x[t] = Q_x[t] + Q_ux[t].T @ k_open_loop[t]
                V_xx[t] = Q_xx[t] + Q_ux[t].T @ K_closed_loop[t]
            t = t - 1
        
        self.Q_u = Q_u 
        barrier_constraint_data = {"V_xx": V_xx[0], "V_x": V_x[0], "V_xx_critical": V_xx_critical, 
                                   "V_x_critical": V_x_critical, "V_t": margins[0], "state_margins": state_margins}        
            
        return K_closed_loop, k_open_loop, barrier_constraint_data
    
    def get_action(self, initial_state, initial_controls=None):
        start_time = time.time()
        # Get initial trajectory with naive controls
        if initial_controls is None:
            initial_controls = np.zeros((self.horizon, self.action_dim))
            initial_controls[:, 0] = 0.01
        
        states, controls = self.initialize_trajectory(initial_state, initial_controls)

        # Update control with ILQ updates
        iters = 0
        J, critical_margin, _, _ = self.get_margin(states, controls)
        
        converged = False
        status = 0
        updated_constraints_data = None

        self.margin = 5.0
        while iters<self.max_iters and not converged:
            iters = iters + 1
            # Backward pass
            K_closed_loop, k_open_loop, _ = self.backward_pass(states, controls)
            
            # Choose the best alpha scaling using appropriate line search methods
            alpha_chosen = self.baseline_line_search( states, controls, K_closed_loop, k_open_loop, critical_margin, J )
            
            if alpha_chosen<1e-13:
                J_new = J
                break

            states, controls, J_new, critical_margin, _, _ = self.forward_pass(states, controls, K_closed_loop, 
                                                                                                        k_open_loop, alpha_chosen)


            if np.abs((J_new-J) / J) < self.tol:  # Small improvement.
                converged = True
            J = J_new

        
        # Backward pass
        _, _, updated_constraints_data = self.backward_pass(states, controls)
        
        process_time = time.time() - start_time
        solver_dict = {"states": states, "controls": controls, "status": status, "margin": critical_margin, "reachable_margin": J, 
                       "iteration_no": iters, "critical_constraint_type": 0,
                       "label": "Optimal safety plan", "id": 'Optimal', 
                       "t_process": process_time, "iterations": iters}
        return controls[0], solver_dict, updated_constraints_data

    def baseline_line_search( self, states, controls, K_closed_loop, k_open_loop, critical, J, beta=0.3 ):
        alpha = 1.0
        while alpha>1e-13:
            _, _, J_new, _, _, _ = self.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha)    

            # Accept if there is improvement
            margin_imp = (J_new - J)
            if margin_imp > 0.:
                return alpha
            alpha = beta*alpha

        return alpha
    
class DDPLRFilter:
    def __init__(self, state_dim: int, action_dim: int, marginFunc: CBF, env: gym.Env, horizon = 50, Rc=1e-5):
        self.rollout_policy_1 =  ReachabilityLQPolicy(state_dim, action_dim, marginFunc, copy.deepcopy(env), horizon, Rc)
        self.rollout_policy_2 =  ReachabilityLQPolicy(state_dim, action_dim, marginFunc, copy.deepcopy(env), horizon, Rc)
        self.reinit_controls = np.zeros((horizon, action_dim))

    def apply_filter(self, state_x, u_perf, linear_sys):
        dyn_copy = copy.deepcopy(linear_sys)
        control_safe_1, solver_dict_plan_1, constraints_data_plan_1 = self.rollout_policy_1.get_action( 
            np.array( state_x ) , initial_controls=self.reinit_controls)
        next_state_x, _ = dyn_copy.step(state_x, u_perf)

        boot_controls = np.array( solver_dict_plan_1['controls'] )
        boot_controls[0:-1] = boot_controls[1:]
        _, solver_dict_plan_2, constraints_data_plan_2 = self.rollout_policy_2.get_action( np.array( next_state_x ) , 
                                                                                          initial_controls=boot_controls)
        boot_controls = np.array(solver_dict_plan_2['controls'])

        if(solver_dict_plan_2['reachable_margin']>0.0):
            filtered_control = np.array( u_perf )
            self.reinit_controls = np.array( solver_dict_plan_2['controls'] )
        else:
            print("Filter active")
            filtered_control = np.array(control_safe_1)
            self.reinit_controls = np.array( boot_controls )
        
        return filtered_control, solver_dict_plan_1["margin"]
    
class DDPCBFFilter:
    def __init__(self, state_dim: int, action_dim: int, marginFunc: CBF, env: gym.Env, horizon: int, Rc: float, gamma: float):
        self.rollout_policy_1 =  ReachabilityLQPolicy(state_dim=state_dim, action_dim=action_dim, marginFunc=marginFunc, 
                                                      env=copy.deepcopy(env), horizon=horizon, Rc=Rc)
        self.rollout_policy_2 =  ReachabilityLQPolicy(state_dim=state_dim, action_dim=action_dim, marginFunc=marginFunc, 
                                                      env=copy.deepcopy(env), horizon=horizon, Rc=Rc)
        self.reinit_controls = np.zeros((horizon, action_dim))
        self.gamma = gamma

    def apply_filter(self, state_x, u_perf, linear_sys):
        dyn_copy = copy.deepcopy(linear_sys)
        control_safe_1, solver_dict_plan_1, constraints_data_plan_1 = self.rollout_policy_1.get_action( 
            np.array( state_x ) , initial_controls=self.reinit_controls)
        next_state_x, _ = dyn_copy.step(state_x, u_perf)

        boot_controls = np.array( solver_dict_plan_1['controls'] )
        boot_controls[0:-1] = boot_controls[1:]
        _, solver_dict_plan_2, constraints_data_plan_2 = self.rollout_policy_2.get_action( np.array( next_state_x ) , 
                                                                                          initial_controls=boot_controls)
        boot_controls = np.array(solver_dict_plan_2['controls'])

        control_cbf = np.array(u_perf)

        _, Bd, _, _ = dyn_copy.get_jacobian(state_x, control_cbf)

        eps_regularization = 1e-6

        constraint_violation = solver_dict_plan_2['reachable_margin'] - self.gamma*solver_dict_plan_1['reachable_margin']
        scaling_factor = 0.5
        scaled_c = scaling_factor*constraint_violation 
        p = Bd.T @ constraints_data_plan_2['V_x']
        P = -eps_regularization * \
                np.eye(self.rollout_policy_1.action_dim) + 0.5 * Bd.T @ constraints_data_plan_2['V_xx'] @ Bd
        p_norm = np.linalg.norm(p)
        barrier_entries = 0
        while constraint_violation<=0 and barrier_entries<5:
            barrier_entries += 1
            #control_correction = -p*scaled_c/((p_norm)**2 + 1e-12)
            control_correction = barrier_filter_quadratic_one(P, p, scaled_c)
            control_cbf = control_cbf + control_correction
            control_cbf_new = control_cbf + control_correction
            control_cbf_new = control_cbf_new.ravel()
            #control_cbf_new = np.clip(control_cbf_new, -2.0, 2.0)
            # Testing barrier quality
            imag_state_x, control_cbf_new = dyn_copy.step( state_x, control_cbf_new )
            _, solver_dict_plan_3, constraints_data_plan_3 = self.rollout_policy_2.get_action( np.array( imag_state_x ) , 
                                                                                            initial_controls=boot_controls)
            solver_dict_plan_2 = solver_dict_plan_3
            constraints_data_plan_2 = constraints_data_plan_3
            control_cbf = control_cbf_new

            _, Bd, _, _  = dyn_copy.get_jacobian(state_x, control_cbf)
            p = Bd.T @ constraints_data_plan_2['V_x']
            P = -eps_regularization * \
                np.eye(self.rollout_policy_1.action_dim) + Bd.T @ constraints_data_plan_2['V_xx'] @ Bd
            p_norm = np.linalg.norm( p )
            constraint_violation = solver_dict_plan_2['reachable_margin'] - self.gamma*(solver_dict_plan_1['reachable_margin'])
            boot_controls = solver_dict_plan_2['controls']
            scaled_c = 1.3*constraint_violation 
        
        print("Constraint violation", constraint_violation)
        #if solver_dict_plan_2["reachable_margin"]<=0.0:
        #    return control_safe_1.ravel(),solver_dict_plan_1["reachable_margin"], constraint_violation, p.copy()
        #else:
        return control_cbf.ravel(), solver_dict_plan_1["reachable_margin"], constraint_violation, p.copy(), barrier_entries