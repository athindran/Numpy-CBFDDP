import sys
sys.path.append(".")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
import numpy as np
import copy

from cbfs_and_costs import CBF
from policies import ReachabilityLQPolicy, DDPCBFFilter
from dynamics import LinearSys

linear_sys = LinearSys()
linear_sys.reset()

cbf_type = 'A'
if cbf_type == 'A':
    cbf = CBF()
# elif cbf_type == 'B':
#     cbf = CBF_b()
# elif cbf_type == 'C':
#     cbf = CBF_c()
else:
    cbf = None

T = 550

def run_simulation(linear_sys, cbf, method=None, R=None, horizon=None, gamma=None):
    linear_sys.reset()
    action_perf = np.array([-1.0])

    simulation_states = np.zeros((2, T))
    cbf_states = np.zeros((T, ))
    controls = np.zeros((T, ))
    solver_types = np.zeros((T, ))
    lie_f_vals = np.zeros((T, ))
    lie_g_vals = np.zeros((T, ))

    if method == 'ddpcbf':
        ddpcbf = DDPCBFFilter(2, 1, copy.deepcopy(cbf), copy.deepcopy(linear_sys), 
                                horizon=horizon, Rc=R)

    obs = linear_sys.get_obs()
    for idx in range(T):
        if method == 'unfilter':
            action_filtered = action_perf
        elif method == 'hcbf':
            action_filtered, lie_f, lie_g = cbf.apply_filter(obs.ravel(), action_perf, linear_sys)
            lie_f_vals[idx] = lie_f.ravel()[0]
            lie_g_vals[idx] = lie_g.ravel()[0]
        elif method == 'ddpcbf':
            action_filtered, ddp_cbf_eval, lie_f, lie_g, barrier_entries = ddpcbf.apply_filter(obs.ravel(), action_perf, linear_sys)
            solver_types[idx] = barrier_entries
            lie_f_vals[idx] = lie_f.ravel()[0]
            lie_g_vals[idx] = lie_g.ravel()[0]

        new_obs, action = linear_sys.step(obs, action=action_filtered)
        obs = linear_sys.get_obs()

        cbf_eval = cbf.eval(new_obs)

        simulation_states[:, idx] = np.array(obs)
        cbf_states[idx] = cbf_eval.ravel()[0]
        controls[idx] = action_filtered.copy().ravel()[0]

        print(f"Step: {idx}, Obs: {new_obs}, Action: {action}, CBF eval: {cbf_eval}")
        #if(cbf_eval<1e-2):
        #    action_perf = np.array([0.1])

    #cbf_states[idx+1:] = cbf_states[idx]
    #simulation_states[:, idx+1:] = np.repeat(simulation_states[:, idx:idx+1], T - idx - 1, axis=1)
    runtime = T

    output_dict = {'simulation_states': simulation_states,
                    'cbf_states': cbf_states,
                    'runtime': runtime,
                    'controls': controls,
                    'lie_f': lie_f_vals,
                    'lie_g': lie_g_vals, 
                    'solver_types': solver_types}

    return output_dict

unconstrained_dict = run_simulation(linear_sys, cbf, method='unfilter')
unconstrained_simulation_states = unconstrained_dict['simulation_states']
unconstrained_cbf_states = unconstrained_dict['cbf_states']
unconstrained_runtime = unconstrained_dict['runtime']
unconstrained_controls = unconstrained_dict['controls']


constrained_dict = run_simulation(linear_sys, cbf, method='hcbf')
constrained_simulation_states = constrained_dict['simulation_states']
constrained_cbf_states = constrained_dict['cbf_states']
constrained_runtime = constrained_dict['runtime']
constrained_controls = constrained_dict['controls']

ddpcbf_dict = run_simulation(linear_sys, cbf, method='ddpcbf', R=1e-2, horizon=5, gamma=0.99)
ddpcbf_simulation_states = ddpcbf_dict['simulation_states']
ddpcbf_cbf_states = ddpcbf_dict['cbf_states']
ddpcbf_runtime = ddpcbf_dict['runtime']
ddpcbf_controls = ddpcbf_dict['controls']
ddpcbf_solver_types = ddpcbf_dict['solver_types']

ftsize=8
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 3))
#axes[0].plot(np.arange(0, T)*linear_sys.dt, unconstrained_cbf_states, label='Unfiltered')
axes[0].plot(np.arange(0, constrained_runtime)*linear_sys.dt, constrained_cbf_states[0:constrained_runtime], label='HCBF-Filtered')
axes[0].plot(np.arange(0, ddpcbf_runtime)*linear_sys.dt, ddpcbf_cbf_states[0:ddpcbf_runtime], label='DDP-CBF Filtered')
axes[0].set_xlabel('Time (s)', fontsize=ftsize)
axes[0].set_ylabel('CBF Value', fontsize=ftsize)
axes[0].legend(fontsize=ftsize)
xticks = np.round(np.linspace(0, T*linear_sys.dt, 4), 2)
axes[0].set_xticks(ticks=xticks, labels=xticks)
yticks = np.round(np.linspace(0.0, constrained_cbf_states.max(), 4), 2)
axes[0].set_yticks(ticks=yticks, labels=yticks)
axes[0].tick_params(labelsize=ftsize)
#axes[0].set_xlim([0, constrained_runtime*linear_sys.dt])
#axes[0].set_ylim([-0.1, round(constrained_cbf_states.max(), 2)])
axes[0].grid()

#axes[0, 1].plot(np.arange(0, T)*linear_sys.dt, unconstrained_controls, label='UnFiltered')
axes[1].plot(np.arange(0, constrained_runtime)*linear_sys.dt, constrained_controls[0:constrained_runtime], label='HCBF-Filtered')
axes[1].plot(np.arange(0, ddpcbf_runtime)*linear_sys.dt, ddpcbf_controls[0:ddpcbf_runtime], label='CBFDDP-Filtered')
axes[1].fill_between(np.arange(0, ddpcbf_runtime)*linear_sys.dt, -1.0, 1.0, where=(ddpcbf_solver_types>0), color='b', alpha=0.2)
axes[1].set_xlabel('Time (s)', fontsize=ftsize)
axes[1].set_ylabel('Controls', fontsize=ftsize)
#axes[1].legend(fontsize=ftsize)
xticks = np.round(np.linspace(0, T*linear_sys.dt, 4), 2)
axes[1].set_xticks(ticks=xticks, labels=xticks)
yticks = np.round(np.linspace(-1.0, 1.0, 4), 2)
axes[1].set_yticks(ticks=yticks, labels=yticks)
axes[1].tick_params(labelsize=ftsize)
axes[1].grid()

axes[2].plot(constrained_simulation_states[0, :], constrained_simulation_states[1, :], label='HCBF-Filtered')
axes[2].plot(ddpcbf_simulation_states[0, :], ddpcbf_simulation_states[1, :], label='CBFDDP-Filtered')
ellipse = Ellipse(xy=cbf.c, width=2*cbf.beta, height=2*cbf.beta, 
                        edgecolor='g', fc='None', lw=2)
axes[2].add_patch(ellipse)
#ellipse_pair = Ellipse(xy=cbf.c + cbf.shift, width=2*cbf.beta, height=2*cbf.beta, 
#                        edgecolor='g', fc='None', lw=2)
#axes[2].add_patch(ellipse_pair)
axes[2].set_xlabel('X axis', fontsize=ftsize)
axes[2].set_ylabel('Y axis', fontsize=ftsize)
axes[2].grid()

# axes[1, 0].plot(np.arange(0, constrained_runtime)*linear_sys.dt, constrained_lie_f[0:constrained_runtime], label='Filtered')
# axes[1, 0].plot(np.arange(0, ddpcbf_runtime)*linear_sys.dt, ddpcbf_lie_f[0:ddpcbf_runtime], label='CBFDDP-Filtered')
# axes[1, 0].set_xlabel('Time (s)', fontsize=ftsize)
# axes[1, 0].set_ylabel('CBF Lie f', fontsize=ftsize)
# #axes[1, 0].legend(fontsize=ftsize)
# #axes[1, 0].set_xticks(ticks=[0, T*linear_sys.dt], labels=[0, T*linear_sys.dt])
# axes[1, 0].tick_params(labelsize=ftsize)
# axes[1, 0].grid()

# axes[1, 1].plot(np.arange(0, constrained_runtime)*linear_sys.dt, constrained_lie_g[0:constrained_runtime], label='Filtered')
# axes[1, 1].plot(np.arange(0, ddpcbf_runtime)*linear_sys.dt, ddpcbf_lie_g[0:ddpcbf_runtime], label='CBFDDP-Filtered')
# axes[1, 1].set_xlabel('Time (s)', fontsize=ftsize)
# axes[1, 1].set_ylabel('CBF Lie g', fontsize=ftsize)
# axes[1, 1].set_yscale('log')
# #axes[1, 1].legend(fontsize=ftsize)
# #axes[1, 1].set_xticks(ticks=[0, T*linear_sys.dt], labels=[0, T*linear_sys.dt])
# axes[1, 1].tick_params(labelsize=ftsize)
# #axes[1, 1].set_ylim([-1.0, 1.0])
# axes[1, 1].grid()

plt.savefig(f'./linear_sys/cbf_{cbf_type}_filtering_sm_max.png', bbox_inches="tight")
