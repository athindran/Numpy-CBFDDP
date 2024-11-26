import sys
sys.path.append(".")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
import numpy as np
import copy

from cbfs_and_costs import MultiCBF, CBF
from policies import ReachabilityLQPolicy, DDPCBFFilter
from dynamics import LinearSys
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

viridis = mpl.colormaps['viridis'].resampled(8)

linear_sys = LinearSys()
linear_sys.reset()

cbf_type = 'A'
if cbf_type == 'A':
    cbf = MultiCBF()
#elif cbf_type == 'B':
#     cbf = MultiCBF_b()
# elif cbf_type == 'C':
#     cbf = CBF_c()
else:
    cbf = None

T = 350

cbf_a_params = {'kappa':1.0, 'gamma':0.99, 'Rc':5e-2, 'horizon':15}
def run_simulation(linear_sys, cbf, method=None, Rc=None, horizon=None, gamma=None):
    obs = linear_sys.reset()
    action_perf = np.array([-1.0])

    simulation_states = np.zeros((2, T))
    cbf_states = np.zeros((T, ))
    controls = np.zeros((T, ))
    solver_types = np.zeros((T, ))
    lie_f_vals = np.zeros((T, ))
    lie_g_vals = np.zeros((T, ))

    if method == 'ddpcbf':
        ddpcbf = DDPCBFFilter(2, 1, copy.deepcopy(cbf), copy.deepcopy(linear_sys), 
                                horizon=horizon, Rc=Rc, gamma=gamma)

    for idx in range(T):
        if method == 'unfilter':
            action_filtered = action_perf
        elif method == 'hcbf':
            action_filtered, lie_f, lie_g, filter_active = cbf.apply_filter(obs.ravel(), action_perf, linear_sys)
            solver_types[idx] = filter_active
            lie_f_vals[idx] = lie_f.ravel()[0]
            lie_g_vals[idx] = lie_g.ravel()[0]
        elif method == 'ddpcbf':
            action_filtered, ddp_cbf_eval, lie_f, lie_g, barrier_entries = ddpcbf.apply_filter(obs.ravel(), action_perf, linear_sys)
            solver_types[idx] = barrier_entries
            lie_f_vals[idx] = lie_f.ravel()[0]
            lie_g_vals[idx] = lie_g.ravel()[0]

        new_obs, action = linear_sys.step(obs, action=action_filtered)
        obs = np.array(new_obs)

        if method!='ddpcbf':
            cbf_eval = cbf.eval(obs.ravel())
        else:
            cbf_eval = ddp_cbf_eval

        simulation_states[:, idx] = np.array(obs)
        cbf_states[idx] = cbf_eval.ravel()[0]
        controls[idx] = action_filtered.copy().ravel()[0]

        print(f"Step: {idx}, Obs: {new_obs}, Action: {action}, CBF eval: {cbf_eval}")
        # if(cbf_eval<1e-2):
        #     action_perf = np.array([0.1])

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

ftsize=10
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(14.0, 19.0))
alphas = [1.0, 1.0, 1.0, 1.0, 1.0]
lw = 2.5

# unconstrained_dict = run_simulation(linear_sys, cbf, method='unfilter')
# unconstrained_simulation_states = unconstrained_dict['simulation_states']
# unconstrained_cbf_states = unconstrained_dict['cbf_states']
# unconstrained_runtime = unconstrained_dict['runtime']
# unconstrained_controls = unconstrained_dict['controls']

cbf.use_smoothening = False
constrained_dict = run_simulation(linear_sys, cbf, method='hcbf')
constrained_simulation_states = constrained_dict['simulation_states']
constrained_cbf_states = constrained_dict['cbf_states']
constrained_runtime = constrained_dict['runtime']
constrained_controls = constrained_dict['controls']
constrained_solver_types = constrained_dict['solver_types']

for row_number in range(5):
    axes[row_number, 0].plot(np.arange(0, constrained_runtime)*linear_sys.dt, constrained_cbf_states[0:constrained_runtime], label='HCBF-Filtered', color=viridis(0), linewidth=lw)
    axes[row_number, 1].plot(np.arange(0, constrained_runtime)*linear_sys.dt, constrained_controls[0:constrained_runtime], label='HCBF-Filtered', color=viridis(0), linewidth=lw)
    axes[row_number, 2].plot(constrained_simulation_states[0, :], constrained_simulation_states[1, :], label='HCBF-Filtered', color=viridis(0), linewidth=lw)
    axes[row_number, 0].fill_between(np.arange(0, constrained_runtime)*linear_sys.dt, 0.0, 1.0, where=(constrained_solver_types>0), color=viridis(0), alpha=0.1)

cbf.use_smoothening = False
ddpcbf_dict = run_simulation(linear_sys, cbf, method='ddpcbf', Rc=cbf_a_params['Rc'], horizon=cbf_a_params['horizon'], gamma=cbf_a_params['gamma'])
ddpcbf_simulation_states = ddpcbf_dict['simulation_states']
ddpcbf_cbf_states = ddpcbf_dict['cbf_states']
ddpcbf_runtime = ddpcbf_dict['runtime']
ddpcbf_controls = ddpcbf_dict['controls']
ddpcbf_solver_types = ddpcbf_dict['solver_types']

for row_number in range(5):
    axes[row_number, 0].plot(np.arange(0, ddpcbf_runtime)*linear_sys.dt, ddpcbf_cbf_states[0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
    axes[row_number, 1].plot(np.arange(0, ddpcbf_runtime)*linear_sys.dt, ddpcbf_controls[0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
    #axes[row_number, 1].fill_between(np.arange(0, ddpcbf_runtime)*linear_sys.dt, -1.0, 1.0, where=(ddpcbf_solver_types>0), color='r', alpha=0.2)
    axes[row_number, 2].plot(ddpcbf_simulation_states[0, :], ddpcbf_simulation_states[1, :], label='CBFDDP-HM', color='r', alpha=0.7, linewidth=lw)


cbf.use_smoothening = True
for kiter, kappa in enumerate([0.1, 0.5, 1.5, 3.0, 4.5]):
    cbf.kappa = kappa
    ddpcbf_smooth_dict = run_simulation(linear_sys, cbf, method='ddpcbf', Rc=cbf_a_params['Rc'], horizon=cbf_a_params['horizon'], gamma=cbf_a_params['gamma'])
    ddpcbf_smooth_simulation_states = ddpcbf_smooth_dict['simulation_states']
    ddpcbf_smooth_cbf_states = ddpcbf_smooth_dict['cbf_states']
    ddpcbf_smooth_runtime = ddpcbf_smooth_dict['runtime']
    ddpcbf_smooth_controls = ddpcbf_smooth_dict['controls']
    ddpcbf_smooth_solver_types = ddpcbf_smooth_dict['solver_types']

    label_tag = f'CBFDDP-SM'
    axes[kiter, 0].plot(np.arange(0, ddpcbf_smooth_runtime)*linear_sys.dt, ddpcbf_smooth_cbf_states[0:ddpcbf_smooth_runtime], label=label_tag, color='b', alpha=alphas[kiter], linewidth=lw)
    axes[kiter, 1].plot(np.arange(0, ddpcbf_smooth_runtime)*linear_sys.dt, ddpcbf_smooth_controls[0:ddpcbf_smooth_runtime], label=label_tag, color='b', alpha=alphas[kiter], linewidth=lw)
    axes[kiter, 0].fill_between(np.arange(0, ddpcbf_smooth_runtime)*linear_sys.dt, 0.0, 1.0, where=(ddpcbf_smooth_solver_types>0), color='b', alpha=0.1)
    axes[kiter, 2].plot(ddpcbf_smooth_simulation_states[0, :], ddpcbf_smooth_simulation_states[1, :], label=label_tag, color='b', alpha=alphas[kiter], linewidth=lw)
    #axes[kiter, 0].set_title(f'CBFDDP-SM $\kappa=${kappa}', fontsize=12)
    axes[kiter, 1].set_title(f'CBFDDP-SM $\kappa=${kappa}', fontsize=12)
    #axes[kiter, 2].set_title(f'CBFDDP-SM $\kappa=${kappa}', fontsize=12)

for row_number in range(5):
    #axes[row_number, 0].plot(np.arange(0, T)*linear_sys.dt, unconstrained_cbf_states, label='Unfiltered')
    if row_number==4:
        axes[row_number, 0].set_xlabel('Time (s)', fontsize=ftsize)
    axes[row_number, 0].plot(np.arange(0, T)*linear_sys.dt, [[0]]*T)
    axes[row_number, 0].set_ylabel('CBF Value', fontsize=ftsize)
    axes[row_number, 0].legend(fontsize=ftsize)
    xticks = np.round(np.linspace(0, T*linear_sys.dt, 2), 2)
    axes[row_number, 0].set_xticks(ticks=xticks, labels=xticks)
    yticks = np.round(np.linspace(0.0, 1.2, 2), 2)
    axes[row_number, 0].set_yticks(ticks=yticks, labels=yticks)
    axes[row_number, 0].tick_params(labelsize=ftsize)
    #axes[row_number, 0].set_xlim([0, constrained_runtime*linear_sys.dt])
    #axes[row_number, 0].set_ylim([-0.1, round(constrained_cbf_states.max(), 2)])
    #axes[row_number, 0].grid()

    #axes[row_number, 0, 1].plot(np.arange(0, T)*linear_sys.dt, unconstrained_controls, label='UnFiltered')
    if row_number==4:
        axes[row_number, 1].set_xlabel('Time (s)', fontsize=ftsize)
    axes[row_number, 1].set_ylabel('Controls', fontsize=ftsize)
    #axes[row_number, 1].legend(fontsize=ftsize)
    xticks = np.round(np.linspace(0, T*linear_sys.dt, 2), 2)
    axes[row_number, 1].set_xticks(ticks=xticks, labels=xticks)
    yticks = np.round(np.linspace(-1.0, 1.0, 2), 2)
    axes[row_number, 1].set_yticks(ticks=yticks, labels=yticks)
    axes[row_number, 1].tick_params(labelsize=ftsize)
    #axes[row_number, 1].grid()

    ellipse = Ellipse(xy=cbf.c, width=2*cbf.beta, height=2*cbf.beta, 
                            edgecolor='k', fc='None', lw=0.5)
    axes[row_number, 2].add_patch(ellipse)
    ellipse_pair = Ellipse(xy=cbf.c + cbf.shift, width=2*cbf.beta, height=2*cbf.beta, 
                            edgecolor='k', fc='None', lw=0.5)
    axes[row_number, 2].add_patch(ellipse_pair)
    xticks = np.round(np.linspace(-1, 2, 2), 2)
    axes[row_number, 2].set_xticks(ticks=xticks, labels=xticks)
    yticks = np.round(np.linspace(-1.2, 1.2, 3), 2)
    if row_number==4:
        axes[row_number, 2].set_xlabel('X axis', fontsize=ftsize)
    axes[row_number, 2].set_ylabel('Y axis', fontsize=ftsize)
    axes[row_number, 2].tick_params(labelsize=ftsize)
    axes[row_number, 2].set_yticks(ticks=yticks, labels=yticks)
#axes[2].grid()

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

plt.savefig(f'./linear_sys/cbf_{cbf_type}_filtering_smooth_max.png', bbox_inches="tight")
