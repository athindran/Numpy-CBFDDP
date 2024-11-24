import sys
sys.path.append(".")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import copy

from cbfs_and_costs import CBF
from policies import ReachabilityLQPolicy, DDPCBFFilter
from dynamics import LinearSys

        
linear_sys = LinearSys()
linear_sys.reset()
cbf = CBF()
obs = linear_sys.get_obs()
action_perf = np.array([-1.0])
T = 500

unconstrained_simulation_states = np.zeros((2, T))
unconstrained_cbf_states = np.zeros((T, ))
unconstrained_controls = np.zeros((T, ))
for idx in range(T):
    new_obs, action = linear_sys.step(obs, action=action_perf)
    cbf_eval = cbf.eval(new_obs)
    obs = linear_sys.get_obs()
    unconstrained_simulation_states[:, idx] = np.array(obs)
    unconstrained_cbf_states[idx] = cbf_eval.ravel()[0]
    unconstrained_controls[idx] = action.copy().ravel()[0]

    print(f"Step: {idx}, Obs: {new_obs}, Action: {action}, CBF eval: {cbf_eval}")
    if(cbf_eval<1e-1):
        break
unconstrained_cbf_states[idx+1:] = unconstrained_cbf_states[idx]
unconstrained_simulation_states[:, idx+1:] = np.repeat(unconstrained_simulation_states[:, idx:idx+1], T - idx - 1, axis=1)
unconstrained_runtime = idx

linear_sys.reset()
obs = linear_sys.get_obs()
action_perf = np.array([-1.0])

constrained_simulation_states = np.zeros((2, T))
constrained_cbf_states = np.zeros((T, ))
constrained_controls = np.zeros((T, ))
constrained_lie_f = np.zeros((T, ))
constrained_lie_g = np.zeros((T, ))
for idx in range(T):
    action_filtered, lie_f, lie_g = cbf.apply_filter(obs.ravel(), action_perf, linear_sys)
    constrained_lie_f[idx] = lie_f.ravel()[0]
    constrained_lie_g[idx] = lie_g.ravel()[0]
    print(action_filtered)
    new_obs, action_filtered = linear_sys.step(obs, action=action_filtered)
    cbf_eval = cbf.eval(new_obs)
    obs = linear_sys.get_obs()
    constrained_simulation_states[:, idx] = np.array(obs)
    constrained_cbf_states[idx] = cbf_eval.ravel()[0]
    constrained_controls[idx] = action_filtered.copy().ravel()[0]
    print(f"Step: {idx}, Obs: {new_obs}, Action: {action_filtered}, CBF eval: {cbf_eval}")
    if(cbf_eval<1e-1):
        action_perf = np.array([0.1])
constrained_runtime = idx


ddpcbf = DDPCBFFilter(2, 1, CBF(), copy.deepcopy(linear_sys), 5, 1e-2)
linear_sys.reset()
obs = linear_sys.get_obs()
action_perf = np.array([-1.0])
ddpcbf_simulation_states = np.zeros((2, T))
ddpcbf_cbf_states = np.zeros((T, ))
ddpcbf_controls = np.zeros((T, ))
ddpcbf_lie_f = np.zeros((T, ))
ddpcbf_lie_g = np.zeros((T, ))
solver_types = np.zeros((T, ))
for idx in range(T):
    action_filtered, ddp_cbf_eval, lie_f, lie_g, barrier_entries = ddpcbf.apply_filter(obs.ravel(), action_perf, linear_sys)
    ddpcbf_lie_f[idx] = lie_f.ravel()[0]
    ddpcbf_lie_g[idx] = lie_g.ravel()[0]
    solver_types[idx] = barrier_entries
    print(action_filtered)
    new_obs, action_filtered = linear_sys.step(obs, action=action_filtered)
    obs = linear_sys.get_obs()
    ddpcbf_simulation_states[:, idx] = np.array(obs)
    cbf_eval = cbf.eval(new_obs)
    ddpcbf_cbf_states[idx] = cbf_eval.ravel()[0]
    ddpcbf_controls[idx] = action_filtered.copy().ravel()[0]
    print(f"Step: {idx}, Obs: {new_obs}, Action: {action_filtered}, CBF eval: {cbf_eval}")
    if(cbf_eval<1e-1):
        action_perf = np.array([0.1])

ddpcbf_runtime = idx

ftsize=8
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
#plt.plot(np.arange(0, T)*linear_sys.dt, unconstrained_cbf_states, label='Unfiltered')
axes[0].plot(np.arange(0, constrained_runtime)*linear_sys.dt, constrained_cbf_states[0:constrained_runtime], label='HCBF-Filtered')
axes[0].plot(np.arange(0, ddpcbf_runtime)*linear_sys.dt, ddpcbf_cbf_states[0:ddpcbf_runtime], label='DDP-CBF Filtered')
axes[0].set_xlabel('Time (s)', fontsize=ftsize)
axes[0].set_ylabel('CBF Value', fontsize=ftsize)
#axes[0].legend(fontsize=ftsize)
xticks = np.round(np.linspace(0, T*linear_sys.dt, 4), 2)
axes[0].set_xticks(ticks=xticks, labels=xticks)
yticks = np.round(np.linspace(0.0, constrained_cbf_states.max(), 4), 2)
axes[0].set_yticks(ticks=yticks, labels=yticks)
axes[0].tick_params(labelsize=ftsize)
axes[0].set_xlim([0, constrained_runtime*linear_sys.dt])
axes[0].set_ylim([-0.1, round(constrained_cbf_states.max(), 2)])
axes[0].grid()

#axes[0, 1].plot(np.arange(0, T)*linear_sys.dt, unconstrained_controls, label='UnFiltered')
axes[1].plot(np.arange(0, constrained_runtime)*linear_sys.dt, constrained_controls[0:constrained_runtime], label='HCBF-Filtered')
axes[1].plot(np.arange(0, ddpcbf_runtime)*linear_sys.dt, ddpcbf_controls[0:ddpcbf_runtime], label='CBFDDP-Filtered')
axes[1].fill_between(np.arange(0, ddpcbf_runtime)*linear_sys.dt, -1.0, 1.0, where=(solver_types[1:]>0), color='b', alpha=0.2)
axes[1].set_xlabel('Time (s)', fontsize=ftsize)
axes[1].set_ylabel('Controls', fontsize=ftsize)
axes[1].legend(fontsize=ftsize)
xticks = np.round(np.linspace(0, T*linear_sys.dt, 4), 2)
axes[1].set_xticks(ticks=xticks, labels=xticks)
yticks = np.round(np.linspace(-1.0, 1.0, 4), 2)
axes[1].set_yticks(ticks=yticks, labels=yticks)
axes[1].tick_params(labelsize=ftsize)
axes[1].grid()

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

plt.savefig('./linear_sys/cbf_filtering.png', bbox_inches="tight")
