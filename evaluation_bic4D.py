import sys
sys.path.append(".")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Ellipse
import matplotlib.lines as mlines
import numpy as np
import copy

from cbfs_and_costs_bic4d import CBF, MultiCBF_b, MultiCBF_c, MultiCBF_d
from policies import ReachabilityLQPolicy, DDPCBFFilter, DDPLRFilter
from dynamics import Bicycle4D
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

viridis = mpl.colormaps['viridis'].resampled(8)

def run_simulation(dyn_sys, cbf, cbf_type, T, method=None, Rc=None, horizon=None, gamma=None):
    obs = dyn_sys.reset(cbf_type)

    simulation_states = np.zeros((4, T))
    cbf_states = np.zeros((T, ))
    controls = np.zeros((2, T))
    solver_types = np.zeros((T, ))
    lie_f_vals = np.zeros((T, ))
    lie_g_vals = np.zeros((T, ))

    if method == 'ddpcbf':
        ddpcbf = DDPCBFFilter(4, 2, copy.deepcopy(cbf), copy.deepcopy(dyn_sys), 
                                horizon=horizon, Rc=Rc, gamma=gamma, scaling_factor=0.4)
    elif method == 'ddplr':
        ddplr = DDPLRFilter(4, 2, copy.deepcopy(cbf), copy.deepcopy(dyn_sys), 
                                horizon=horizon, Rc=Rc)

    for idx in range(T):
        action_perf = get_action_perf(obs.ravel(), cbf_type)
        if method == 'unfilter':
            action_filtered = action_perf
        elif method == 'hcbf':
            action_filtered, lie_f, lie_g, filter_active = cbf.apply_filter(obs.ravel(), action_perf, dyn_sys)
            solver_types[idx] = filter_active
            lie_f_vals[idx] = lie_f.ravel()[0]
            lie_g_vals[idx] = lie_g.ravel()[0]
        elif method == 'ddpcbf':
            if idx == 0:
                initialize = np.zeros((2,))
            else:
                initialize = controls[:, idx - 1] - action_perf
            action_filtered, ddp_cbf_eval, lie_f, lie_g, barrier_entries = ddpcbf.apply_filter(obs.ravel(), action_perf, 
                                                                                                dyn_sys, initialize=initialize)
            solver_types[idx] = barrier_entries
            lie_f_vals[idx] = lie_f.ravel()[0]
            lie_g_vals[idx] = lie_g.ravel()[0]
        elif method == 'ddplr':
            action_filtered, ddp_cbf_eval = ddplr.apply_filter(obs.ravel(), action_perf, dyn_sys)
            #print("Action", action_filtered)

        new_obs, action = dyn_sys.step(obs, action=action_filtered)
        obs = np.array(new_obs)

        if method=='ddpcbf' or method=='ddplr':
            cbf_eval = ddp_cbf_eval
        else:
            cbf_eval = cbf.eval(obs.ravel())

        simulation_states[:, idx] = np.array(obs)
        cbf_states[idx] = cbf_eval.ravel()[0]
        controls[:, idx] = action.copy().ravel()

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

def set_up_plots_and_axes(axes, cbf_type, nrows, cbf, T, dyn_sys):
    ftsize=13
    axes[0].set_xlabel('Time (s)', fontsize=ftsize)
    axes[0].xaxis.set_label_coords(.5, -.05)
    axes[0].plot(np.arange(0, T)*dyn_sys.dt, [[0]]*T)
    axes[0].set_ylabel('CBF Value', fontsize=ftsize)

    axes[0].legend(fontsize=ftsize, loc="upper right", bbox_to_anchor=(2.5, 1.25), ncol=3)
    
    xticks = np.round(np.linspace(0, T*dyn_sys.dt, 2), 2)
    axes[0].set_xticks(ticks=xticks, labels=xticks)
    yticks = np.round(np.linspace(0.0, 1.2, 2), 2)
    axes[0].set_yticks(ticks=yticks, labels=yticks)
    axes[0].tick_params(labelsize=ftsize)
    axes[0].set_xlim([0, T*dyn_sys.dt])
    if cbf_type=='A':
        axes[0].set_ylim([-0.1, 4.8])
    elif cbf_type=='B' or cbf_type=='C':
        axes[0].set_ylim([-0.1, 1.5])
    axes[0].yaxis.set_label_coords(-0.05, 0.5)
    #axes[0].grid()

    #axes[0, 1].plot(np.arange(0, T)*dyn_sys.dt, unconstrained_controls, label='UnFiltered')
    axes[1].set_xlabel('Time (s)', fontsize=ftsize)
    axes[1].xaxis.set_label_coords(.5, -.05)
    axes[2].set_xlabel('Time (s)', fontsize=ftsize)
    axes[2].xaxis.set_label_coords(.5, -.05)
    axes[1].yaxis.set_label_coords(-0.05, 0.5)
    axes[2].yaxis.set_label_coords(-0.05, 0.5)

    axes[1].set_ylabel('Accel', fontsize=ftsize)
    #axes[1].legend(fontsize=ftsize)
    xticks = np.round(np.linspace(0, T*dyn_sys.dt, 2), 2)
    axes[1].set_xticks(ticks=xticks, labels=xticks)
    axes[1].tick_params(labelsize=ftsize)
    axes[1].set_xlim([0, T*dyn_sys.dt])

    axes[2].set_ylabel('Steer', fontsize=ftsize)
    #axes[1].legend(fontsize=ftsize)
    xticks = np.round(np.linspace(0, T*dyn_sys.dt, 2), 2)
    axes[2].set_xticks(ticks=xticks, labels=xticks)
    axes[2].tick_params(labelsize=ftsize)
    axes[2].set_xlim([0, T*dyn_sys.dt])

    if cbf_type=='A':
        axes[1].set_ylim([-1.2, 1.2])
        yticks = np.round(np.linspace(-1.0, 1.0, 2), 2)
        axes[1].set_yticks(ticks=yticks, labels=yticks)
        axes[2].set_ylim([-1.2, 1.2])
        yticks = np.round(np.linspace(-1.0, 1.0, 2), 2)
        axes[2].set_yticks(ticks=yticks, labels=yticks)
    elif cbf_type=='B':
        axes[1].set_ylim([-1.2, 1.2])
        yticks = np.round(np.linspace(-1.0, 1.0, 2), 2)
        axes[1].set_yticks(ticks=yticks, labels=yticks)
        axes[2].set_ylim([-3.2, 3.2])
        yticks = np.round(np.linspace(-3.2, 3.2, 2), 2)
        axes[2].set_yticks(ticks=yticks, labels=yticks)
    elif cbf_type=='C':
        axes[1].set_ylim([-2.0, 2.0])
        yticks = np.round(np.linspace(-2.0, 2.0, 2), 2)
        axes[1].set_yticks(ticks=yticks, labels=yticks)

    ellipse = Ellipse(xy=cbf.c, width=2*cbf.beta/np.sqrt(cbf.P[0, 0]), height=2*cbf.beta/np.sqrt(cbf.P[1, 1]), 
                            edgecolor='k', fc='None', lw=0.5)
    axes[3].add_patch(ellipse)

    if cbf_type == 'B':
        ellipse_pair = Ellipse(xy=cbf.c + cbf.shift, width=2*cbf.beta/np.sqrt(cbf.P[0, 0]), height=2*cbf.beta/np.sqrt(cbf.P[1, 1]), 
                                edgecolor='k', fc='None', lw=0.5)
        axes[3].add_patch(ellipse_pair)

    axes[3].set_xlabel('State $x$', fontsize=ftsize)
    axes[3].xaxis.set_label_coords(0.5, -.05)

    if cbf_type=='A':        
        axes[3].set_xlim([-1.0, 2.5])
        axes[3].set_ylim([-1.2, 1.2])
        xticks = np.round(np.linspace(-1, 2, 2), 2)
        yticks = np.round(np.linspace(-1.2, 1.2, 2), 2)
    elif cbf_type=='B':
        axes[3].set_xlim([-1.0, 2.5])
        axes[3].set_ylim([-3.5, 3.5])
        xticks = np.round(np.linspace(-1, 2, 2), 2)
        yticks = np.round(np.linspace(-3.5, 3.5, 2), 2)
    elif cbf_type=='C':
        axes[3].set_xlim([-2.5, 3.5])
        axes[3].set_ylim([-1.5, 1.6])
        xticks = np.round(np.linspace(-2.5, 3.5, 2), 2)
        yticks = np.round(np.linspace(-1.5, 1.6, 2), 2)
        yvals = cbf.line_constraint_x * np.ones((101, ))
        xvals = np.linspace(-2.5, 3.5, 101)
        axes[3].plot(xvals, yvals, 'k-')

    axes[3].set_xticks(ticks=xticks, labels=xticks)
    axes[3].set_yticks(ticks=yticks, labels=yticks)
    axes[3].set_ylabel('State $y$', fontsize=ftsize)
    axes[3].tick_params(labelsize=ftsize)
    axes[3].yaxis.set_label_coords(-0.05, 0.5)

def set_up_plots_and_axes_multiple_rows(axes, axeid, cbf_type, nrows, cbf, T, dyn_sys):
    ftsize=13
    for row_number in range(nrows):
        if row_number==nrows-1:
            axes[row_number, 0].set_xlabel('Time (s)', fontsize=ftsize)
            axes[row_number, 0].xaxis.set_label_coords(.5, -.05)
        axes[row_number, 0].plot(np.arange(0, T)*dyn_sys.dt, [[0]]*T)
        axes[row_number, 0].set_ylabel('CBF Value', fontsize=ftsize)

        if row_number == 0:
            axes[row_number, 0].legend(fontsize=ftsize, loc="upper right", bbox_to_anchor=(2.85, 1.35), ncol=4)
        
        xticks = np.round(np.linspace(0, T*dyn_sys.dt, 2), 2)
        axes[row_number, 0].set_xticks(ticks=xticks, labels=xticks)
        yticks = np.round(np.linspace(0.0, 1.2, 2), 2)
        axes[row_number, 0].set_yticks(ticks=yticks, labels=yticks)
        axes[row_number, 0].tick_params(labelsize=ftsize)
        axes[row_number, 0].set_xlim([0, T*dyn_sys.dt])
        if cbf_type=='A':
            axes[row_number, 0].set_ylim([-0.1, 4.8])
        elif cbf_type=='B' or cbf_type=='C' or cbf_type=='D':
            axes[row_number, 0].set_ylim([-0.1, 1.5])
        axes[row_number, 0].yaxis.set_label_coords(-0.05, 0.5)
        #axes[row_number, 0].grid()

        #axes[row_number, 0, 1].plot(np.arange(0, T)*dyn_sys.dt, unconstrained_controls, label='UnFiltered')
        if row_number==nrows-1:
            axes[row_number, 1].set_xlabel('Time (s)', fontsize=ftsize)
            axes[row_number, 1].xaxis.set_label_coords(.5, -.05)
            axes[row_number, 2].set_xlabel('Time (s)', fontsize=ftsize)
            axes[row_number, 2].xaxis.set_label_coords(.5, -.05)
        axes[row_number, 1].yaxis.set_label_coords(-0.05, 0.5)
        axes[row_number, 2].yaxis.set_label_coords(-0.05, 0.5)

        axes[row_number, 1].set_ylabel('Accel', fontsize=ftsize)
        #axes[row_number, 1].legend(fontsize=ftsize)
        xticks = np.round(np.linspace(0, T*dyn_sys.dt, 2), 2)
        axes[row_number, 1].set_xticks(ticks=xticks, labels=xticks)
        axes[row_number, 1].tick_params(labelsize=ftsize)
        axes[row_number, 1].set_xlim([0, T*dyn_sys.dt])

        axes[row_number, 2].set_ylabel('Steer', fontsize=ftsize)
        #axes[row_number, 1].legend(fontsize=ftsize)
        xticks = np.round(np.linspace(0, T*dyn_sys.dt, 2), 2)
        axes[row_number, 2].set_xticks(ticks=xticks, labels=xticks)
        axes[row_number, 2].tick_params(labelsize=ftsize)
        axes[row_number, 2].set_xlim([0, T*dyn_sys.dt])

        if cbf_type=='A':
            axes[row_number, 1].set_ylim([-1.2, 1.2])
            yticks = np.round(np.linspace(-1.0, 1.0, 2), 2)
            axes[row_number, 1].set_yticks(ticks=yticks, labels=yticks)
            axes[row_number, 2].set_ylim([-3.2, 3.2])
            yticks = np.round(np.linspace(-3.2, 3.2, 2), 2)
            axes[row_number, 2].set_yticks(ticks=yticks, labels=yticks)
        elif cbf_type=='B' or cbf_type=='D':
            axes[row_number, 1].set_ylim([-2.2, 2.2])
            yticks = np.round(np.linspace(-2.0, 2.0, 2), 2)
            axes[row_number, 1].set_yticks(ticks=yticks, labels=yticks)
            axes[row_number, 2].set_ylim([-2.2, 2.2])
            yticks = np.round(np.linspace(-2.0, 2.0, 2), 2)
            axes[row_number, 2].set_yticks(ticks=yticks, labels=yticks)
        elif cbf_type=='C':
            axes[row_number, 1].set_ylim([-2.2, 2.2])
            yticks = np.round(np.linspace(-2.0, 2.0, 2), 2)
            axes[row_number, 1].set_yticks(ticks=yticks, labels=yticks)
            axes[row_number, 2].set_ylim([-2.2, 2.2])
            yticks = np.round(np.linspace(-2.0, 2.0, 2), 2)
            axes[row_number, 2].set_yticks(ticks=yticks, labels=yticks)
        
        if axeid == 0:
            ellipse = Ellipse(xy=cbf.c, width=2*cbf.beta/np.sqrt(cbf.P[0, 0]), height=2*cbf.beta/np.sqrt(cbf.P[1, 1]), 
                                    edgecolor='k', fc='None', lw=0.5)
            axes[row_number, 3].add_patch(ellipse)

            if cbf_type == 'B' or cbf_type=='C' or cbf_type=='D':
                ellipse_pair = Ellipse(xy=cbf.c + cbf.shift, width=2*cbf.beta/np.sqrt(cbf.P[0, 0]), height=2*cbf.beta/np.sqrt(cbf.P[1, 1]), 
                                        edgecolor='k', fc='None', lw=0.5)
                axes[row_number, 3].add_patch(ellipse_pair)

            if row_number==nrows-1:
                axes[row_number, 3].set_xlabel('State $x$', fontsize=ftsize)
                axes[row_number, 3].xaxis.set_label_coords(0.5, -.05)

            if cbf_type=='A':        
                axes[row_number, 3].set_xlim([-1.0, 2.5])
                axes[row_number, 3].set_ylim([-1.2, 1.2])
                xticks = np.round(np.linspace(-1, 2, 2), 2)
                yticks = np.round(np.linspace(-1.2, 1.2, 2), 2)
            elif cbf_type=='B' or cbf_type=='D':
                axes[row_number, 3].set_xlim([-1.0, 2.5])
                axes[row_number, 3].set_ylim([-3.5, 3.5])
                xticks = np.round(np.linspace(-1, 2, 2), 2)
                yticks = np.round(np.linspace(-3.5, 3.5, 2), 2)
            elif cbf_type=='C':
                axes[row_number, 3].set_xlim([-1.0, 2.5])
                axes[row_number, 3].set_ylim([-3.5, 3.5])
                xticks = np.round(np.linspace(-1, 2, 2), 2)
                yticks = np.round(np.linspace(-3.5, 3.5, 2), 2)

                xvals = np.linspace(-2.5, 3.5, 101)
                yvals = cbf.line_constraint_x * np.ones((101, )) + 0.1*xvals
                axes[row_number, 3].plot(xvals, yvals, 'k-')

            axes[row_number, 3].set_xticks(ticks=xticks, labels=xticks)
            axes[row_number, 3].set_yticks(ticks=yticks, labels=yticks)
            axes[row_number, 3].set_ylabel('State $y$', fontsize=ftsize)
            axes[row_number, 3].tick_params(labelsize=ftsize)
            axes[row_number, 3].yaxis.set_label_coords(-0.05, 0.5)
        else:
            axes[row_number, 3].set_xlabel('Time (s)', fontsize=ftsize)
            axes[row_number, 3].set_ylabel('Velocity x', fontsize=ftsize)
            axes[row_number, 3].set_xlim(np.linspace(0, T*dyn_sys.dt, 2))
            axes[row_number, 3].set_ylim([0.0, 1.5])
            
            xvals = np.linspace(0, T*dyn_sys.dt, 101)
            yvals = cbf.vel_constraint_x * np.ones((101, ))
            axes[row_number, 3].plot(xvals, yvals, 'k-')
            
            xticks = np.round(np.linspace(0, T*dyn_sys.dt, 2), 2)
            axes[row_number, 3].set_xticks(ticks=xticks, labels=xticks)
            yticks = np.round(np.linspace(0.0, 1.5, 2), 2)
            axes[row_number, 3].set_yticks(ticks=yticks, labels=yticks)
            axes[row_number, 3].yaxis.set_label_coords(-0.05, 0.5)


def get_action_perf(obs, cbf_type):
    if cbf_type == 'A' or cbf_type == 'B' or cbf_type=='C':
        return np.array([1.0, 0.0])
    elif cbf_type == 'D':
        if obs[0]>=1.5:
            return np.array([0.5, 0.0])
        else:
            return np.array([-1.0, 0.0])

def main(cbf_type):
    dyn_sys = Bicycle4D()

    if cbf_type == 'A':
        cbf = CBF()
    elif cbf_type == 'B':
        cbf = MultiCBF_b()
    elif cbf_type == 'C':
        cbf = MultiCBF_c()
    elif cbf_type == 'D':
        cbf = MultiCBF_d() 
    else:
        cbf = None

    cbf_a_params = {'kappa':1.0, 'gamma':0.97, 'Rc':5e-2, 'horizon':40}
    cbf_b_params = {'kappa':1.0, 'gamma':0.98, 'Rc':5e-2, 'horizon':40}
    cbf_c_params = {'kappa':1.0, 'gamma':0.99, 'Rc':5e-2, 'horizon':40}
    cbf_d_params = {'kappa':1.0, 'gamma':0.99, 'Rc':5e-2, 'horizon':40}

    if cbf_type == 'A':
        cbf_params = cbf_a_params
        T = 250
        kappavals = []
        enable_lr = True
        nrows = 1
    elif cbf_type == 'B':
        cbf_params = cbf_b_params
        T = 250
        kappavals = [5.0, 4.0, 3.0, 2.0]
        enable_lr = False
        nrows = 4
    elif cbf_type == 'C':
        cbf_params = cbf_c_params
        T = 350
        kappavals = [5.0, 4.0, 3.0, 2.0]
        enable_lr = False
        nrows = 4
    elif cbf_type == 'D':
        cbf_params = cbf_d_params
        T = 650
        kappavals = [4.0, 3.0]
        enable_lr = False
        nrows = 2

    fig, axes = plt.subplots(nrows=nrows, ncols=4, sharey='col', sharex='col', figsize=(15.0, 3.5*nrows))

    if cbf_type == 'D':
        fig2, axes2 = plt.subplots(nrows=nrows, ncols=4, sharey='col', sharex='col', figsize=(15.0, 3.5*nrows))
    else:
        axes2 = None

    alphas = [1.0, 1.0, 1.0, 1.0, 1.0]
    lw = 2.5

    # unconstrained_dict = run_simulation(dyn_sys, cbf, method='unfilter')
    # unconstrained_simulation_states = unconstrained_dict['simulation_states']
    # unconstrained_cbf_states = unconstrained_dict['cbf_states']
    # unconstrained_runtime = unconstrained_dict['runtime']
    # unconstrained_controls = unconstrained_dict['controls']

    # cbf.use_smoothening = False
    # constrained_dict = run_simulation(dyn_sys, cbf, method='hcbf')
    # constrained_simulation_states = constrained_dict['simulation_states']
    # constrained_cbf_states = constrained_dict['cbf_states']
    # constrained_runtime = constrained_dict['runtime']
    # constrained_controls = constrained_dict['controls']
    # constrained_solver_types = constrained_dict['solver_types']

    # for row_number in range(5):
    #     axes[row_number, 0].plot(np.arange(0, constrained_runtime)*dyn_sys.dt, constrained_cbf_states[0:constrained_runtime], label='HCBF-Filtered', color=viridis(0), linewidth=lw)
    #     axes[row_number, 1].plot(np.arange(0, constrained_runtime)*dyn_sys.dt, constrained_controls[0:constrained_runtime], label='HCBF-Filtered', color=viridis(0), linewidth=lw)
    #     axes[row_number, 2].plot(constrained_simulation_states[0, :], constrained_simulation_states[1, :], label='HCBF-Filtered', color=viridis(0), linewidth=lw)
    #     axes[row_number, 0].fill_between(np.arange(0, constrained_runtime)*dyn_sys.dt, 0.0, 2.0, where=(constrained_solver_types>0), color=viridis(0), alpha=0.1, label='HCBF-active')

    if enable_lr:
        print(f"Starting simulation for DDP-LR")
        cbf.use_smoothening = False
        ddplr_dict = run_simulation(dyn_sys, cbf, cbf_type, T, method='ddplr', Rc=1e-3, horizon=40, gamma=0.0)
        ddplr_simulation_states = ddplr_dict['simulation_states']
        ddplr_cbf_states = ddplr_dict['cbf_states']
        ddplr_runtime = ddplr_dict['runtime']
        ddplr_controls = ddplr_dict['controls']
        ddplr_solver_types = ddplr_dict['solver_types']
        if nrows>1:
            for row_number in range(nrows):
                axes[row_number, 0].plot(np.arange(0, ddplr_runtime)*dyn_sys.dt, ddplr_cbf_states[0:ddplr_runtime], label='LRDDP-HM', color='g', linewidth=lw)
                axes[row_number, 1].plot(np.arange(0, ddplr_runtime)*dyn_sys.dt, ddplr_controls[0, 0:ddplr_runtime], label='LRDDP-HM', color='g',  alpha=0.6, linewidth=lw)
                axes[row_number, 2].plot(np.arange(0, ddplr_runtime)*dyn_sys.dt, ddplr_controls[1, 0:ddplr_runtime], label='LRDDP-HM', color='g',  alpha=0.6, linewidth=lw)
                axes[row_number, 3].plot(ddplr_simulation_states[0, :], ddplr_simulation_states[1, :], label='LRDDP-HM', color='g', linewidth=lw)
        else:
            axes[0].plot(np.arange(0, ddplr_runtime)*dyn_sys.dt, ddplr_cbf_states[0:ddplr_runtime], label='LRDDP-HM', color='g', linewidth=lw)
            axes[1].plot(np.arange(0, ddplr_runtime)*dyn_sys.dt, ddplr_controls[0, 0:ddplr_runtime], label='LRDDP-HM', color='g',  alpha=0.6, linewidth=lw)
            axes[2].plot(np.arange(0, ddplr_runtime)*dyn_sys.dt, ddplr_controls[1, 0:ddplr_runtime], label='LRDDP-HM', color='g',  alpha=0.6, linewidth=lw)
            axes[3].plot(ddplr_simulation_states[0, :], ddplr_simulation_states[1, :], label='LRDDP-HM', color='g', linewidth=lw)


    print(f"Starting simulation for DDP-CBF HM")
    cbf.use_smoothening = False
    ddpcbf_dict = run_simulation(dyn_sys, cbf, cbf_type, T, method='ddpcbf', Rc=cbf_params['Rc'], horizon=cbf_params['horizon'], gamma=cbf_params['gamma'])
    ddpcbf_simulation_states = ddpcbf_dict['simulation_states']
    ddpcbf_cbf_states = ddpcbf_dict['cbf_states']
    ddpcbf_runtime = ddpcbf_dict['runtime']
    ddpcbf_controls = ddpcbf_dict['controls']
    ddpcbf_solver_types = ddpcbf_dict['solver_types']

    for row_number in range(nrows):
        if nrows>1:
            axes[row_number, 0].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_cbf_states[0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
            axes[row_number, 1].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_controls[0, 0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
            axes[row_number, 2].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_controls[1, 0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
            
            if axes2 is not None:
                axes2[row_number, 0].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_cbf_states[0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
                axes2[row_number, 1].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_controls[0, 0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
                axes2[row_number, 2].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_controls[1, 0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
                axes2[row_number, 3].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_simulation_states[2, :], label='CBFDDP-HM', color='r', alpha=0.7, linewidth=lw)     

            #axes[row_number, 1].fill_between(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, -5.0, 5.0, where=(ddpcbf_solver_types>0), color='r', alpha=0.2)
            #axes[row_number, 2].fill_between(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, -5.0, 5.0, where=(ddpcbf_solver_types>0), color='r', alpha=0.2)
            axes[row_number, 3].plot(ddpcbf_simulation_states[0, :], ddpcbf_simulation_states[1, :], label='CBFDDP-HM', color='r', alpha=0.7, linewidth=lw)
        else:
            axes[0].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_cbf_states[0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
            axes[1].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_controls[0, 0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
            axes[2].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_controls[1, 0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
            axes[1].fill_between(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, -5.0, 5.0, where=(ddpcbf_solver_types>0), color='r', alpha=0.2)
            axes[2].fill_between(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, -5.0, 5.0, where=(ddpcbf_solver_types>0), color='r', alpha=0.2)
            axes[3].plot(ddpcbf_simulation_states[0, :], ddpcbf_simulation_states[1, :], label='CBFDDP-HM', color='r', alpha=0.7, linewidth=lw)     

            if axes2 is not None:
                axes2[0].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_cbf_states[0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
                axes2[1].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_controls[0, 0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
                axes2[2].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_controls[1, 0:ddpcbf_runtime], label='CBFDDP-HM', color='r', linewidth=lw)
                axes2[1].fill_between(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, -5.0, 5.0, where=(ddpcbf_solver_types>0), color='r', alpha=0.2)
                axes2[2].fill_between(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, -5.0, 5.0, where=(ddpcbf_solver_types>0), color='r', alpha=0.2)
                axes2[3].plot(np.arange(0, ddpcbf_runtime)*dyn_sys.dt, ddpcbf_simulation_states[2, :], label='CBFDDP-HM', color='r', alpha=0.7, linewidth=lw)     


    cbf.use_smoothening = True
    for kiter, kappa in enumerate(kappavals):
        print(f"Starting simulation for DDP-CBF SM kappa={kappa}")
        cbf.kappa = kappa
        ddpcbf_smooth_dict = run_simulation(dyn_sys, cbf, cbf_type, T, method='ddpcbf', Rc=cbf_params['Rc'], horizon=cbf_params['horizon'], gamma=cbf_params['gamma'])
        ddpcbf_smooth_simulation_states = ddpcbf_smooth_dict['simulation_states']
        ddpcbf_smooth_cbf_states = ddpcbf_smooth_dict['cbf_states']
        ddpcbf_smooth_runtime = ddpcbf_smooth_dict['runtime']
        ddpcbf_smooth_controls = ddpcbf_smooth_dict['controls']
        ddpcbf_smooth_solver_types = ddpcbf_smooth_dict['solver_types']

        label_tag = f'CBFDDP-SM'
        if nrows>1:
            axes[kiter, 0].plot(np.arange(0, ddpcbf_smooth_runtime)*dyn_sys.dt, ddpcbf_smooth_cbf_states[0:ddpcbf_smooth_runtime], label=label_tag, color='b', alpha=alphas[kiter], linewidth=lw)
            axes[kiter, 1].plot(np.arange(0, ddpcbf_smooth_runtime)*dyn_sys.dt, ddpcbf_smooth_controls[0, 0:ddpcbf_smooth_runtime], label=label_tag, color='b', alpha=alphas[kiter], linewidth=lw)
            axes[kiter, 2].plot(np.arange(0, ddpcbf_smooth_runtime)*dyn_sys.dt, ddpcbf_smooth_controls[1, 0:ddpcbf_smooth_runtime], label=label_tag, color='b', alpha=alphas[kiter], linewidth=lw)
            axes[kiter, 0].fill_between(np.arange(0, ddpcbf_smooth_runtime)*dyn_sys.dt, 0.0, 2.0, where=(ddpcbf_smooth_solver_types>0), color='b', alpha=0.1, label='CBFDDP-SM active')
            axes[kiter, 3].plot(ddpcbf_smooth_simulation_states[0, :], ddpcbf_smooth_simulation_states[1, :], label=label_tag, color='b', alpha=alphas[kiter], linewidth=lw)
            #axes[kiter, 0].set_title(f'CBFDDP-SM $\kappa=${kappa}', fontsize=12)
            if cbf_type == 'D':
                axes[kiter, 1].set_title(f'CBFDDP-SM $\kappa=${kappa}' + ' with velocity constraint ', fontsize=12)
            else:
                axes[kiter, 1].set_title(f'CBFDDP-SM $\kappa=${kappa}', fontsize=16)
            #axes[kiter, 2].set_title(f'CBFDDP-SM $\kappa=${kappa}', fontsize=12)

            if axes2 is not None:
                axes2[kiter, 0].plot(np.arange(0, ddpcbf_smooth_runtime)*dyn_sys.dt, ddpcbf_smooth_cbf_states[0:ddpcbf_smooth_runtime], label=label_tag, color='b', alpha=alphas[kiter], linewidth=lw)
                axes2[kiter, 1].plot(np.arange(0, ddpcbf_smooth_runtime)*dyn_sys.dt, ddpcbf_smooth_controls[0, 0:ddpcbf_smooth_runtime], label=label_tag, color='b', alpha=alphas[kiter], linewidth=lw)
                axes2[kiter, 2].plot(np.arange(0, ddpcbf_smooth_runtime)*dyn_sys.dt, ddpcbf_smooth_controls[1, 0:ddpcbf_smooth_runtime], label=label_tag, color='b', alpha=alphas[kiter], linewidth=lw)
                axes2[kiter, 0].fill_between(np.arange(0, ddpcbf_smooth_runtime)*dyn_sys.dt, 0.0, 2.0, where=(ddpcbf_smooth_solver_types>0), color='b', alpha=0.1, label='CBFDDP-SM active')
                axes2[kiter, 3].plot(np.arange(0, ddpcbf_smooth_runtime)*dyn_sys.dt, ddpcbf_smooth_simulation_states[2, :], label=label_tag, color='b', alpha=alphas[kiter], linewidth=lw)
                #axes[kiter, 0].set_title(f'CBFDDP-SM $\kappa=${kappa}', fontsize=12)
                if cbf_type == 'D':
                    axes2[kiter, 1].set_title(f'CBFDDP-SM $\kappa=${kappa}' + ' velocity constraint ', fontsize=12)
                else:
                    axes2[kiter, 1].set_title(f'CBFDDP-SM $\kappa=${kappa}', fontsize=16)            
                    #axes[kiter, 2].set_title(f'CBFDDP-SM $\kappa=${kappa}', fontsize=12)


    if nrows == 1:
        set_up_plots_and_axes(axes, cbf_type, nrows, cbf, T, dyn_sys)
    else:
        for axeid, axes_c in enumerate([axes, axes2]):
            if axes_c is None:
                continue
            set_up_plots_and_axes_multiple_rows(axes_c, axeid, cbf_type, nrows, cbf, T, dyn_sys)

    fig.savefig(f'./dyn_sys/cbf_2d_{cbf_type}_bic4d_filtering_smooth_max.png', bbox_inches="tight")
    if cbf_type=='D':
        fig2.savefig(f'./dyn_sys/cbf_2d_{cbf_type}_bic4d_filtering_smooth_max_2.png', bbox_inches="tight")
            
if __name__ == "__main__":
    for cbf_type in ['A', 'B', 'C', 'D']:
        print(f"Starting simulation for {cbf_type}")
        main(cbf_type)


