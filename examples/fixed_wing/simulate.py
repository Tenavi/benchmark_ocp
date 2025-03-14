import os

import numpy as np

from optimalcontrol.controls import from_pickle
from optimalcontrol.simulate import monte_carlo
from optimalcontrol.utilities import load_data, save_data

from examples.fixed_wing.problem_definition import FixedWing
from examples.fixed_wing import example_config as config


if __name__ == '__main__':
    # Initialize the optimal control problem
    ocp = FixedWing(**config.params)

    q0_state = ocp.n_states - 1

    # Load controllers
    controllers = []
    for fn in os.listdir(config.controller_dir):
        if fn.endswith('.pickle'):
            filepath = os.path.join(config.controller_dir, fn)
            controllers.append(from_pickle(filepath))
            print(f"Loaded {controllers[-1]}")

    # Load the training and test datasets
    all_data = {split: load_data(os.path.join(config.data_dir,
                                              f'{split}_data.csv'))
                for split in ('test', 'train')}

    # Restrict training data only to the actual trajectories used for training
    # This attribute was created manually in the training script
    used_idx = controllers[-1]._train_idx
    extra_idx = np.arange(len(all_data['train']))
    extra_idx = extra_idx[~np.isin(extra_idx, used_idx)]
    all_data['unused_train'] = all_data['train'][extra_idx]
    all_data['train'] = all_data['train'][used_idx]

    print("\n" + "+" * 80 + "\n")

    # Evaluate performance of the learned controller in closed-loop simulation
    for data_name, data in all_data.items():
        sims = {}

        if len(data) > 0:
            x0_pool = np.hstack([sol['x'][:, :1] for sol in data])
            t_eval = np.arange(0., config.t_int + config.dt_save / 2.,
                               config.dt_save)

            for controller in controllers:
                sims[str(controller)], _ = monte_carlo(ocp, controller, x0_pool,
                                                       [0., config.t_int],
                                                       t_eval=t_eval,
                                                       **config.sim_kwargs)

                # Save simulation data
                save_data(sims[str(controller)],
                          os.path.join(config.data_dir,
                                       f'{data_name}_{controller}_sims.csv'))

            """# Plot the results
            print("Making plots...")
            
            figs = {'training': dict(), 'test': dict()}
            
            for data_idx, data_name in zip((train_idx, test_idx), ('training', 'test')):
                lqr_costs = [ocp.total_cost(sim['t'], sim['x'], sim['u'])[-1]
                             for sim in lqr_sims[data_idx]]
                nn_costs = [ocp.total_cost(sim['t'], sim['x'], sim['u'])[-1]
                            for sim in nn_sims[data_idx]]
            
                figs[data_name]['cost_comparison'] = plotting.plot_total_cost(
                    [sol['v'][0] for sol in data[data_idx]],
                    controller_costs={'LQR': lqr_costs,
                                      f'{type(nn_control).__name__:s}': nn_costs},
                    title=f'Closed-loop cost evaluation ({data_name})')
            
                plotting.save_fig_dict(figs, config.fig_dir)
            
                for controller, sims in zip((lqr, nn_control), (lqr_sims, nn_sims)):
                    ctrl_name = f'{type(controller).__name__:s}'
                    fig_name = 'closed_loop_' + ctrl_name
                    fig_dir = os.path.join(config.fig_dir, data_name, fig_name)
                    plot_closed_loop(sims[data_idx], data[data_idx], t_max=config.t_int,
                                     subtitle=ctrl_name + ', ' + data_name,
                                     save_dir=fig_dir)"""

            """print(f"Plotting {ctrl_name}-controlled simulations")
            
            # Loop through each closed-loop trajectory, assuming this corresponds to
            # the same open-loop optimal trajectory
            for i in tqdm(range(len(sim_data))):
                if args.show_plots:
                    plot_fixed_wing(ocp, [sim_data[i], data[i]],
                                    sim_labels=[ctrl_name, 'optimal'])
                    plt.show()
                else:
                    plot_fixed_wing(ocp, [sim_data[i], data[i]],
                                    sim_labels=[ctrl_name, 'optimal'],
                                    save_dir=os.path.join(config.fig_dir,
                                                          f'{ctrl_name}_sims',
                                                          f'sim_{i}'))"""
