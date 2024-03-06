import io
import os
import argparse
from contextlib import redirect_stdout

import torch
import matplotlib.pyplot as plt

from src.manifolds.pointcloud import CorrectedPointCloudManifold
from src.datasets.pointcloud import prepare_data, TrajGeodesicsDataset
from train_pointcloud import LitDeepPointCloudMetric


def evaluate_geodesics(proteins, manifold, logger=None, log_debug_info=False, plot_name='geodesic_rmsd'):
    num_proteins = proteins.shape[0]
    protein_len = proteins.shape[1]

    t_steps = 21
    p0 = proteins[0]
    p1 = proteins[-1]
    T = torch.linspace(0, 1, t_steps)  # torch.tensor([1/4,1/2,3/4]) # torch.tensor([1/2])
    pt = torch.zeros(t_steps, protein_len, 3)
    mdt = torch.zeros(t_steps, protein_len, 3)

    def run_computations():
        for i, t in enumerate(T):
            print(f"computing geodesic {i + 1}")
            md_ind = int(i / (t_steps - 1) * (num_proteins - 1))
            pt[i] = manifold.s_geodesic(p0[None, None], p1[None, None], torch.tensor([t]),
                                        step_size=5e-1,
                                        #tol=0.1,
                                        #max_iter=400,
                                        debug=True).squeeze()
            mdt[i] = proteins[md_ind]

    # we want to capture the debug info from the geodesic computation
    if log_debug_info:
        with redirect_stdout(io.StringIO()) as f:
            run_computations()
        eval_info = f.getvalue()
        if logger is not None:
            logger.experiment.add_text("geodesic_eval_info", eval_info)
    else:
        run_computations()

    # compute and plot RMSD in Anstrom for geodesics vs trajectory
    rmsd_T = torch.sqrt(torch.sum((pt - mdt) ** 2, [1, 2]) / protein_len).detach()

    fig_size = 21  # cm
    plt.figure(figsize=(fig_size / 4, fig_size / 4))
    plt.plot(T, rmsd_T, 'tab:red')
    plt.xlim([0, 1])
    plt.ylim([0, rmsd_T.max() + 1])
    plt.xlabel(r'$t$')
    plt.ylabel(r'RMSD from MD simulation ($\AA$)')

    figure = plt.gcf()
    if logger is not None:
        logger.experiment.add_figure(plot_name, figure, close=True)
    else:
        # save figure
        figure.savefig(plot_name + '.png', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None)
    args = parser.parse_args()

    seed = 42

    struct = 1
    data_folder = os.path.join('', "data", "molecular_dynamics")
    results_folder = 'results'

    ca_pos = prepare_data(struct, data_folder)
    dataset = TrajGeodesicsDataset(ca_pos, window_size=10)

    if args.ckpt_path is None:
        evaluate_geodesics(dataset.proteins, dataset.manifold)
    else:
        model = LitDeepPointCloudMetric.load_from_checkpoint(args.ckpt_path)
        n_pointclouds, n_atoms, atom_dim = ca_pos.shape
        beta = 4e-2
        corrected_manifold = CorrectedPointCloudManifold(model.correction_encoder, atom_dim, n_atoms, base=ca_pos[0],
                                                         alpha=1., beta=beta)
        evaluate_geodesics(dataset.proteins, corrected_manifold, plot_name=f'geodesic_rmsd_path_beta_{beta}')
