#!/usr/bin/env python3

# Place import files below
import matplotlib.pyplot as plt
import numpy as np

from common_functions import make_cumulative_function, save_figures
from process_data import LGData, UDGData
from universal_settings import d_lg, lg_galaxy_data_file, sim_ids, sim_styles


def main():
    # Plot settings
    try:
        plt.style.use('./paper.mplstyle')
    except OSError:
        pass
    plot_max_val = 2525.

    # File locations
    n_v_dist_plot = "fig3_cumulative_radial_distribution.pdf"

    # Load simulation data for Fig. 3
    udg_data = UDGData()
    data_list = [
        'simulation_id', 'halo_ids', 'select_udgs_reff2_mu2',
        'select_gal_mstar_nstar', 'dist_from_mw', 'dist_from_midpoint'
    ]
    (simulation_id, halo_ids, select_udgs_reff2_mu2, select_gal_mstar_nstar,
     dist_from_mw, dist_from_midpoint) = udg_data.retrieve_data(data_list)

    # Read observational data
    obs_data = LGData(lg_galaxy_data_file)

    # Observed field galaxy cumulative radial distribution
    field_dist = obs_data.dist[obs_data.select_field_galaxies]
    field_dist_errplus = obs_data.dist_errplus[obs_data.select_field_galaxies]
    field_dist_errminus = obs_data.dist_errminus[
        obs_data.select_field_galaxies]

    # Cumulative function and errors
    (d_field, d_field_plus,
     d_field_minus), n_field = make_field_cumulative_function(
         field_dist, field_dist_errplus, field_dist_errminus, plot_max_val)

    ####################################################################
    # Print out relevant information
    ####################################################################
    for sim_id in sim_ids:
        galaxies_in_sim = simulation_id == sim_id
        udgs_in_sim = galaxies_in_sim * select_udgs_reff2_mu2
        field_in_sim = galaxies_in_sim * select_gal_mstar_nstar

        print(sim_id)
        print_results(udgs_in_sim,
                      field_in_sim,
                      distances=dist_from_midpoint / 1.e3,
                      target_distance=2.5,
                      relative_to='LG')
        print_results(udgs_in_sim,
                      field_in_sim,
                      distances=dist_from_midpoint / 1.e3,
                      target_distance=1.,
                      relative_to='LG',
                      compare_distance=2.5)
        print('------')
        print_results(udgs_in_sim,
                      field_in_sim,
                      distances=dist_from_mw / 1.e3,
                      target_distance=2.5,
                      relative_to='MW')
        print_results(udgs_in_sim,
                      field_in_sim,
                      distances=dist_from_mw / 1.e3,
                      target_distance=1.,
                      relative_to='MW',
                      compare_distance=2.5)
        print('##########')

    ####################################################################
    # Plot N vs. distance (by sim)
    ####################################################################
    fig, (ratio_ax, cdf_ax) = plt.subplots(2,
                                           1,
                                           sharex=True,
                                           figsize=(8, 8),
                                           gridspec_kw={
                                               'hspace': 0,
                                               'wspace': 0,
                                               'height_ratios': [1, 3],
                                           })

    print()
    for s_i, sim_id in enumerate(sim_ids):
        galaxies_in_sim = simulation_id == sim_id
        select_sim_distance = (np.around(dist_from_mw, 2) <=
                               (d_lg * 1.e3))  # kpc
        select_sim_udgs = (galaxies_in_sim * select_udgs_reff2_mu2 *
                           select_sim_distance)
        select_sim_field = (galaxies_in_sim * select_gal_mstar_nstar *
                            select_sim_distance)

        # Print relevant information
        print(sim_id)
        print("N_UDGs,tot: {:>4}".format(select_sim_udgs.sum()))
        print("N_field,tot: {:>3}".format(select_sim_field.sum()))
        print("---------------")

        d_udg, n_udgs = make_cumulative_function(
            dist_from_mw[select_sim_udgs],
            min_val=0.,
            max_val=plot_max_val,
            bins=dist_from_mw[select_sim_field])
        d_all, n_all = make_cumulative_function(
            dist_from_mw[select_sim_field],
            min_val=0.,
            max_val=plot_max_val,
            bins=dist_from_mw[select_sim_field])

        # Plot UDGs
        udg_line, = cdf_ax.plot(d_udg,
                                n_udgs,
                                drawstyle='steps-post',
                                lw=1.75,
                                color=sim_styles[sim_id]['color'])
        # Plot all field galaxies
        cdf_ax.plot(d_all,
                    n_all,
                    linestyle=':',
                    lw=1.5,
                    drawstyle='steps-post',
                    color=udg_line.get_color())

        # Plot ratio N_UDG(<r) / N_field(<r)
        ratio_ax.plot(d_all,
                      n_udgs / n_all,
                      lw=1.75,
                      color=udg_line.get_color(),
                      drawstyle='steps-post',
                      label=sim_id.replace('_', '\_'))

        # Plot incomplete census of observed field galaxies
        if s_i == 0:
            cdf_ax.errorbar(d_field,
                            n_field,
                            xerr=np.asarray([d_field_minus, d_field_plus]),
                            marker='.',
                            color='k',
                            markersize=5,
                            markevery=slice(1, -1, 1),
                            lw=1.25,
                            drawstyle='steps-post',
                            elinewidth=1,
                            capsize=3,
                            capthick=1,
                            label='Observed f\kern0ptield',
                            zorder=0)

    ####################################################################
    # Axis settings
    cdf_ax.set(xlabel=r'$r_{\rm MW}\, \left({\rm kpc}\right)$',
               ylabel=r'$N\!\left(<r_{\rm MW}\right)$',
               xlim=[0., plot_max_val],
               ylim=[0., None])
    ratio_ax.set(ylabel=r'$N_{\rm UDG}\, /\, N_{\rm f\kern0ptield}$',
                 ylim=[0., 1.05])
    cdf_ax.minorticks_on()
    ratio_ax.minorticks_on()

    ####################################################################
    # Add legends
    ####################################################################
    # Colour-coded simulation legend in top panel
    sim_leg = ratio_ax.legend(loc='upper right',
                              handlelength=0,
                              handletextpad=0,
                              labelspacing=0.1,
                              borderpad=0.05)
    for t_item, sim_id in zip(sim_leg.get_texts(), sim_ids):
        t_item.set_color(sim_styles[sim_id]['color'])

    # Data legend in lower panel
    orig_handles, orig_labels = cdf_ax.get_legend_handles_labels()
    cdf_ax.legend([
        plt.Line2D((0, 1), (0, 0), color='k', linestyle='-'),
        plt.Line2D((0, 1), (0, 0), color='k', linestyle=':'), *orig_handles
    ], [
        r'${\rm \textsc{hestia}\, UDGs}$',
        r'${\rm \textsc{hestia}\, f\kern0ptield}$', *orig_labels
    ],
                  loc='upper left')

    save_figures(fig, n_v_dist_plot)

    return None


def make_field_cumulative_function(field_dist, field_dist_errplus,
                                   field_dist_errminus, max_val):
    """Number of observed field galaxies as a function of distances

    Args:
        field_dist (arr): Distances of galaxies from Sun.
        field_dist_errplus (arr): Upper distance uncertainty of
            galaxies.
        field_dist_errminus (arr): Lower distance uncertainty of
            galaxies.
        max_val (fl): Maximum radius to construct function out to.

    Returns:
        tuple:  [0]: Distance to galaxies.
                [1]: Upper distance uncertainty.
                [2]: Lower distance uncertainty.
                [3]: Cumulative number of galaxies.
    """
    srt_idx = np.argsort(field_dist)
    srt_d_field = np.concatenate(([0.], field_dist[srt_idx], [max_val]))
    srt_d_field_plus = np.concatenate(
        ([np.nan], field_dist_errplus[srt_idx], [np.nan])) - srt_d_field
    srt_d_field_minus = srt_d_field - np.concatenate(
        ([np.nan], field_dist_errminus[srt_idx], [np.nan]))
    n_lg = np.concatenate(
        (np.arange(len(srt_d_field) - 1), [len(srt_d_field) - 2]))
    return (srt_d_field, srt_d_field_plus, srt_d_field_minus), n_lg


def print_results(udg_selection,
                  field_selection,
                  distances=None,
                  target_distance=2.5,
                  relative_to='LG',
                  compare_distance=None):
    """Print relevant or interesting results.

    Args:
        udg_selection (arr): Boolean array of UDGs
        field_selection (arr): Boolean array to select field galaxies.
        distances (arr, optional): Distances to galaxies. Defaults to
            None.
        target_distance (fl, optional): Radius of Local Group volume.
            Defaults to 2.5.
        relative_to (str, optional): Description string describing the
            location from which the distances are measured. Defaults to
            'LG'.
        compare_distance (fl, optional): Computes the number of objects
            inside a given distance and compares this with the target
            distance. Defaults to None.

    Returns:
        None
    """
    if distances is not None:
        select_distance = np.around(distances, 2) <= target_distance
    else:
        select_distance = np.ones(len(udg_selection), dtype=bool)

    print("Within {:.1f} Mpc of the {}".format(target_distance, relative_to))
    print("    N_UDG,tot (Reff2, Mu2): {:>4}".format(
        (udg_selection * select_distance).sum()))
    print("    N_field,tot: {:>15}".format(
        (field_selection * select_distance).sum()))
    print("    N_UDG,tot / N_field,tot = {:.2f}".format(
        (udg_selection * select_distance).sum() /
        (field_selection * select_distance).sum()))
    if compare_distance is not None:
        select_c_distance = distances <= compare_distance
        print()
        print("N_UDG,tot(< {:.1f} Mpc) / N_UDG,tot(< {:.1f} Mpc) = {:>8.2f}".
              format(target_distance, compare_distance,
                     (udg_selection * select_distance).sum() /
                     (udg_selection * select_c_distance).sum()))
        print("N_field,tot(< {:.1f} Mpc) / N_field,tot(< {:.1f} Mpc) = {:.2f}".
              format(target_distance, compare_distance,
                     (field_selection * select_distance).sum() /
                     (field_selection * select_c_distance).sum()))
    return None


if __name__ == "__main__":
    main()
