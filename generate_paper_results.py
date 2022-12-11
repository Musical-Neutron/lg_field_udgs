#!/usr/bin/env python3

# Place import files below
import copy
import h5py
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

from common_functions import make_cumulative_function, survey_cone, v_sphere
from process_data import FattahiData, UDGData
from universal_settings import (
    MV_bins, d_lg, des_mu_lim, fattahi_data_file, generated_data_file_template,
    h, k08_mu_vs_d_file, k08_mv_vs_d_file, lsst_mu_lim, mass_in_lg_file, mu_e,
    mu_e2, n_sightings, reff, reff2, sdss_mu_lim, sim_ids, sim_n_part,
    survey_data_file, target_lg_masses, zoa_extent_list)


def main():
    np.random.seed(10)
    # Load relevant simulation data
    udg_data = UDGData()
    data_list = [
        'simulation_id', 'halo_ids', 'select_udgs_reff1_mu1',
        'select_udgs_reff1_mu2', 'select_udgs_reff2_mu1',
        'select_udgs_reff2_mu2', 'dist_from_mw', 'dist_from_m31',
        'dist_from_midpoint', 'abs_mag_Vband', 'app_mag_Vband_rel_mw',
        'app_mag_Vband_rel_m31', 'rband_mu_mag_arsec', 'position_rel_mw',
        'position_rel_m31', 'n_los', 'los_simulation_id', 'los_halo_ids',
        'los_re_rband', 'los_mue_rband', 'los_mue_Vband',
        'select_gal_mstar_nstar'
    ]
    (simulation_id, halo_ids, select_udgs_reff1_mu1, select_udgs_reff1_mu2,
     select_udgs_reff2_mu1, select_udgs_reff2_mu2, dist_from_mw, dist_from_m31,
     dist_from_midpoint, abs_mag_Vband, app_mag_Vband_rel_mw,
     app_mag_Vband_rel_m31, rband_mu_mag_arsec, position_rel_mw,
     position_rel_m31, n_los, los_simulation_id, los_halo_ids, los_re_rband,
     los_mueff_rband, los_mueff_Vband,
     select_gal_mstar_nstar) = udg_data.retrieve_data(data_list)

    # Analysis settings
    max_app_mv = 22
    abs_lf_survey_names = ['SDSS', 'DES', 'LSST']
    abs_lf_survey_mu_lims = [sdss_mu_lim, des_mu_lim, lsst_mu_lim]
    reff_list = [reff, reff, reff2, reff2]
    mu_list = [mu_e, mu_e2, mu_e, mu_e2]
    selection_list = [
        select_udgs_reff1_mu1, select_udgs_reff1_mu2, select_udgs_reff2_mu1,
        select_udgs_reff2_mu2
    ]

    # Universal output settings
    mock_obs_attr_dict = {
        'Meta':
        np.string_('Each row provides N_UDG,SDSS for a given mock SDSS ' +
                   'observation'),
        'Number of pointings':
        n_sightings
    }
    lf_attr_dict = {
        'Meta':
        np.string_('Each row provides N_UDG(< M_V) for a given mock SDSS ' +
                   'observation'),
        'Number of pointings':
        n_sightings
    }
    zoa_attr_dict = {
        'Meta':
        np.string_(
            'Each row provides N_UDG,ZoA / N_UDG,tot for a given disc ' +
            'orientation'),
        'Number of pointings':
        n_sightings
    }
    zoa_attr_dict.update(
        dict([('Column {:02d}'.format(c_i),
               'ZoA extent: {:>4.1f} [deg]'.format(deg))
              for c_i, deg in enumerate(zoa_extent_list)]))

    ####################################################################
    # Load relevant data
    ####################################################################
    # Koposov+(2008) detection efficiency
    mag_v_d = np.genfromtxt(k08_mv_vs_d_file, delimiter=',')
    mu_v_d = np.genfromtxt(k08_mu_vs_d_file, delimiter=',')
    popt, _ = curve_fit(straight_line, np.log10(mag_v_d[:, 0]), mag_v_d[:, 1],
                        [-4.769, 6.])
    mu_popt, _ = curve_fit(straight_line, np.log10(mu_v_d[:, 0]), mu_v_d[:, 1],
                           [-0.6, 31.3])

    # Extend parameter fits to new space
    M_Vlim_mw = straight_line(np.log10(dist_from_mw), *popt)  # d in kpc
    mu_lim_mw = straight_line(np.log10(dist_from_mw), *mu_popt)  # d in kpc
    M_Vlim_m31 = straight_line(np.log10(dist_from_m31), *popt)  # d in kpc
    mu_lim_m31 = straight_line(np.log10(dist_from_m31), *mu_popt)  # d in kpc

    # Read LG masses of HESTIA simulations
    lg_mass_in_dlg = np.genfromtxt(mass_in_lg_file)

    # Load LG mass-galaxy number density relation from Fattahi+(2020)
    fattahi_data = FattahiData(fattahi_data_file)

    # Load SDSS survey data
    survey_data = np.genfromtxt(survey_data_file)
    sdss_surveyArea, sdss_cos_surveyAngle = survey_cone(survey_data[:, 0][0])

    # Generate uniformly-distributed mock survey pointing directions
    surface_points = uniform_points_on_sphere_surface(n_sightings)

    ####################################################################
    # Generate data
    ####################################################################
    # Generate set of scale factors to rescale the simulations to
    # various target masses
    mass_idx = np.floor(np.log10(target_lg_masses))
    mass_pf = target_lg_masses / 10**mass_idx
    # M_LG in string format
    mass_str = '{:0.1f}x10^{:d} Msun'
    t_mass_strings = [
        mass_str.format(pf, idx)
        for pf, idx in zip(mass_pf, np.int64(mass_idx))
    ]
    # Append t_mass_strings to mock_obs_attr_dict
    for i, tm_str in enumerate(t_mass_strings):
        mock_obs_attr_dict.update({'Column {:02d}'.format(i): tm_str})

    m_density = lg_mass_in_dlg * h / v_sphere(d_lg * h)
    n_gal_scale_factors = []
    for target_mass in target_lg_masses:
        target_mass_density = target_mass * h / v_sphere(d_lg * h)
        n_gal_scale_factor = 10.**(
            fattahi_data.interp_func(np.log10(target_mass_density)) -
            fattahi_data.interp_func(np.log10(m_density)))
        n_gal_scale_factors.append(n_gal_scale_factor)

    n_gal_scale_factors = np.row_stack(n_gal_scale_factors)
    scale_factor_data_to_write = np.column_stack(
        (target_lg_masses, n_gal_scale_factors))

    # Iterate over simulations
    for s_i, (sim_id, sim_rescale_factor) in enumerate(
            zip(sim_ids, n_gal_scale_factors.T)):
        print("Processing: {}".format(sim_id))
        galaxies_in_sim = simulation_id == sim_id
        field_in_sim = (galaxies_in_sim * select_gal_mstar_nstar).sum()
        select_sim_distance = dist_from_midpoint <= (d_lg * 1.e3)  # kpc

        # Select galaxies from the lines of sight data set
        los_galaxies_in_sim = los_simulation_id == sim_id
        unique_los_halo_ids = np.unique(los_halo_ids[los_galaxies_in_sim])

        rescaled_n_field_out_data = np.around(field_in_sim *
                                              sim_rescale_factor)

        # Initialize variables to hold output data
        tot_udg_abs_lf_lg_out_data = []
        tot_udg_lf_mw_out_data = []
        tot_udg_lf_m31_out_data = []

        mw_out_data = []
        m31_out_data = []

        mw_udg_frac_zoa_out_data = []
        m31_udg_frac_zoa_out_data = []

        # scaled_mw_out_data = []
        # scaled_m31_out_data = []
        # scaled_combined_out_data = []
        rescaled_mock_obs_mw_out_data = []
        rescaled_mock_obs_m31_out_data = []

        abs_lf_bins = np.sort(abs_mag_Vband[galaxies_in_sim *
                                            (~np.isinf(abs_mag_Vband))])[::-1]

        rescaled_n_udg_out_data = np.empty(
            (len(target_lg_masses), len(selection_list)), dtype=np.uint32)
        for sel_i, (r_eff, mu_eff, selection) in enumerate(
                zip(reff_list, mu_list, selection_list)):
            udgs_in_sim = selection * galaxies_in_sim * select_sim_distance
            n_udgs_in_sim = udgs_in_sim.sum()
            rescaled_n_udgs = np.around((n_udgs_in_sim * sim_rescale_factor))

            # Prepare rescaled N_UDG(< d_lg Mpc) data to be written
            rescaled_n_udg_out_data[:, sel_i] = rescaled_n_udgs

            # LG UDG luminosity functions
            abs_mv_udg_tot, n_abs_mv_udg_tot = make_cumulative_function(
                abs_mag_Vband[udgs_in_sim], bins=abs_lf_bins)

            udg_abs_lfs_lg = [abs_mv_udg_tot, n_abs_mv_udg_tot]
            for mu_lim in abs_lf_survey_mu_lims:
                udgs_in_survey = udgs_in_sim * (rband_mu_mag_arsec <= mu_lim)
                _, n_abs_mv_udg_survey = make_cumulative_function(
                    abs_mag_Vband[udgs_in_survey], bins=abs_lf_bins)
                udg_abs_lfs_lg.append(n_abs_mv_udg_survey)
                pass

            udg_abs_lfs_lg = np.column_stack(udg_abs_lfs_lg)

            app_mv_udg_tot_mw, n_udg_app_mv_mw = make_cumulative_function(
                app_mag_Vband_rel_mw[udgs_in_sim],
                min_val=6,
                max_val=max_app_mv)
            app_mv_udg_tot_m31, n_udg_app_mv_m31 = make_cumulative_function(
                app_mag_Vband_rel_m31[udgs_in_sim],
                min_val=6,
                max_val=max_app_mv)

            mw_udg_app_lfs_lg = np.column_stack(
                (app_mv_udg_tot_mw, n_udg_app_mv_mw))
            m31_udg_app_lfs_lg = np.column_stack(
                (app_mv_udg_tot_m31, n_udg_app_mv_m31))

            # Generate mock SDSS UDG luminosity functions
            mw_lfs = np.empty(n_sightings).tolist()
            m31_lfs = np.empty(n_sightings).tolist()

            frac_udgs_in_mw_zoa = [
                np.zeros(n_sightings) for _ in zoa_extent_list
            ]
            frac_udgs_in_m31_zoa = [
                np.empty(n_sightings) for _ in zoa_extent_list
            ]

            halo_ids_to_match = halo_ids[udgs_in_sim]
            udgs_in_sim_idx = np.where(udgs_in_sim)[0]
            selected_los_mue_Vband = np.zeros(len(halo_ids))
            selected_los_re_rband = np.zeros(len(halo_ids))

            # Iterate over n_sightings
            for i in np.arange(n_sightings):
                # Select random orientation of each UDG from
                # pre-computed data
                los_selection = (
                    np.random.randint(n_los, size=len(unique_los_halo_ids)) +
                    (np.arange(len(unique_los_halo_ids)) * n_los))

                # Cross-match random orientation data with face-on
                # catalogue
                if i == 0:
                    x_bool_match, y_idx_match = cross_match(
                        halo_ids_to_match,
                        los_halo_ids[los_galaxies_in_sim][los_selection])

                # Cross-match relevant properties
                # selected_los_mue_rband[
                rband_mu_mag_arsec[
                    udgs_in_sim_idx[x_bool_match]] = los_mueff_rband[
                        los_galaxies_in_sim][los_selection][y_idx_match]
                selected_los_mue_Vband[
                    udgs_in_sim_idx[x_bool_match]] = los_mueff_Vband[
                        los_galaxies_in_sim][los_selection][y_idx_match]
                selected_los_re_rband[
                    udgs_in_sim_idx[x_bool_match]] = los_re_rband[
                        los_galaxies_in_sim][los_selection][y_idx_match]

                # Does the galaxy with random orientation appear as a
                # UDG?
                select_los_udg = ((selected_los_re_rband >= r_eff) *
                                  (rband_mu_mag_arsec >= mu_eff))

                # Does galaxy pass SDSS surface brightness criterion?
                los_udgs_in_sdss = (select_los_udg *
                                    (rband_mu_mag_arsec <= sdss_mu_lim))

                # Calculate detection efficiencies of each UDG wrt the
                # MW and M31, respectively
                detection_efficiency_mw = det_eff(abs_mag_Vband,
                                                  selected_los_mue_Vband,
                                                  M_Vlim_mw, mu_lim_mw, 1., 1.)
                detection_efficiency_m31 = det_eff(abs_mag_Vband,
                                                   selected_los_mue_Vband,
                                                   M_Vlim_m31, mu_lim_m31, 1.,
                                                   1.)

                # Select UDGs that pass the detection efficiency
                # criterion
                sub_select_mw_udgs = los_udgs_in_sdss * (
                    detection_efficiency_mw >= np.random.rand(
                        len(los_udgs_in_sdss)))
                sub_select_m31_udgs = los_udgs_in_sdss * (
                    detection_efficiency_m31 >= np.random.rand(
                        len(los_udgs_in_sdss)))

                # Fraction of total face-on UDGs in ZoA
                for z_i, z_extent in enumerate(zoa_extent_list):
                    udg_in_mw_zoa = object_in_zoa(position_rel_mw[udgs_in_sim],
                                                  surface_points[i],
                                                  zoa_extent=z_extent)
                    udg_in_m31_zoa = object_in_zoa(
                        position_rel_m31[udgs_in_sim],
                        surface_points[i],
                        zoa_extent=z_extent)
                    frac_udgs_in_mw_zoa[z_i][i] = (udg_in_mw_zoa.sum() /
                                                   udgs_in_sim.sum())
                    frac_udgs_in_m31_zoa[z_i][i] = (udg_in_m31_zoa.sum() /
                                                    udgs_in_sim.sum())

                # Generate a mock SDSS observation of the UDG population
                mw_lfs[i] = generate_mock_lf(
                    sub_pos=position_rel_mw[sub_select_mw_udgs],
                    sub_dis=dist_from_mw[sub_select_mw_udgs],
                    survey_direction=surface_points[i],
                    cos_survey_angle=sdss_cos_surveyAngle,
                    sub_MV=abs_mag_Vband[sub_select_mw_udgs],
                    MV_bins=MV_bins)
                m31_lfs[i] = generate_mock_lf(
                    sub_pos=position_rel_m31[sub_select_m31_udgs],
                    sub_dis=dist_from_m31[sub_select_m31_udgs],
                    survey_direction=surface_points[i],
                    cos_survey_angle=sdss_cos_surveyAngle,
                    sub_MV=abs_mag_Vband[sub_select_m31_udgs],
                    MV_bins=MV_bins)

            # Stack together all the data
            mw_lfs = np.row_stack(mw_lfs)
            m31_lfs = np.row_stack(m31_lfs)

            max_mw_lf = mw_lfs[:, MV_bins <= -11.]
            max_m31_lf = m31_lfs[:, MV_bins <= -11.]

            med_max_mw_lf = np.nanmedian(max_mw_lf[:, -1])
            med_max_m31_lf = np.nanmedian(max_m31_lf[:, -1])

            # Fraction of UDGs detectable by SDSS
            f_max_mw_lf = med_max_mw_lf / udgs_in_sim.sum()
            f_max_m31_lf = med_max_m31_lf / udgs_in_sim.sum()

            # Rescale N_UDG to different LG masses
            rescaled_med_nudg_mw = round_to_nearest_multiple(
                med_max_mw_lf * sim_rescale_factor, 0.5, decimal_precision=1)
            rescaled_med_nudg_m31 = round_to_nearest_multiple(
                med_max_m31_lf * sim_rescale_factor, 0.5, decimal_precision=1)

            # Compile sets of rescaled mock observations
            rescaled_mock_obs_mw = []
            rescaled_mock_obs_m31 = []
            for r_med_mw, r_med_m31 in zip(rescaled_med_nudg_mw,
                                           rescaled_med_nudg_m31):
                # Rescale mock observations
                rescaled_mw_obs = rescale_observations(
                    max_mw_lf[:, -1], r_med_mw, sdss_surveyArea,
                    udgs_in_sim.sum() * f_max_mw_lf)
                rescaled_m31_obs = rescale_observations(
                    max_m31_lf[:, -1], r_med_m31, sdss_surveyArea,
                    udgs_in_sim.sum() * f_max_m31_lf)

                rescaled_mock_obs_mw.append(rescaled_mw_obs)
                rescaled_mock_obs_m31.append(rescaled_m31_obs)

            # Compile data to write to file
            tot_udg_abs_lf_lg_out_data.append(udg_abs_lfs_lg)
            tot_udg_lf_mw_out_data.append(mw_udg_app_lfs_lg)
            tot_udg_lf_m31_out_data.append(m31_udg_app_lfs_lg)
            mw_out_data.append(mw_lfs)
            m31_out_data.append(m31_lfs)
            mw_udg_frac_zoa_out_data.append(frac_udgs_in_mw_zoa)
            m31_udg_frac_zoa_out_data.append(frac_udgs_in_m31_zoa)

            rescaled_mock_obs_mw_out_data.append(
                np.column_stack(rescaled_mock_obs_mw))
            rescaled_mock_obs_m31_out_data.append(
                np.column_stack(rescaled_mock_obs_m31))

        # Write out data
        with h5py.File(generated_data_file_template.format(sim_n_part, sim_id),
                       'w') as f:
            ############################################################
            # Independent data sets
            ############################################################
            rescaled_n_udg_dset = f.create_dataset(
                'Rescaled N_UDG,tot', data=rescaled_n_udg_out_data)
            rescaled_n_udg_dset.attrs.update({
                'Meta':
                'Number of field UDGs in the simulation rescaled to a ' +
                'target M_LG'
            })

            rescaled_n_field_dset = f.create_dataset(
                'Rescaled N_field', data=rescaled_n_field_out_data)
            rescaled_n_field_dset.attrs.update({
                'Meta':
                'Number of field galaxies in the simulation rescaled to a ' +
                'target M_LG'
            })
            scale_factor_dataset = f.create_dataset(
                'LG mass rescale factors',
                data=scale_factor_data_to_write[:, [0, s_i + 1]])
            scale_factor_dataset.attrs.update({
                'Column 00':
                'Target LG mass [Msun]',
                'Column 01':
                'Scale factors (multiply by N_UDG to get rescaled number of ' +
                'UDGs in a LG of given mass)'
            })

            ############################################################
            # LG Group
            ############################################################
            lg_group = f.create_group('LG')
            lg_group.attrs['Meta'] = np.string_('Data wrt. the LG analogue')

            lg_tot_abs_vband_lf_subgrp = lg_group.create_group(
                'UDG absolute V-band LFs')
            lg_tot_abs_vband_lf_subgrp.attrs['Meta'] = np.string_(
                'Absolute V-band magnitude luminosity functions of the ' +
                'entire UDG population wrt. the centre of the Local Group ' +
                '(the midpoint between the MW and M31 analogues)')

            ############################################################
            # MW Group
            ############################################################
            mw_group = f.create_group('MW')
            mw_group.attrs['Meta'] = np.string_('Data wrt. the MW analogue')

            mw_app_lf_subgroup = mw_group.create_group('Apparent UDG LF')
            mw_app_lf_subgroup.attrs.update({
                'Meta':
                'Apparent V-band magnitude UDG luminosity functions wrt. ' +
                'the Milky Way'
            })

            mw_lf_subgroup = mw_group.create_group('SDSS UDG LF')
            mw_lf_subgroup.attrs.update({
                'Meta':
                'Mock SDSS UDG luminosity functions (absolute V-band ' +
                'magnitude)',
                'Number of pointings':
                n_sightings
            })
            mw_rs_nudg_subgroup = mw_group.create_group('Rescaled N_UDG,SDSS')
            mw_rs_nudg_subgroup.attrs.update({
                'Meta':
                'Total number of UDGs in each mock SDSS observation ' +
                'rescaled to a given LG mass',
                'Number of pointings':
                n_sightings
            })
            mw_zoa_subgroup = mw_group.create_group('f_UDG in ZoA')
            mw_zoa_subgroup.attrs.update({
                'Meta': 'Fraction of N_tot in the ZoA',
                'Number of pointings': n_sightings
            })

            ############################################################
            # M31 Group
            ############################################################
            m31_group = f.create_group('M31')
            m31_group.attrs['Meta'] = np.string_('Data wrt. the M31 analogue')
            m31_lf_subgroup = m31_group.create_group('SDSS UDG LF')
            m31_lf_subgroup.attrs.update({
                'Meta':
                'Mock SDSS UDG luminosity functions (absolute V-band ' +
                'magnitude)',
                'Number of pointings':
                n_sightings
            })
            m31_app_lf_subgroup = m31_group.create_group('Apparent UDG LF')
            m31_app_lf_subgroup.attrs.update({
                'Meta':
                'Apparent V-band magnitude UDG luminosity functions wrt. M31'
            })
            m31_rs_nudg_subgroup = m31_group.create_group(
                'Rescaled N_UDG,SDSS')
            m31_rs_nudg_subgroup.attrs.update({
                'Meta':
                'Total number of UDGs in each mock SDSS observation ' +
                'rescaled to a given LG mass',
                'Number of pointings':
                n_sightings
            })
            m31_zoa_subgroup = m31_group.create_group('f_UDG in ZoA')
            m31_zoa_subgroup.attrs.update({
                'Meta': 'Fraction of N_tot in the ZoA',
                'Number of pointings': n_sightings
            })

            # Iterate over UDG selection criteria
            for i, (r_eff, mu_eff, mw_lf, m31_lf, mw_udg_frac_zoa,
                    m31_udg_frac_zoa, lg_abs_lf, lg_app_mw_lf, lg_app_m31_lf,
                    rs_mock_mw, rs_mock_m31) in enumerate(
                        zip(reff_list, mu_list, mw_out_data, m31_out_data,
                            mw_udg_frac_zoa_out_data,
                            m31_udg_frac_zoa_out_data,
                            tot_udg_abs_lf_lg_out_data, tot_udg_lf_mw_out_data,
                            tot_udg_lf_m31_out_data,
                            rescaled_mock_obs_mw_out_data,
                            rescaled_mock_obs_m31_out_data)):
                group_attr_dict = {
                    'Selection {:d}'.format(i + 1):
                    'Reff: {} kpc, mu_eff: {} mag arcsec^-2'.format(
                        r_eff, mu_eff)
                }
                dataset_attr_dict = {
                    'R_e': '{} kpc'.format(r_eff),
                    'mu_e': '{} mag arcsec^-2'.format(mu_eff)
                }
                # Set various subgroup attributes
                lg_tot_abs_vband_lf_subgrp.attrs.update(group_attr_dict)

                mw_lf_subgroup.attrs.update(group_attr_dict)
                mw_app_lf_subgroup.attrs.update(group_attr_dict)
                mw_zoa_subgroup.attrs.update(group_attr_dict)

                m31_lf_subgroup.attrs.update(group_attr_dict)
                m31_app_lf_subgroup.attrs.update(group_attr_dict)
                m31_zoa_subgroup.attrs.update(group_attr_dict)

                ########################################################
                # Create data sets
                ########################################################
                # LG data
                lg_abs_lf_dataset = lg_tot_abs_vband_lf_subgrp.create_dataset(
                    'Selection {:d}'.format(i + 1), data=lg_abs_lf)

                # Set data set attributes
                lg_abs_lf_dataset.attrs.update(dataset_attr_dict)
                lg_abs_lf_dataset.attrs.update({
                    'Column 00': 'M_V',
                    'Column 01': 'N_UDG,tot(<M_V)'
                })
                for n, survey_name in enumerate(abs_lf_survey_names):
                    lg_abs_lf_dataset.attrs.update({
                        'Column {:02d}'.format(n + 2):
                        'N_UDG,{}(<M_V)'.format(survey_name)
                    })

                ########################################################
                # MW data
                mw_lf_dataset = mw_lf_subgroup.create_dataset(
                    'Selection {:d}'.format(i + 1), data=mw_lf)
                mw_lf_dataset.attrs.update(dataset_attr_dict)
                mw_app_lf_dataset = mw_app_lf_subgroup.create_dataset(
                    'Selection {:d}'.format(i + 1), data=lg_app_mw_lf)
                mw_app_lf_dataset.attrs.update({
                    'Column 00': 'm_V',
                    'Column 01': 'N_UDG,tot(<m_V)'
                })
                mw_app_lf_dataset.attrs.update(dataset_attr_dict)
                mw_zoa_dataset = mw_zoa_subgroup.create_dataset(
                    'Selection {:d}'.format(i + 1),
                    data=np.column_stack(mw_udg_frac_zoa))
                rs_nudg_mw_dataset = mw_rs_nudg_subgroup.create_dataset(
                    'Selection {:d}'.format(i + 1), data=rs_mock_mw)

                # Set data set attributes
                mw_zoa_dataset.attrs.update(dataset_attr_dict)
                mw_zoa_dataset.attrs.update(zoa_attr_dict)
                mw_lf_dataset.attrs.update(lf_attr_dict)
                rs_nudg_mw_dataset.attrs.update(dataset_attr_dict)
                rs_nudg_mw_dataset.attrs.update(mock_obs_attr_dict)

                ########################################################
                # M31 data
                m31_lf_dataset = m31_lf_subgroup.create_dataset(
                    'Selection {:d}'.format(i + 1), data=m31_lf)
                m31_lf_dataset.attrs.update(dataset_attr_dict)
                m31_app_lf_dataset = m31_app_lf_subgroup.create_dataset(
                    'Selection {:d}'.format(i + 1), data=lg_app_m31_lf)
                m31_app_lf_dataset.attrs.update({
                    'Column 00': 'm_V',
                    'Column 01': 'N_UDG,tot(<m_V)'
                })
                m31_app_lf_dataset.attrs.update(dataset_attr_dict)
                m31_zoa_dataset = m31_zoa_subgroup.create_dataset(
                    'Selection {:d}'.format(i + 1),
                    data=np.column_stack(m31_udg_frac_zoa))
                rs_nudg_m31_dataset = m31_rs_nudg_subgroup.create_dataset(
                    'Selection {:d}'.format(i + 1), data=rs_mock_m31)

                # Set data set attributes
                m31_zoa_dataset.attrs.update(dataset_attr_dict)
                m31_zoa_dataset.attrs.update(zoa_attr_dict)
                m31_lf_dataset.attrs.update(lf_attr_dict)
                rs_nudg_m31_dataset.attrs.update(dataset_attr_dict)
                rs_nudg_m31_dataset.attrs.update(mock_obs_attr_dict)

    return None


def object_in_zoa(pos,
                  disk_normal,
                  observer_offset=[8., 0., 0.],
                  zoa_extent=5.):
    """Calculates the number of objects in the Zone of Avoidance for a
    given orientation of the disc.

    Args:
        sub_pos (arr): 3D positions of the galaxies with respect to the
            observer. [kpc]
        disk_normal (len(3) arr): 3D normal vector to the disk plane,
            relative to the centre of the disk.
        observer_offset (len(3) arr): 3D position of the observer
            relative to the disk origin. [kpc]
        zoa_extent (fl): The angular extent of the Zone of Avoidance
            with respect to the disc plane [degrees]. Defaults to 5 deg.

    Returns:
        bool arr: Boolean array to select the number of objects in the
            Zone of Avoidance.
    """
    if (zoa_extent > 90.) or (zoa_extent < 0.):
        raise ValueError("zoa_extent must be in the range [0, 90]")
    cos_angle = (np.dot(
        (pos - observer_offset), disk_normal) / (np.linalg.norm(
            (pos - observer_offset), axis=1) * np.linalg.norm(disk_normal)))

    zoa_ex_rad = 90. + np.asarray([zoa_extent, -zoa_extent])
    zoa_ex_rad *= np.pi / 180.
    cos_zoa_ex_rad = np.cos(zoa_ex_rad)

    arr_mask = np.logical_and(cos_angle >= cos_zoa_ex_rad[0],
                              cos_angle <= cos_zoa_ex_rad[1])

    return arr_mask


def generate_mock_lf(sub_pos, sub_dis, survey_direction, cos_survey_angle,
                     sub_MV, MV_bins):
    """Generates a mock luminosity function for a given survey.

    Args:
        sub_pos (arr): 3D coordinates of galaxies relative to the
            observer
        sub_dis (arr): 3D distances of galaxies relative to the observer
        survey_direction (arr): 3D coordinates of the survey pointing
            direction
        cos_survey_angle (fl): cosine of survey opening angle
        sub_MV ([type]): [description]
        MV_bins ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Find all subhaloes inside the mock survey
    cos_theta = (sub_pos * survey_direction).sum(axis=1) / sub_dis
    select = cos_theta >= cos_survey_angle

    # mv_in_mock = np.isin(MV_bins, sub_MV[select])

    mv_in_mock = binned_cumulative_distribution(sub_MV[select], MV_bins,
                                                'less')

    return mv_in_mock


def binned_cumulative_distribution(values, bins, order='greater'):
    """Returns a binned cumulative distribution function.

    Args:
        values (arr): Array of values to be counted.
        bins (arr): Array of bin values.
        order (str): Direction in which to compute the cumulative
            function. Defaults to 'greater'.
            'greater': Sums values greater than bins.
            'less': Sum values less than bins.

    Returns:
        arr: Cumulative distribution by bin value.
    """
    if order == 'greater':
        cumul_values = len(values) - np.searchsorted(np.sort(values), bins)
    elif order == 'less':
        srt_idx = values.argsort()
        cumul_values = np.searchsorted(values, bins, 'right', sorter=srt_idx)
    else:
        raise ValueError("order must be 'greater' or 'less'")
    return cumul_values


def det_eff(M_v, mu, M_Vlim, mu_lim, sig_M, sig_mu):
    """Calculate detection efficiency

    Args:
        M_v (fl): V-band absolute magnitude
        mu (list): Surface brightness [mag / arcsec^2]
        M_Vlim (fl): Detection threshold, abs magnitude [mag]
        mu_lim (fl): Detection threshold, surface brightness
            [mag / arcsec^2]
        sig_M (fl): Abs mag detection threshold width [mag / arcsec^2]
        sig_mu (fl): Surface brightness detection threshold width [mag]

    Returns:
        list: Detection efficiency
    """
    # if isinstance(M_v, (np.ndarray, list)):
    #     ret_val = G((np.asarray(M_v)[:, np.newaxis] - M_Vlim) / sig_M) * G(
    #         (mu - mu_lim) / sig_mu)
    # else:
    #     ret_val = G((M_v - M_Vlim) / sig_M) * G((mu - mu_lim) / sig_mu)
    ret_val = G((M_v - M_Vlim) / sig_M) * G((mu - mu_lim) / sig_mu)

    return ret_val


def G(x):
    """Gaussian integral.

    Args:
        x (list): variable

    Returns:
        list: Gaussian integral.
    """
    return 0.5 * erfc(x / np.sqrt(2.))


def mu(size, M_v):
    """Calculate surface brightness

    Args:
        size (list, fl): Size of galaxy in pc.
        M_v (fl): Absolute magnitude in V-band.

    Returns:
        list: Surface brightness in mag / arcsec^2.
    """
    # if isinstance(M_v, (np.ndarray, list)):
    #     ret_val = 21.572 + np.asarray(M_v)[:, np.newaxis] + (
    #         2.5 * np.log10(np.pi * size**2.))
    # else:
    #     ret_val = 21.572 + M_v + (2.5 * np.log10(np.pi * size**2.))
    ret_val = 21.572 + M_v + (2.5 * np.log10(np.pi * size**2.))

    return ret_val


def straight_line(x, m, c):
    return m * x + c


def uniform_points_on_sphere_surface(n_points):
    """Generates array of points on sphere surface distributed uniformly.
    http://web.archive.org/web/20120421191837/http://www.cgafaq.info/wiki/Evenly_distributed_points_on_sphere

    Returns:
        arr: Array of points
    """
    dz = 2. / n_points
    d_phi = np.pi * (3. - 5.**0.5)
    points = np.zeros((n_points, 3))

    # Calculate z-coordinate
    step = np.arange(n_points)
    points[:, 2] = 1. - dz * step

    # Calculate x and y coordinates
    r = (1. - points[:, 2] * points[:, 2])**0.5
    phi = d_phi * step
    points[:, 0] = r * np.cos(phi)
    points[:, 1] = r * np.sin(phi)

    # points_to_return = cartesian_to_spherical(points)
    points_to_return = points

    return points_to_return


def cartesian_to_spherical(coordinates):
    """Converts from Cartesian to Spherical basis.

        Args:
            coordinates (nd.array (N,3)): (x, y, z) values to convert.

        Returns:
            np.ndarray (N,3): coordinates in Spherical basis
                (r, theta, phi).
        """
    # Validation checks
    coordinates_permitted_types = (list, np.ndarray)
    if not isinstance(coordinates, coordinates_permitted_types):
        raise TypeError("coordinates must be one of: {}".format(
            coordinates_permitted_types))
    else:
        coordinates = np.asarray(coordinates)

    # Check if single vector is supplied or set of vectors
    array_of_arrays = isinstance(coordinates[0], np.ndarray)
    if not array_of_arrays:
        coordinates = coordinates.reshape((1, 3))

    # Function is designed to work in 3D
    if np.size(coordinates, 1) != 3:
        raise ValueError("coordinates should be (N,3) array")

    # Convert to spherical coordinates
    r = np.sqrt((coordinates * coordinates).sum(axis=1))
    theta = np.arccos(coordinates[:, 2] / r)
    phi = np.arctan2(coordinates[:, 1], coordinates[:, 0])

    # Output based on input array
    if array_of_arrays:
        return_array = np.column_stack((r, theta, phi))
    else:
        return_array = np.concatenate((r, theta, phi))

    return return_array


def cross_match(x, y):
    """Find where two arrays have elements in common.

    Args:
        x (arr): numpy array
        y (arr): numpy array

    Returns:
        tuple:
            [0]: boolean array to identify which elements of x are in y
            [1]: array of indexes identifying where elements in x are in
                y.

    Example:
        x = [3, 5, 7, 1, 9, 8, 6, 6]
        y = [2, 1, 5, 10, 100, 6]

        x_bool_match, y_idx_match = cross_match(x, y)

        x_bool_match == [False, True, False, True, False, False, True, True]
        y_idx_match == [2, 1, 5, 5]

        x[x_bool_match] == [5, 1, 6, 6]
        y[y_idx_match] == [5, 1, 6, 6]
    """

    index = np.argsort(y)
    sorted_y = y[index]
    sorted_index = np.searchsorted(sorted_y, x)

    xindex = np.take(index, sorted_index, mode="clip")
    mask = y[xindex] != x

    result = np.ma.array(xindex, mask=mask)

    return ~result.mask, result.data[~result.mask]


def round_to_nearest_multiple(values,
                              multiple,
                              direction='up',
                              decimal_precision=2):
    """Rounds numbers up/down towards the nearest multiple

    Args:
        values (arr, len(N)): Values to round.
        multiple (fl): Multiple to round values towards.
        direction (str, optional): Round 'up' or round 'down'. Defaults
            to 'up'.
        decimal_precision (int, optional): The number of decimals to
            retain after rounding. Defaults to 2.

    Raises:
        ValueError: If direction is neither 'up' nor 'down'

    Returns:
        arr, len(N): Rounded values
    """
    direction_allowed_values = ['up', 'down']
    if direction not in direction_allowed_values:
        raise ValueError(
            "direction must be one of: {}".format(direction_allowed_values))

    if direction == 'up':
        rounded_values = np.ceil(
            np.round(np.asarray(values) / multiple,
                     decimal_precision)) * multiple
    if direction == 'down':
        rounded_values = np.floor(
            np.round(np.asarray(values) / multiple,
                     decimal_precision)) * multiple

    return rounded_values


def rescale_observations(observations, target_median, probability, n_udg_sdss):
    """Rescale mock SDSS observations for a different LG mass

    Args:
        observations (arr, len(N)): Array of observations of the total
            UDG population, N_UDG,SDSS.
        target_median (fl): The target median the distribution should
            satisfy.
        probability (fl/arr): When adding additional UDGs to the
            observations, the probability that a given UDG will be in
            the footprint of the survey.
        n_udg_sdss (fl): The total number of UDGs in the simulation that
            are potentially detectable by the survey.

    Returns:
        arr, len(N): Rescaled observations of N_UDG,survey
    """
    new_observations = copy.deepcopy(observations)
    new_n_udg = copy.deepcopy(n_udg_sdss)
    current_median = np.nanmedian(new_observations)
    # Scale observations down
    if current_median > target_median:
        while current_median > target_median:
            # Weight preferentially towards high-N observations that
            # contain a larger fraction of the total number of
            # potentially observable UDGs
            weights = new_observations / new_n_udg
            # Only remove UDGs from observations with N_UDG > 0
            weights[np.isinf(weights)] = -0.01
            new_observations = new_observations - (
                np.ones(len(new_observations)) *
                (np.random.rand(len(new_observations)) <= weights))
            # Ensure that non-physical mock observations are set to 0
            new_observations[new_observations < 0] = 0
            # Recompute median and the total number of UDGs
            current_median = np.nanmedian(new_observations)
            new_n_udg -= 1
    # Scale observations up
    if current_median < target_median:
        while current_median < target_median:
            new_observations = new_observations + (
                np.ones(len(new_observations)) *
                (np.random.rand(len(new_observations)) <= probability))
            current_median = np.nanmedian(new_observations)

    return new_observations


if __name__ == "__main__":
    main()
