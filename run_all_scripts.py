#!/usr/bin/env python3

# Place import files below
import generate_paper_results
import fig_01_plot_selection_criteria
import fig_03_plot_cumulative_rad_dist
import fig_04_plot_luminosity_functions
import fig_05_plot_mock_sdss_observations
import print_paper_results


def main():
    # Generate paper data
    print("Generating paper data")
    print("Note: this step is slow...")
    generate_paper_results.main()

    # Plot Fig. 1
    fig_01_plot_selection_criteria.main()

    # Plot Fig. 3
    fig_03_plot_cumulative_rad_dist.main()

    # Plot Fig. 4
    fig_04_plot_luminosity_functions.main()

    # Plot Fig. 5
    fig_05_plot_mock_sdss_observations.main()

    # Print paper results
    print_paper_results.main()
    return None


if __name__ == "__main__":
    main()
