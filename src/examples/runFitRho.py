import sys
import numpy as np
import re
import matplotlib.pyplot as plt
import os
from lmfit import Model, Parameters, minimize, Minimizer
import scipy
from mpmath import mp, mpf

import rhos.rhoUtils as u
from rhos.rhoUtils import init_precision, LogMessage, end, Obs, adjust_precision, Inputs
from rhos import *


def main():

    ############################# Settings #############################################
    path = './results/tmax32sigma0.161825Ne10nboot300mNorm0.6473prec280Na1/Logs'
    file_path_input = './corr_NEWPAUL_as_gi_nt64_N40_N80_gauss_s0p23/tmax32sigma0.161825Ne10nboot300mNorm0.6473prec280Na1/Logs/ResultBayes.txt'
    output_name = './fitresults/fit_results_NEWPAUL_nt64_cauchy_as_gi_double.pdf'
    # Channel choices: 'PS', 'V', 'T', 'AV', 'AT', 'S'
    channel = 'V'
    # Representation choices: 'fund', 'as'
    representation = 'as'
    # Plot x-limits
    plot_min_lim = 0.20
    plot_max_lim = 2.15
    ####################################################################################
    ########################### Preferences ################################
    # If you want to fit with Cauchy (False == Gaussians)
    cauchy_fit = False
    # If both false, it's two Gaussians/Cauchy fit
    triple_fit = False
    four_fit = False
    print_cov_matrix = False
    plot_cov_mat = False
    plot_corr_mat = False
    flag_chi2 = True    # To compute and show the correlated chi-square
    ####################################################################################
    #####################Fitting initial parameter guesses #############################
    params = Parameters()
    params.add('amplitude_1', value=3.638752601521968e-06, min=0.0)
    params.add('mean_1', value=1.0, min=0.97, max=1.03)
    params.add('amplitude_2', value=3.4394563454225194e-06, min=0.0)
    params.add('mean_2', value=1.45, min=1.2, max=1.45)
    if triple_fit == True:
        params.add('amplitude_3', value=2.608025553079287e-06, min=0.0)
        params.add('mean_3', value=2.5, min=2.2, max=2.7)
    if four_fit == True:
        params.add('amplitude_4', value=4.875793098794379e-10, min=0, max=1e-10)
        params.add('mean_4', value=2.5, min=2.5, max=2.8)
    #####################################################################################
    # Extract directory path
    directory = os.path.dirname(output_name)

    # Check if directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    if four_fit == True:
        triple_fit = True
    # Check if the provided channel is valid
    valid_channels = ['PS', 'V', 'T', 'AV', 'AT', 'S']
    if channel not in valid_channels:
        print('Please insert a valid mesonic channel!')
        sys.exit('Error: Invalid channel')

    # Check if the provided representation is valid
    valid_repr = ['fund', 'as']
    if representation not in valid_repr:
        print('Please insert a valid mesonic representation!')
        sys.exit('Error: Invalid representation')

    # Extract sigma and mpi from the path using regular expressions
    sigma_match = re.search(r'sigma([\d.]+)', path)
    mpi_match = re.search(r'mNorm([\d.]+)', path)
    nboot_match = re.search(r'nboot([\d.]+)', path)
    # Update values if matches are found
    if sigma_match:
        sigma = float(sigma_match.group(1))

    if mpi_match:
        mpi = float(mpi_match.group(1))

    if nboot_match:
        nboot = int(nboot_match.group(1))

    # Smearing radius in ratio with Mpi
    sigma /= mpi

    # Get a list of all the file names in the directory
    file_names = os.listdir(path)

    # Filter the file names to include only those starting with 'RhoSamplesE'
    file_names = [file_name for file_name in file_names if file_name.startswith('RhoSamplesE')]

    # Extract the energy values from the file names
    energies = [file_name.split('E')[1].split('sig')[0] for file_name in file_names]

    # Sort the energies in ascending order
    energies.sort()

    # Define the dimensions of the matrix
    ne = len(energies)

    amplitude_vals1 = []
    mean_vals1 = []
    amplitude_vals2 = []
    mean_vals2 = []

    if triple_fit == True:
        amplitude_vals3 = []
        mean_vals3 = []

    if four_fit == True:
        amplitude_vals4 = []
        mean_vals4 = []

    # Create an empty matrix
    rho_resampled = np.zeros((nboot, ne))

    # Fill the matrix using the values from the files
    for i, energy in enumerate(energies):
        file_name = f'RhoSamplesE{energy}sig{sigma*mpi}'
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for j, line in enumerate(lines):
                values = line.split()
                rho_resampled[j, i] = float(values[1])

    rho_T = rho_resampled.T
    # Compute covariance matrix
    cov_matrix = np.cov(rho_T, bias=False)

    # Read the file and extract the last column of numbers
    file_path = file_path_input

    # Initialize an empty list to store the numbers from the last column
    numbers = []
    factors = np.array([])

    # Read the file and extract the last column of numbers, ignoring the first line
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) > 1:  # Ensure there is at least one line in the file
            for line in lines[1:]:
                columns = line.strip().split()
                if columns:
                    # Convert the last column to a float and append it to the numbers list
                    last_column = columns[-2]
                    numbers.append(float(last_column))

    for k in range(len(np.diag(cov_matrix))):
        factors = np.append(factors, numbers[k] / np.sqrt(cov_matrix[k][k]))

    for k in range(len(np.diag(cov_matrix))):
        for j in range(len(np.diag(cov_matrix))):
            cov_matrix[j][k] *= factors[k] * factors[j]
            # cov_matrix[j][k] *= 1.0

    if print_cov_matrix == True:
        print(LogMessage(), "Evaluate covariance")
        with open(os.path.join('./covarianceMatrix_rho.txt'), "w") as output:
            for i in range(len(np.diag(cov_matrix))):
                for j in range(len(np.diag(cov_matrix))):
                    print(i, j, cov_matrix[i, j], file=output)

    print(LogMessage(), "Cond[Cov rho] = {:3.3e}".format(float(np.linalg.cond(cov_matrix))))

    corrmat = np.zeros((ne, ne))
    sigmavec = cov_matrix.diagonal()

    for vi in range(ne):
        for vj in range(ne):
            corrmat[vi][vj] = cov_matrix[vi][vj] / (np.sqrt(sigmavec[vi]) * np.sqrt(sigmavec[vj]))

    if plot_cov_mat:
        plt.imshow(cov_matrix, cmap="viridis")
        plt.colorbar()
        plt.show()
        plt.clf()
    if plot_corr_mat:
        plt.imshow(corrmat, cmap="viridis")
        plt.colorbar()
        plt.show()
        plt.clf()

    inv_cov = np.linalg.inv(cov_matrix)
    # Activating text rendering by LaTeX
    #plt.style.use("paperdraft.mplstyle")
    # Extract the required columns
    x = np.array(energies, dtype=float) / mpi
    rho_central = np.zeros(ne)
    drho_central = np.zeros(ne)
    # Create a new figure with a specific size (width, height) in inches
    plt.figure(figsize=(7, 4.5))  # Width: 8 inches, Height: 6 inches
    for ei in range(ne):
        rho_central[ei] = rho_T[ei].mean()
    drho_central = np.sqrt(cov_matrix.diagonal())
    # Plot the data
    plt.errorbar(x, rho_central, yerr=drho_central, fmt='o', color='black', markersize=3.0,
                 label='Spectral density', elinewidth=1.2)
    # Define the Gaussian function
    def gaussian(x, amplitude, mean):
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
    # Define the sum of two Gaussian functions
    def double_gaussian(params, x):
        amplitude_1 = params['amplitude_1'].value
        mean_1 = params['mean_1'].value
        amplitude_2 = params['amplitude_2'].value
        mean_2 = params['mean_2'].value

        model = gaussian(x, amplitude_1, mean_1) + gaussian(x, amplitude_2, mean_2)
        return model
    def double_gaussian2(amplitude1, mean1, amplitude2, mean2, x):
        model = gaussian(x, amplitude1, mean1) + gaussian(x, amplitude2, mean2)
        return model
    def triple_gaussian2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, x):
        model = gaussian(x, amplitude1, mean1) + gaussian(x, amplitude2, mean2) + gaussian(x,
                                                                                           amplitude3,
                                                                                           mean3)
        return model
    def four_gaussian2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, amplitude4, mean4, x):
        model = gaussian(x, amplitude1, mean1) + gaussian(x, amplitude2, mean2) + gaussian(x,
                                                                                           amplitude3,
                                                                                           mean3, ) + gaussian(x,
                                                                                                               amplitude4,
                                                                                                               mean4)
        return model
    #######################################################################
    # Cauchy functions
    def cauchy(x, amplitude, mean):
        return amplitude * (sigma / ((x - mean) ** 2 + sigma ** 2))
    def double_cauchy(params, x):
        amplitude_1 = params['amplitude_1'].value
        mean_1 = params['mean_1'].value
        amplitude_2 = params['amplitude_2'].value
        mean_2 = params['mean_2'].value

        model = cauchy(x, amplitude_1, mean_1) + cauchy(x, amplitude_2, mean_2)
        return model
    def double_cauchy2(amplitude1, mean1, amplitude2, mean2, x):
        model = cauchy(x, amplitude1, mean1) + cauchy(x, amplitude2, mean2)
        return model
    def triple_cauchy2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, x):
        model = cauchy(x, amplitude1, mean1) + cauchy(x, amplitude2, mean2) + cauchy(x,
                                                                                     amplitude3,
                                                                                     mean3)
        return model
    def four_cauchy2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, amplitude4, mean4, x):
        model = cauchy(x, amplitude1, mean1) + cauchy(x, amplitude2, mean2) + cauchy(x,
                                                                                     amplitude3,
                                                                                     mean3, ) + cauchy(x,
                                                                                                       amplitude4,
                                                                                                       mean4)
        return model
    #######################################################################
    def chisq_correlated(params, x, data, cov):
        assert len(x) == len(data)

        e0 = params['mean_1']
        e1 = params['mean_2']
        ampl_1 = params['amplitude_1']
        ampl_2 = params['amplitude_2']
        if triple_fit == True:
            e2 = params['mean_3']
            ampl_3 = params['amplitude_3']
        if four_fit == True:
            e3 = params['mean_4']
            ampl_4 = params['amplitude_4']

        cov_inv = np.linalg.inv(cov)

        model = double_gaussian2(ampl_1, e0, ampl_2, e1, x)

        if cauchy_fit == True:
            model = double_cauchy2(ampl_1, e0, ampl_2, e1, x)

        if triple_fit == True:
            model = triple_gaussian2(ampl_1, e0, ampl_2, e1, ampl_3, e2, x)
            if cauchy_fit == True:
                model = triple_cauchy2(ampl_1, e0, ampl_2, e1, ampl_3, e2, x)
        if four_fit == True:
            model = four_gaussian2(ampl_1, e0, ampl_2, e1, ampl_3, e2, ampl_4, e3, x)
            if cauchy_fit == True:
                model = four_cauchy2(ampl_1, e0, ampl_2, e1, ampl_3, e2, ampl_4, e3, x)

        diff = abs(data - model)
        residual = cov_inv.dot(diff)
        return residual
    def correlated_residual(amplitude1, mean1, amplitude2, mean2, x, data, cov):
        cov_inv = np.linalg.inv(cov)
        model = double_gaussian2(amplitude1, mean1, amplitude2, mean2, x)
        if cauchy_fit == True:
            model = double_cauchy2(amplitude1, mean1, amplitude2, mean2, x)

        diff = data - model
        residual = diff * cov_inv * diff.T
        return residual
    def correlated_residual_three(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, x, data, cov):
        cov_inv = np.linalg.inv(cov)
        model = triple_gaussian2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, x)
        if cauchy_fit == True:
            model = triple_cauchy2(amplitude1, mean1, amplitude2, mean2, x)
        diff = model - data
        residual = diff * cov_inv * diff.T

        return residual
    def correlated_residual_four(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, amplitude4, mean4, x, data,
                                 cov):
        cov_inv = np.linalg.inv(cov)
        model = four_gaussian2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, amplitude4, mean4, x)
        if cauchy_fit == True:
            model = four_cauchy2(amplitude1, mean1, amplitude2, mean2, x)
        diff = abs(model - data)
        residual = diff * cov_inv * diff.T
        return residual

    choleskyCov = np.linalg.cholesky(cov_matrix)

    for k in range(nboot):
        y = rho_resampled[k, :]

        FITWrapper_corr = Minimizer(chisq_correlated, params, fcn_args=(x, y, choleskyCov))
        result = FITWrapper_corr.minimize()

        # Generate the fitted curve
        x_fit = np.linspace(plot_min_lim, plot_max_lim, 1000)

        y_fit = double_gaussian(result.params, x_fit)
        if cauchy_fit == True:
            y_fit = double_cauchy(result.params, x_fit)

        # Plot the fitted curve
        #        plt.plot(x_fit, y_fit, label='Fitted Curve', linewidth=1.3, color='red', alpha=0.2)

        amplitude_vals1.append(float(result.params['amplitude_1']))
        mean_vals1.append(float(result.params['mean_1']))
        amplitude_vals2.append(float(result.params['amplitude_2']))
        mean_vals2.append(float(result.params['mean_2']))

        if triple_fit == True:
            amplitude_vals3.append(float(result.params['amplitude_3']))
            mean_vals3.append(float(result.params['mean_3']))

        if four_fit == True:
            amplitude_vals4.append(float(result.params['amplitude_4']))
            mean_vals4.append(float(result.params['mean_4']))

        print(LogMessage(), '#############################')
        print(LogMessage(), 'Fit number ', k, ' done.')
        print(LogMessage(), 'Amplitude_1: ', float(result.params['amplitude_1']))
        print(LogMessage(), 'Mean_1: ', float(result.params['mean_1']))
        print(LogMessage(), 'Amplitude_2: ', float(result.params['amplitude_2']))
        print(LogMessage(), 'Mean_2: ', float(result.params['mean_2']))

        if triple_fit == True:
            print(LogMessage(), 'Amplitude_3: ', float(result.params['amplitude_3']))
            print(LogMessage(), 'Mean_3: ', float(result.params['mean_3']))

        if four_fit == True:
            print(LogMessage(), 'Amplitude_4: ', float(result.params['amplitude_4']))
            print(LogMessage(), 'Mean_4: ', float(result.params['mean_4']))
        print(LogMessage(), '#############################')
    ################## End of cycle #######################################à
    amplitude1 = np.average(amplitude_vals1)
    damplitude1 = np.std(amplitude_vals1)
    amplitude2 = np.average(amplitude_vals2)
    damplitude2 = np.std(amplitude_vals2)
    mean1 = np.average(mean_vals1)
    dmean1 = np.std(mean_vals1)
    mean2 = np.average(mean_vals2)
    dmean2 = np.std(mean_vals2)
    if triple_fit == True:
        amplitude3 = np.average(amplitude_vals3)
        damplitude3 = np.std(amplitude_vals3)
        mean3 = np.average(mean_vals3)
        dmean3 = np.std(mean_vals3)
    if four_fit == True:
        amplitude4 = np.average(amplitude_vals4)
        damplitude4 = np.std(amplitude_vals4)
        mean4 = np.average(mean_vals4)
        dmean4 = np.std(mean_vals4)

    y_gaussian_1 = [[0] * len(x_fit) for _ in range(nboot)]
    y_gaussian_2 = [[0] * len(x_fit) for _ in range(nboot)]
    y_gaussian_3 = [[0] * len(x_fit) for _ in range(nboot)]
    y_gaussian_4 = [[0] * len(x_fit) for _ in range(nboot)]
    y_gaussian_sum = [[0] * len(x_fit) for _ in range(nboot)]
    for k in range(nboot):
        # Plot the individual Gaussian components
        y_gaussian_1[k] = gaussian(x_fit, amplitude_vals1[k], mean_vals1[k])
        y_gaussian_2[k] = gaussian(x_fit, amplitude_vals2[k], mean_vals2[k])
        y_gaussian_sum[k] = (y_gaussian_1[k] + y_gaussian_2[k])
        # plt.plot(x_fit, gaussian(x_fit, amplitude_vals1_new[k], mean_vals1_new[k]), label='Gaussian 1', color='orange', linewidth=1.3, alpha=0.2)
        # plt.plot(x_fit, gaussian(x_fit, amplitude_vals2_new[k], mean_vals2_new[k]), label='Gaussian 2', color='blue', linewidth=1.3, alpha=0.2)
        if cauchy_fit == True:
            y_gaussian_1[k] = cauchy(x_fit, amplitude_vals1[k], mean_vals1[k])
            y_gaussian_2[k] = cauchy(x_fit, amplitude_vals2[k], mean_vals2[k])
            y_gaussian_sum[k] = (y_gaussian_1[k] + y_gaussian_2[k])

        if triple_fit == True:
            y_gaussian_3[k] = gaussian(x_fit, amplitude_vals3[k], mean_vals3[k])
            y_gaussian_sum[k] = y_gaussian_1[k] + y_gaussian_2[k] + y_gaussian_3[k]
            # plt.plot(x_fit, y_gaussian_3, label='Gaussian 3', color='gray', linewidth=1.6, alpha=0.2)
            if cauchy_fit == True:
                y_gaussian_3[k] = cauchy(x_fit, amplitude_vals3[k], mean_vals3[k])
                y_gaussian_sum[k] = y_gaussian_1[k] + y_gaussian_2[k] + y_gaussian_3[k]
        if four_fit == True:
            y_gaussian_4[k] = gaussian(x_fit, amplitude_vals4[k], mean_vals4[k])
            y_gaussian_sum[k] = y_gaussian_1[k] + y_gaussian_2[k] + y_gaussian_3[k] + y_gaussian_4[k]
            # plt.plot(x_fit, y_gaussian_4, label='Gaussian 4', color='brown', linewidth=1.3, alpha=0.2)
            if cauchy_fit == True:
                y_gaussian_4[k] = cauchy(x_fit, amplitude_vals4[k], mean_vals4[k])
                y_gaussian_sum[k] = y_gaussian_1[k] + y_gaussian_2[k] + y_gaussian_3[k] + y_gaussian_4[k]

    x1 = np.linspace(plot_min_lim, plot_max_lim, 1000)

    if triple_fit == True:
        if four_fit == True:
            if cauchy_fit == False:
                plt.plot(x1, gaussian(x1, amplitude4, mean4), color='mediumturquoise')
            else:
                plt.plot(x1, cauchy(x1, amplitude4, mean4), color='mediumturquoise')
            if cauchy_fit == False:
                plt.plot(x1,
                         four_gaussian2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, amplitude4, mean4, x1),
                         color='gray')
            else:
                plt.plot(x1,
                         four_cauchy2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, amplitude4, mean4, x1),
                         color='gray')
        else:
            if cauchy_fit == False:
                plt.plot(x1, triple_gaussian2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, x1),
                         color='gray')
            else:
                plt.plot(x1, triple_cauchy2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, x1),
                         color='gray')
    else:
        if cauchy_fit == False:
            plt.plot(x1, double_gaussian2(amplitude1, mean1, amplitude2, mean2, x1), color='orange')
        else:
            plt.plot(x1, double_cauchy2(amplitude1, mean1, amplitude2, mean2, x1), color='orange')

    #plt.title('Cauchy kernel', fontsize=14)
    # Customize the plot
    if channel == 'PS':
        plt.xlabel('$E/m_{PS}$', fontsize=15)
    elif channel == 'V':
        plt.xlabel('$E/m_{V}$', fontsize=15)
    elif channel == 'T':
        plt.xlabel('$E/m_{T}$', fontsize=15)
    elif channel == 'AV':
        plt.xlabel('$E/m_{AV}$', fontsize=15)
    elif channel == 'AT':
        plt.xlabel('$E/m_{AT}$', fontsize=15)
    elif channel == 'S':
        plt.xlabel('$E/m_{S}$', fontsize=15)
    if representation == 'fund':
        plt.ylabel('$\\rho^{\\rm f}_\sigma (E)$', fontsize=15)
    elif representation == 'as':
        plt.ylabel('$\\rho^{\\rm as}_\sigma (E)$', fontsize=15)

    plt.grid(linestyle='dashed', alpha=0.6)
    # plt.legend()
    ############################# Print results ###################################################
    print('############################ Fit results ############################')
    print(LogMessage(), 'E0: ', mean1, '+-', dmean1, '\t', '(', mean1 * mpi, '+-', dmean1 * mpi, ')')
    print(LogMessage(), 'E1: ', mean2, '+-', dmean2, '\t', '(', mean2 * mpi, '+-', dmean2 * mpi, ')')

    if triple_fit == True:
        print(LogMessage(), 'E2: ', mean3, '+-', dmean3, '\t', '(', mean3 * mpi, '+-', dmean3 * mpi, ')')
    if four_fit == True:
        print(LogMessage(), 'E3: ', mean4, '+-', dmean4, '\t', '(', mean4 * mpi, '+-', dmean4 * mpi, ')')

    print(LogMessage(), 'E1/E0: ', mean2 / mean1, '+-',
          np.sqrt((dmean2 / mean1) ** 2 + (dmean1 * mean2 / mean1 ** 2) ** 2))

    print(LogMessage(), '--- Fit parameters --- ')
    print(LogMessage(), 'Amplitude1: ', amplitude1, '+-', damplitude1, "\t", 'Mean1: ', mean1, '+-', dmean1)
    print(LogMessage(), 'Amplitude2: ', amplitude2, '+-', damplitude2, "\t", 'Mean2: ', mean2, '+-', dmean2)

    if triple_fit == True:
        print(LogMessage(), 'Amplitude3: ', amplitude3, '+-', damplitude3, "\t", 'Mean3: ', mean3, '+-', dmean3)
    if four_fit == True:
        print(LogMessage(), 'Amplitude4: ', amplitude4, '+-', damplitude4, "\t", 'Mean4: ', mean4, '+-', dmean4)

    result1 = gaussian(x1, amplitude1, mean1)
    transpose_y_gaussian1 = [[y_gaussian_1[j][i] for j in range(nboot)] for i in range(len(x1))]
    if cauchy_fit == True:
        result1 = cauchy(x1, amplitude1, mean1)
        transpose_y_gaussian1 = [[y_gaussian_1[j][i] for j in range(nboot)] for i in range(len(x1))]
    upper_band1 = [0] * len(x1)
    lower_band1 = [0] * len(x1)

    for j in range(len(x1)):
        error = 1.0 * np.std((transpose_y_gaussian1)[j])
        upper_band1[j] = result1[j] + error
        lower_band1[j] = result1[j] - error
    if cauchy_fit == False:
        plt.plot(x1, gaussian(x1, amplitude1, mean1), color='chocolate', linewidth=1.6)
        plt.fill_between(x1, lower_band1, upper_band1, color='chocolate', alpha=0.2, label='Error Bands')
    else:
        plt.plot(x1, cauchy(x1, amplitude1, mean1), color='chocolate', linewidth=1.6)
        plt.fill_between(x1, lower_band1, upper_band1, color='chocolate', alpha=0.2, label='Error Bands')
    result2 = gaussian(x1, amplitude2, mean2)
    transpose_y_gaussian2 = [[y_gaussian_2[j][i] for j in range(nboot)] for i in range(len(x1))]
    if cauchy_fit == True:
        result2 = cauchy(x1, amplitude2, mean2)
        transpose_y_gaussian2 = [[y_gaussian_2[j][i] for j in range(nboot)] for i in range(len(x1))]
    upper_band2 = [0] * len(x1)
    lower_band2 = [0] * len(x1)
    # print(transpose_y_gaussian2[500])
    for j in range(len(x1)):
        error = 1.0 * np.std((transpose_y_gaussian2)[j])
        upper_band2[j] = result2[j] + error
        lower_band2[j] = result2[j] - error

    if cauchy_fit == False:
        plt.plot(x1, gaussian(x1, amplitude2, mean2), color='olive', linewidth=1.6)
        plt.fill_between(x1, lower_band2, upper_band2, color='olive', alpha=0.25, label='Error Bands')
    else:
        plt.plot(x1, cauchy(x1, amplitude2, mean2), color='olive', linewidth=1.6)
        plt.fill_between(x1, lower_band2, upper_band2, color='olive', alpha=0.25, label='Error Bands')
    if four_fit == True:
        if cauchy_fit == False:
            plt.plot(x1, gaussian(x1, amplitude3, mean3), color='orange', linewidth=1.6)

            result3 = gaussian(x1, amplitude3, mean3)
            transpose_y_gaussian3 = [[y_gaussian_3[j][i] for j in range(nboot)] for i in range(len(x1))]
            upper_band3 = [0] * len(x1)
            lower_band3 = [0] * len(x1)
            for j in range(len(x1)):
                error = 1.0 * np.std((transpose_y_gaussian3)[j])
                upper_band3[j] = result3[j] + error
                lower_band3[j] = result3[j] - error

            plt.fill_between(x1, lower_band3, upper_band3, color='orange', alpha=0.25, label='Error Bands')

            result_sum = (gaussian(x1, amplitude2, mean2) + gaussian(x1, amplitude1, mean1) + gaussian(x1, amplitude3,
                                                                                                       mean3) + gaussian(
                x1, amplitude4,
                mean4))

            transpose_y_gaussian_sum = [[y_gaussian_sum[j][i] for j in range(nboot)] for i in
                                        range(len(x1))]
            upper_band5 = [0] * len(x1)
            lower_band5 = [0] * len(x1)
            # print(transpose_y_gaussian1[999])
            for j in range(len(x1)):
                error = 1.0 * np.std((transpose_y_gaussian_sum)[j])
                upper_band5[j] = result_sum[j] + error
                lower_band5[j] = result_sum[j] - error

            plt.plot(x1, gaussian(x1, amplitude4, mean4), color='pink', linewidth=1.6)

            result4 = gaussian(x1, amplitude4, mean4)
            transpose_y_gaussian4 = [[y_gaussian_4[j][i] for j in range(nboot)] for i in range(len(x1))]
            upper_band4 = [0] * len(x1)
            lower_band4 = [0] * len(x1)
            for j in range(len(x1)):
                error = 1.0 * np.std((transpose_y_gaussian4)[j])
                upper_band4[j] = result4[j] + error
                lower_band4[j] = result4[j] - error

            plt.fill_between(x1, lower_band4, upper_band4, color='pink', alpha=0.25, label='Error Bands')


        else:
            plt.plot(x1, cauchy(x1, amplitude3, mean3), color='orange', linewidth=1.6)

            result3 = cauchy(x1, amplitude3, mean3)
            transpose_y_gaussian3 = [[y_gaussian_3[j][i] for j in range(nboot)] for i in range(len(x1))]
            upper_band3 = [0] * len(x1)
            lower_band3 = [0] * len(x1)
            for j in range(len(x1)):
                error = 1.0 * np.std((transpose_y_gaussian3)[j])
                upper_band3[j] = result3[j] + error
                lower_band3[j] = result3[j] - error

            plt.fill_between(x1, lower_band3, upper_band3, color='orange', alpha=0.25, label='Error Bands')

            result_sum = (
                    cauchy(x1, amplitude2, mean2) + cauchy(x1, amplitude1, mean1) + cauchy(x1, amplitude3,
                                                                                           mean3) + cauchy(
                x1, amplitude4,
                mean4))

            transpose_y_gaussian_sum = [[y_gaussian_sum[j][i] for j in range(nboot)] for i in
                                        range(len(x1))]
            upper_band5 = [0] * len(x1)
            lower_band5 = [0] * len(x1)
            # print(transpose_y_gaussian1[999])
            for j in range(len(x1)):
                error = 1.0 * np.std((transpose_y_gaussian_sum)[j])
                upper_band5[j] = result_sum[j] + error
                lower_band5[j] = result_sum[j] - error

            plt.plot(x1, cauchy(x1, amplitude4, mean4), color='pink', linewidth=1.6)

            result4 = cauchy(x1, amplitude4, mean4)
            transpose_y_gaussian4 = [[y_gaussian_4[j][i] for j in range(nboot)] for i in range(len(x1))]
            upper_band4 = [0] * len(x1)
            lower_band4 = [0] * len(x1)
            for j in range(len(x1)):
                error = 1.0 * np.std((transpose_y_gaussian4)[j])
                upper_band4[j] = result4[j] + error
                lower_band4[j] = result4[j] - error

            plt.fill_between(x1, lower_band4, upper_band4, color='pink', alpha=0.25, label='Error Bands')

        plt.fill_between(x1, lower_band5, upper_band5, color='gray', alpha=0.25, label='Error Bands')
    elif triple_fit == True:
        if cauchy_fit == False:
            plt.plot(x1, gaussian(x1, amplitude3, mean3), color='orange', linewidth=1.6)

            result_sum = (gaussian(x1, amplitude2, mean2) + gaussian(x1, amplitude1, mean1) + gaussian(x1, amplitude3,
                                                                                                       mean3))
            result3 = gaussian(x1, amplitude3, mean3)
            transpose_y_gaussian3 = [[y_gaussian_3[j][i] for j in range(nboot)] for i in range(len(x1))]
            upper_band3 = [0] * len(x1)
            lower_band3 = [0] * len(x1)
            for j in range(len(x1)):
                error = 1.0 * np.std((transpose_y_gaussian3)[j])
                upper_band3[j] = result3[j] + error
                lower_band3[j] = result3[j] - error

            plt.fill_between(x1, lower_band3, upper_band3, color='orange', alpha=0.25, label='Error Bands')

        else:
            plt.plot(x1, cauchy(x1, amplitude3, mean3), color='orange', linewidth=1.6)

            result_sum = (cauchy(x1, amplitude2, mean2) + cauchy(x1, amplitude1, mean1) + cauchy(x1, amplitude3,
                                                                                                 mean3))

            result3 = cauchy(x1, amplitude3, mean3)
            transpose_y_gaussian3 = [[y_gaussian_3[j][i] for j in range(nboot)] for i in range(len(x1))]
            upper_band3 = [0] * len(x1)
            lower_band3 = [0] * len(x1)
            for j in range(len(x1)):
                error = 1.0 * np.std((transpose_y_gaussian3)[j])
                upper_band3[j] = result3[j] + error
                lower_band3[j] = result3[j] - error

            plt.fill_between(x1, lower_band3, upper_band3, color='orange', alpha=0.25, label='Error Bands')

        transpose_y_gaussian_sum = [[y_gaussian_sum[j][i] for j in range(nboot)] for i in range(len(x1))]
        upper_band4 = [0] * len(x1)
        lower_band4 = [0] * len(x1)
        # print(transpose_y_gaussian1[999])
        for j in range(len(x1)):
            error = 1.0 * np.std((transpose_y_gaussian_sum)[j])
            upper_band4[j] = result_sum[j] + error
            lower_band4[j] = result_sum[j] - error

        plt.fill_between(x1, lower_band4, upper_band4, color='gray', alpha=0.25, label='Error Bands')
    else:
        if cauchy_fit == False:
            plt.plot(x1, double_gaussian2(amplitude1, mean1, amplitude2, mean2, x1), color='orange', linewidth=1.8)
            result_sum = (gaussian(x1, amplitude2, mean2) + gaussian(x1, amplitude1, mean1))
        else:
            plt.plot(x1, double_cauchy2(amplitude1, mean1, amplitude2, mean2, x1), color='orange', linewidth=1.8)
            result_sum = (cauchy(x1, amplitude2, mean2) + cauchy(x1, amplitude1, mean1))

        transpose_y_gaussian_sum = [[y_gaussian_sum[j][i] for j in range(nboot)] for i in range(len(x1))]
        upper_band3 = [0] * len(x1)
        lower_band3 = [0] * len(x1)
        # print(transpose_y_gaussian1[999])
        for j in range(len(x1)):
            error = 1.0 * np.std((transpose_y_gaussian_sum)[j])
            upper_band3[j] = result_sum[j] + error
            lower_band3[j] = result_sum[j] - error

        plt.fill_between(x1, lower_band3, upper_band3, color='orange', alpha=0.25, label='Error Bands')

    if flag_chi2:
        if triple_fit == True:
            if four_fit == True:
                if len(x) > 8:
                    chi_square_red = correlated_residual_four(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3,
                                                              amplitude4, mean4, x, rho_central, cov_matrix) / (len(x) - 8)
                else:
                    print('Cannot compute Chi square!')
                    flag_chi2 = False
            else:
                if len(x) > 6:
                    chi_square_red = correlated_residual_three(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, x,
                                                               rho_central, cov_matrix)[0, 0] / (len(x) - 6)
                else:
                    print('Cannot compute Chi square!')
                    flag_chi2 = False
        else:
            if len(x) > 4:
                chi_square_red = correlated_residual(amplitude1, mean1, amplitude2, mean2, x, rho_central, cov_matrix)[
                                     0, 0] / (
                                         len(x) - 4)
                print('len(x): ', len(x))
            else:
                print('Cannot compute Chi square!')
                flag_chi2 = False
        print(LogMessage(), ' Reduced Chi Square (with correlation): ', chi_square_red)

    # Save the figure with the specified size
    plt.savefig(output_name, format='pdf', dpi=300)
    # Display the plot
    plt.show()

if __name__ == "__main__":
    main()
