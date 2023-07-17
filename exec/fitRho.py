import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from lmfit import Model, Parameters, minimize

sys.path.append("..")
from importall import *


def main():
    # Define the path to the directory containing the files
    path = './tmax24sigma0.35928Ne7nboot1000mNorm0.3992prec280/Logs/'

    # If both false, it's two gaussians fit
    triple_fit = False
    four_fit = False

    if four_fit == True:
        triple_fit = True

    mpi = 0.399

    # Get a list of all the file names in the directory
    file_names = os.listdir(path)

    # Filter the file names to include only those starting with 'RhoSamplesE'
    file_names = [file_name for file_name in file_names if file_name.startswith('RhoSamplesE')]

    # Extract the energy values from the file names
    energies = [file_name.split('E')[1].split('sig')[0] for file_name in file_names]

    # Sort the energies in ascending order
    energies.sort()

    # Define the dimensions of the matrix
    nboot = 1000
    ne = len(energies)

    amplitude_vals1 = []
    mean_vals1 = []
    stddev_vals1 = []
    amplitude_vals2 = []
    mean_vals2 = []
    stddev_vals2 = []

    if triple_fit == True:
        amplitude_vals3 = []
        mean_vals3 = []
        stddev_vals3 = []

    if four_fit == True:
        amplitude_vals4 = []
        mean_vals4 = []
        stddev_vals4 = []

    # Create an empty matrix
    rho_resampled = np.zeros((nboot, ne))

    # Fill the matrix using the values from the files
    for i, energy in enumerate(energies):
        file_name = f'RhoSamplesE{energy}sig0.35928'
        file_path = os.path.join(path, file_name)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for j, line in enumerate(lines):
                values = line.split()
                rho_resampled[j, i] = float(values[1])

    cov_matrix = np.zeros((ne, ne))

    rho_T = rho_resampled.T
    # Compute covariance matrix
    cov_matrix = np.cov(rho_T, bias=False)

    print(LogMessage(), "Cond[Cov rho] = {:3.3e}".format(float(np.linalg.cond(cov_matrix))))
    plot_cov_mat = False

    if plot_cov_mat:
        plt.imshow(cov_matrix, cmap="viridis")
        plt.colorbar()
        plt.show()
        plt.clf()

    inv_cov = np.linalg.inv(cov_matrix)

    # Activating text rendering by LaTeX
#    plt.style.use("paperdraft.mplstyle")

    # Extract the required columns
    x = np.array(energies, dtype=float) / mpi
    rho_central = np.zeros(ne)
    drho_central = np.zeros(ne)
    for ei in range(ne):
        rho_central[ei] = rho_T[ei].mean()
    drho_central = np.sqrt(cov_matrix.diagonal())

    # Plot the data
    plt.errorbar(x, rho_central, yerr=drho_central, fmt='o', label='Data', color='black', markersize=3.0, elinewidth=1.2)

    for k in range(nboot):
        y = rho_resampled[k, :]
        # Plot the data
#        plt.errorbar(x, y, yerr=drho_central, fmt='o', label='Data', markersize=3.0, elinewidth=1.2)
        # Define the Gaussian function
        def gaussian(x, amplitude, mean, stddev):
            return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

        # Define the sum of two Gaussian functions
        def double_gaussian(params, x):
            amplitude_1 = params['amplitude_1'].value
            mean_1 = params['mean_1'].value
            stddev_1 = params['stddev_1'].value
            amplitude_2 = params['amplitude_2'].value
            mean_2 = params['mean_2'].value
            stddev_2 = params['stddev_2'].value

            model = gaussian(x, amplitude_1, mean_1, stddev_1) + gaussian(x, amplitude_2, mean_2, stddev_2)
            return model

        # Define the sum of two Gaussian functions
        def triple_gaussian(params, x):
            amplitude_1 = params['amplitude_1'].value
            mean_1 = params['mean_1'].value
            stddev_1 = params['stddev_1'].value
            amplitude_2 = params['amplitude_2'].value
            mean_2 = params['mean_2'].value
            stddev_2 = params['stddev_2'].value
            amplitude_3 = params['amplitude_3'].value
            mean_3 = params['mean_3'].value
            stddev_3 = params['stddev_3'].value

            model = gaussian(x, amplitude_1, mean_1, stddev_1) + gaussian(x, amplitude_2, mean_2, stddev_2) + gaussian(x, amplitude_3, mean_3, stddev_3)
            return model

        # Define the sum of two Gaussian functions
        def four_gaussian(params, x):
            amplitude_1 = params['amplitude_1'].value
            mean_1 = params['mean_1'].value
            stddev_1 = params['stddev_1'].value
            amplitude_2 = params['amplitude_2'].value
            mean_2 = params['mean_2'].value
            stddev_2 = params['stddev_2'].value
            amplitude_3 = params['amplitude_3'].value
            mean_3 = params['mean_3'].value
            stddev_3 = params['stddev_3'].value
            amplitude_4 = params['amplitude_4'].value
            mean_4 = params['mean_4'].value
            stddev_4 = params['stddev_4'].value

            model = gaussian(x, amplitude_1, mean_1, stddev_1) + gaussian(x, amplitude_2, mean_2, stddev_2) + gaussian(x, amplitude_3, mean_3, stddev_3) + gaussian(x, amplitude_4, mean_4, stddev_4)
            return model

        # Create lmfit Parameters object
        params = Parameters()
        params.add('amplitude_1', value=1.0e-7)
        params.add('mean_1', value=1.0)
        params.add('stddev_1', value=0.85)
        params.add('amplitude_2', value=0.4e-7)
        params.add('mean_2', value=2.0)
        params.add('stddev_2', value=0.65)

        if triple_fit == True:
            params.add('amplitude_3', value=0.7e-9)
            params.add('mean_3', value=3.0)
            params.add('stddev_3', value=0.35)

        if four_fit == True:
            params.add('amplitude_4', value=0.02e-7)
            params.add('mean_4', value=3.7)
            params.add('stddev_4', value=0.35)

        # Define the residual function
        def residual(params, x, y, inv_cov):
            amplitude_1 = params['amplitude_1'].value
            amplitude_2 = params['amplitude_2'].value
            mean_1 = params['mean_1'].value
            mean_2 = params['mean_2'].value

            if triple_fit == True:
                amplitude_3 = params['amplitude_3'].value
                mean_3 = params['mean_3'].value
            if four_fit == True:
                amplitude_4 = params['amplitude_4'].value
                mean_4 = params['mean_4'].value

            # Check constraint and penalize if violated
            if amplitude_1 <= amplitude_2:
                penalty1 = 1e16 * (amplitude_2 - amplitude_1)
            else:
                penalty1 = 0

            # Check constraint and penalize if violated
            if amplitude_1 <= 0:
                penalty2 = 1e16 * (amplitude_1)
            else:
                penalty2 = 0

            # Check constraint and penalize if violated
            if amplitude_2 <= 0:
                penalty3 = 1e16 * (amplitude_2)
            else:
                penalty3 = 0

            # Check constraint and penalize if violated
            if mean_2 <= mean_1:
                penalty4 = 1e16 * (mean_1 - mean_2)
            else:
                penalty4 = 0

            if triple_fit == True:
                # Check constraint and penalize if violated
                if amplitude_2 <= amplitude_3:
                    penalty5 = 1e16 * (amplitude_3 - amplitude_2)
                else:
                    penalty5 = 0

                # Check constraint and penalize if violated
                if amplitude_3 <= 0:
                    penalty6 = 1e16 * (amplitude_1)
                else:
                    penalty6 = 0

                # Check constraint and penalize if violated
                if mean_3 <= mean_2:
                    penalty7 = 1e16 * (mean_2 - mean_3)
                else:
                    penalty7 = 0
                model = triple_gaussian(params, x)
            else:
                model = double_gaussian(params, x)

            if four_fit == True:
                # Check constraint and penalize if violated
                if amplitude_3 <= amplitude_4:
                    penalty8 = 1e16 * (amplitude_4 - amplitude_3)
                else:
                    penalty8 = 0

                # Check constraint and penalize if violated
                if amplitude_4 <= 0:
                    penalty9 = 1e16 * (amplitude_4)
                else:
                    penalty9 = 0

                # Check constraint and penalize if violated
                if mean_4 <= mean_3:
                    penalty10 = 1e16 * (mean_3 - mean_4)
                else:
                    penalty10 = 0
                model = four_gaussian(params, x)


            residual = (y - model) @ (inv_cov @ (y - model))
            residual = residual*(1.0/float(ne))

            if triple_fit == True:
                if four_fit == True:
                    return np.concatenate(
                        ([residual] * ne, [penalty1, penalty2, penalty3, penalty4, penalty5, penalty6, penalty7, penalty8, penalty9, penalty10]))
                else:
                    return np.concatenate(([residual] * ne, [penalty1, penalty2, penalty3, penalty4,penalty5, penalty6, penalty7]))
            else:
                return np.concatenate(([residual]*ne, [penalty1, penalty2, penalty3, penalty4]))

        # Minimize the residual function
        result = minimize(residual, params, args=(x, y, inv_cov))

        # Generate the fitted curve
        x_fit = np.linspace(min(x), max(x)+1.3, 1000)
        y_fit = double_gaussian(result.params, x_fit)


        # Plot the fitted curve
#        plt.plot(x_fit, y_fit, label='Fitted Curve', linewidth=1.3, color='red', alpha=0.2)

        amplitude_vals1.append(float(result.params['amplitude_1']))
        mean_vals1.append(float(result.params['mean_1']))
        stddev_vals1.append(float(result.params['stddev_1']))
        amplitude_vals2.append(float(result.params['amplitude_2']))
        mean_vals2.append(float(result.params['mean_2']))
        stddev_vals2.append(float(result.params['stddev_2']))

        if triple_fit == True:
            amplitude_vals3.append(float(result.params['amplitude_3']))
            mean_vals3.append(float(result.params['mean_3']))
            stddev_vals3.append(float(result.params['stddev_3']))

        if four_fit == True:
            amplitude_vals4.append(float(result.params['amplitude_4']))
            mean_vals4.append(float(result.params['mean_4']))
            stddev_vals4.append(float(result.params['stddev_4']))

        '''
        # Plot the individual Gaussian components
        y_gaussian_1 = gaussian(x_fit, result.params['amplitude_1'], result.params['mean_1'], result.params['stddev_1'])
        y_gaussian_2 = gaussian(x_fit, result.params['amplitude_2'], result.params['mean_2'], result.params['stddev_2'])
        plt.plot(x_fit, y_gaussian_1, label='Gaussian 1', color='orange', linewidth=1.3, alpha=0.2)
        plt.plot(x_fit, y_gaussian_2, label='Gaussian 2', color='blue',linewidth=1.3, alpha=0.2)
        if triple_fit == True:
            y_gaussian_3 = gaussian(x_fit, result.params['amplitude_3'], result.params['mean_3'],
                                    result.params['stddev_3'])
            plt.plot(x_fit, y_gaussian_3, label='Gaussian 3', color='gray', linewidth=1.3, alpha=0.2)
        if four_fit == True:
            y_gaussian_4 = gaussian(x_fit, result.params['amplitude_4'], result.params['mean_4'],
                                    result.params['stddev_4'])
            plt.plot(x_fit, y_gaussian_4, label='Gaussian 4', color='brown', linewidth=1.3, alpha=0.2)
        '''
#    for k in range(10):
#        print('Amplitude1: ', amplitude_vals1[k], '\tMean1: ', mean_vals1[k],'\tstddev1: ', stddev_vals1[k])


    amplitude1 = np.average(amplitude_vals1)
    mean1 = np.average(mean_vals1)
    stddev1 = np.average(stddev_vals1)
    damplitude1 = np.std(amplitude_vals1)
    dmean1 = np.std(mean_vals1)
    dstddev1 = np.std(stddev_vals1)

    amplitude2 = np.average(amplitude_vals2)
    mean2 = np.average(mean_vals2)
    stddev2 = np.average(stddev_vals2)
    damplitude2 = np.std(amplitude_vals2)
    dmean2 = np.std(mean_vals2)
    dstddev2 = np.std(stddev_vals2)

    if triple_fit == True:
        amplitude3 = np.average(amplitude_vals3)
        mean3 = np.average(mean_vals3)
        stddev3 = np.average(stddev_vals3)
        damplitude3 = np.std(amplitude_vals3)
        dmean3 = np.std(mean_vals3)
        dstddev3 = np.std(stddev_vals3)

    if four_fit == True:
        amplitude4 = np.average(amplitude_vals4)
        mean4 = np.average(mean_vals4)
        stddev4 = np.average(stddev_vals4)
        damplitude4 = np.std(amplitude_vals4)
        dmean4 = np.std(mean_vals4)
        dstddev4 = np.std(stddev_vals4)

    x1 = np.linspace(min(x), max(x)+1.3, 1000)

    # Calculate the upper and lower error bands
    upper_band1 = (amplitude1 + damplitude1) * np.exp(-((x1 - (mean1 + dmean1)) ** 2) / (2 * (stddev1 + dstddev1) ** 2))
    lower_band1 = (amplitude1 - damplitude1) * np.exp(-((x1 - (mean1 - dmean1)) ** 2) / (2 * (stddev1 - dstddev1) ** 2))

    plt.plot(x1, gaussian(x1,amplitude1,mean1,stddev1), color='blue')
    plt.fill_between(x1, lower_band1, upper_band1, color='blue', alpha=0.4, label='Error Bands')

    upper_band2 = (amplitude2 + damplitude2) * np.exp(-((x1 - (mean2 + dmean2)) ** 2) / (2 * (stddev2 + dstddev2) ** 2))
    lower_band2 = (amplitude2 - damplitude2) * np.exp(-((x1 - (mean2 - dmean2)) ** 2) / (2 * (stddev2 - dstddev2) ** 2))

    plt.plot(x1, gaussian(x1, amplitude2, mean2, stddev2), color='green')
    plt.fill_between(x1, lower_band2, upper_band2, color='green', alpha=0.4, label='Error Bands')



    if triple_fit == True:
        def triple_gaussian2(amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, amplitude3, mean3, stddev3, x):
            model = gaussian(x, amplitude1, mean1, stddev1) + gaussian(x, amplitude2, mean2, stddev2) + gaussian(x,
                                                                                                                 amplitude3,
                                                                                                                 mean3,
                                                                                                                 stddev3)
            return model

        # Calculate the upper and lower error bands
        upper_band4 = (amplitude3 + damplitude3) * np.exp(
            -((x1 - (mean3 + dmean3)) ** 2) / (2 * (stddev3 + dstddev3) ** 2))
        lower_band4 = (amplitude3 - damplitude3) * np.exp(
            -((x1 - (mean3 - dmean3)) ** 2) / (2 * (stddev3 - dstddev3) ** 2))

        plt.plot(x1, gaussian(x1, amplitude3, mean3, stddev3), color='purple')
        plt.fill_between(x1, lower_band4, upper_band4, color='purple', alpha=0.4, label='Error Bands')

        if four_fit == True:
            def four_gaussian2(amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, amplitude3, mean3, stddev3, amplitude4, mean4, stddev4, x):
                model = gaussian(x, amplitude1, mean1, stddev1) + gaussian(x, amplitude2, mean2, stddev2) + gaussian(x,
                                                                                                                     amplitude3,
                                                                                                                     mean3,
                                                                                                                     stddev3) + gaussian(x,
                                                                                                                     amplitude4,
                                                                                                                     mean4,
                                                                                                                     stddev4)
                return model

            # Calculate the upper and lower error bands
            upper_band5 = (amplitude4 + damplitude4) * np.exp(
                -((x1 - (mean4 + dmean4)) ** 2) / (2 * (stddev4 + dstddev4) ** 2))
            lower_band5 = (amplitude4 - damplitude4) * np.exp(
                -((x1 - (mean4 - dmean4)) ** 2) / (2 * (stddev4 - dstddev4) ** 2))

            plt.plot(x1, gaussian(x1, amplitude4, mean4, stddev4), color='brown')
            plt.fill_between(x1, lower_band5, upper_band5, color='brown', alpha=0.4, label='Error Bands')

            plt.plot(x1, four_gaussian2(amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, amplitude3, mean3,
                                          stddev3, amplitude4, mean4, stddev4, x1), color='gray')

            upper_band3 = (amplitude1 + damplitude1) * np.exp(
                -((x1 - (mean1 + dmean1)) ** 2) / (2 * (stddev1 + dstddev1) ** 2)) + (
                                      amplitude2 + damplitude2) * np.exp(
                -((x1 - (mean2 + dmean2)) ** 2) / (2 * (stddev2 + dstddev2) ** 2)) + (
                                      amplitude3 + damplitude3) * np.exp(
                -((x1 - (mean3 + dmean3)) ** 2) / (2 * (stddev3 + dstddev3) ** 2)) + (
                                      amplitude4 + damplitude4) * np.exp(
                -((x1 - (mean4 + dmean4)) ** 2) / (2 * (stddev4 + dstddev4) ** 2))
            lower_band3 = (amplitude1 - damplitude1) * np.exp(
                -((x1 - (mean1 - dmean1)) ** 2) / (2 * (stddev1 - dstddev1) ** 2)) + (
                                      amplitude2 - damplitude2) * np.exp(
                -((x1 - (mean2 - dmean2)) ** 2) / (2 * (stddev2 - dstddev2) ** 2)) + (
                                      amplitude3 - damplitude3) * np.exp(
                -((x1 - (mean3 - dmean3)) ** 2) / (2 * (stddev3 - dstddev3) ** 2)) + (
                                      amplitude4 - damplitude4) * np.exp(
                -((x1 - (mean4 - dmean4)) ** 2) / (2 * (stddev4 - dstddev4) ** 2))

            plt.fill_between(x1, lower_band3, upper_band3, color='gray', alpha=0.4, label='Error Bands')
        else:
            plt.plot(x1, triple_gaussian2(amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, amplitude3, mean3, stddev3, x1), color='gray')

            upper_band3 = (amplitude1 + damplitude1) * np.exp(
                -((x1 - (mean1 + dmean1)) ** 2) / (2 * (stddev1 + dstddev1) ** 2)) + (amplitude2 + damplitude2) * np.exp(
                -((x1 - (mean2 + dmean2)) ** 2) / (2 * (stddev2 + dstddev2) ** 2)) + (amplitude3 + damplitude3) * np.exp(
                -((x1 - (mean3 + dmean3)) ** 2) / (2 * (stddev3 + dstddev3) ** 2))
            lower_band3 = (amplitude1 - damplitude1) * np.exp(
                -((x1 - (mean1 - dmean1)) ** 2) / (2 * (stddev1 - dstddev1) ** 2)) + (amplitude2 - damplitude2) * np.exp(
                -((x1 - (mean2 - dmean2)) ** 2) / (2 * (stddev2 - dstddev2) ** 2)) + (amplitude3 - damplitude3) * np.exp(
                -((x1 - (mean3 - dmean3)) ** 2) / (2 * (stddev3 - dstddev3) ** 2))

            plt.fill_between(x1, lower_band3, upper_band3, color='gray', alpha=0.4, label='Error Bands')

    else:
        def double_gaussian2(amplitude1, mean1, stddev1,amplitude2, mean2, stddev2, x):
            model = gaussian(x, amplitude1, mean1, stddev1) + gaussian(x, amplitude2, mean2, stddev2)
            return model

        plt.plot(x1, double_gaussian2(amplitude1, mean1, stddev1,amplitude2, mean2, stddev2, x1), color='gray')

        upper_band3 = (amplitude1 + damplitude1) * np.exp(-((x1 - (mean1 + dmean1)) ** 2) / (2 * (stddev1 + dstddev1) ** 2)) + (amplitude2 + damplitude2) * np.exp(-((x1 - (mean2 + dmean2)) ** 2) / (2 * (stddev2 + dstddev2) ** 2))
        lower_band3 = (amplitude1 - damplitude1) * np.exp(-((x1 - (mean1 - dmean1)) ** 2) / (2 * (stddev1 - dstddev1) ** 2)) + (amplitude2 - damplitude2) * np.exp(-((x1 - (mean2 - dmean2)) ** 2) / (2 * (stddev2 - dstddev2) ** 2))

        plt.fill_between(x1, lower_band3, upper_band3, color='gray', alpha=0.4, label='Error Bands')

    # Customize the plot
    plt.xlabel('$E/m_V$')
    plt.ylabel('$\\rho_\sigma (E)$')
#    plt.legend()
    plt.grid()

    print(LogMessage(),'E0: ', mean1, '+-', dmean1, '\t', '(', mean1*mpi, '+-', dmean1*mpi,')')
    print(LogMessage(),'E1: ', mean2, '+-', dmean2, '\t', '(', mean2*mpi, '+-', dmean2*mpi,')')
    if triple_fit == True:
        print(LogMessage(),'E2: ', mean3, '+-', dmean3, '\t', '(', mean3*mpi, '+-', dmean3*mpi,')')
    if four_fit == True:
        print(LogMessage(),'E3: ', mean4, '+-', dmean4, '\t', '(', mean4*mpi, '+-', dmean4*mpi,')')

    print(LogMessage(), '--- Fit parameters --- ')
    print(LogMessage(), 'Amplitude1: ', amplitude1, '+-', damplitude1, "\t", 'Mean1: ', mean1, '+-', dmean1, '\t', 'Std_dev1: ', stddev1, '+-',dstddev1)
    print(LogMessage(), 'Amplitude2: ', amplitude2, '+-', damplitude2, "\t", 'Mean2: ', mean2, '+-', dmean2, '\t',
          'Std_dev2: ', stddev2, '+-', dstddev2)
    if triple_fit == True:
        print(LogMessage(), 'Amplitude3: ', amplitude3, '+-', damplitude3, "\t", 'Mean3: ', mean3, '+-', dmean3, '\t',
              'Std_dev3: ', stddev3, '+-', dstddev3)
    if four_fit == True:
        print(LogMessage(), 'Amplitude4: ', amplitude4, '+-', damplitude4, "\t", 'Mean4: ', mean4, '+-', dmean4, '\t',
              'Std_dev4: ', stddev4, '+-', dstddev4)
    # Display the plot
    plt.show()


    # Estimation of chi square
    flag_chi2 = True
    if triple_fit == True:
        if four_fit == True:
            chi_square = (rho_central - four_gaussian2(amplitude1, mean1, stddev1, amplitude2, mean2, stddev2,
                                                       amplitude3,
                                                       mean3, stddev3, amplitude4, mean4, stddev4, x)) @ \
                         (inv_cov @ (rho_central - four_gaussian2(amplitude1, mean1, stddev1, amplitude2, mean2,
                                                                  stddev2,
                                                                  amplitude3, mean3, stddev3, amplitude4, mean4,
                                                                  stddev4, x)))
            if len(x) > 12:
                chi_square_red = chi_square / (len(x) - 12)
            else:
                print('Cannot compute Chi square!')
                flag_chi2 = False
        else:
            chi_square = (rho_central - triple_gaussian2(amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, amplitude3,
                                                         mean3, stddev3, x)) @ \
                         (inv_cov @ (rho_central - triple_gaussian2(amplitude1, mean1, stddev1, amplitude2, mean2, stddev2,
                                                                    amplitude3, mean3, stddev3, x)))
            if len(x) > 9:
                chi_square_red = chi_square / (len(x) - 9)
            else:
                print('Cannot compute Chi square!')
                flag_chi2 = False

    else:
        chi_square = (rho_central - double_gaussian2(amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, x)) @\
                     (inv_cov @ (rho_central - double_gaussian2(amplitude1, mean1, stddev1, amplitude2, mean2, stddev2, x)))
        if len(x) >= 6:
            chi_square_red = chi_square / (len(x) - 6)
        else:
            print('Cannot compute Chi square!')
            flag_chi2 = False

    if flag_chi2:
        print('Reduced Chi Square: ', chi_square_red)

if __name__ == "__main__":
    main()
