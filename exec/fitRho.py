import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from lmfit import Model, Parameters, minimize, Minimizer

sys.path.append("..")
from importall import *


def main():
    # Define the path to the directory containing the files
    path = './tmax24sigma0.1926Ne5nboot2000mNorm0.321prec280/Logs/'
    # Smearing radius in ratio with Mpi
    sigma = 0.6

    # If both false, it's two gaussians fit
    triple_fit = False
    four_fit = False

    if four_fit == True:
        triple_fit = True

    mpi = 0.321

    # Get a list of all the file names in the directory
    file_names = os.listdir(path)

    # Filter the file names to include only those starting with 'RhoSamplesE'
    file_names = [file_name for file_name in file_names if file_name.startswith('RhoSamplesE')]

    # Extract the energy values from the file names
    energies = [file_name.split('E')[1].split('sig')[0] for file_name in file_names]

    # Sort the energies in ascending order
    energies.sort()

    # Define the dimensions of the matrix
    nboot = 2000
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
        file_name = f'RhoSamplesE{energy}sig0.1926'
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
    # Define the sum of two Gaussian functions
    def triple_gaussian(params, x):
        amplitude_1 = params['amplitude_1'].value
        mean_1 = params['mean_1'].value
        amplitude_2 = params['amplitude_2'].value
        mean_2 = params['mean_2'].value
        amplitude_3 = params['amplitude_3'].value
        mean_3 = params['mean_3'].value

        model = gaussian(x, amplitude_1, mean_1) + gaussian(x, amplitude_2, mean_2) + gaussian(x, amplitude_3, mean_3)
        return model
    def triple_gaussian2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, x):
        model = gaussian(x, amplitude1, mean1) + gaussian(x, amplitude2, mean2) + gaussian(x,
                                                                                           amplitude3,
                                                                                           mean3)
        return model
    # Define the sum of two Gaussian functions
    def four_gaussian(params, x):
        amplitude_1 = params['amplitude_1'].value
        mean_1 = params['mean_1'].value
        amplitude_2 = params['amplitude_2'].value
        mean_2 = params['mean_2'].value
        amplitude_3 = params['amplitude_3'].value
        mean_3 = params['mean_3'].value
        amplitude_4 = params['amplitude_4'].value
        mean_4 = params['mean_4'].value

        model = gaussian(x, amplitude_1, mean_1) + gaussian(x, amplitude_2, mean_2) + gaussian(x, amplitude_3,
                                                                                               mean_3) + gaussian(x,
                                                                                                                  amplitude_4,
                                                                                                                  mean_4)
        return model
    def four_gaussian2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, amplitude4, mean4, x):
        model = gaussian(x, amplitude1, mean1) + gaussian(x, amplitude2, mean2) + gaussian(x,
                                                                                           amplitude3,
                                                                                           mean3, ) + gaussian(x,
                                                                                                               amplitude4,
                                                                                                               mean4)
        return model


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

        if triple_fit == True:
            model = triple_gaussian2(ampl_1, e0, ampl_2, e1, ampl_3, e2, x)
        if four_fit == True:
            model = four_gaussian2(ampl_1, e0, ampl_2, e1, ampl_3, e2, ampl_4, e3, x)

        diff = abs(model - data)
        residual = cov_inv.dot(diff)
        return residual

    def correlated_residual(amplitude1,mean1,amplitude2,mean2, x, data, choelesky):
        cov_inv = np.linalg.inv(choelesky)
        model = double_gaussian2(amplitude1,mean1,amplitude2,mean2,x)
        diff = abs(model - data)
        residual = cov_inv.dot(diff)
        sum = residual.dot(residual)
        return sum
    def correlated_residual_three(amplitude1,mean1,amplitude2,mean2,amplitude3,mean3, x, data, choelesky):
        cov_inv = np.linalg.inv(choelesky)
        model = triple_gaussian2(amplitude1,mean1,amplitude2,mean2,amplitude3,mean3,x)
        diff = abs(model - data)
        residual = cov_inv.dot(diff)
        sum = residual.dot(residual)
        return sum
    def correlated_residual_four(amplitude1,mean1,amplitude2,mean2,amplitude3,mean3,amplitude4,mean4, x, data, choelesky):
        cov_inv = np.linalg.inv(choelesky)
        model = four_gaussian2(amplitude1,mean1,amplitude2,mean2,amplitude3,mean3,amplitude4,mean4,x)
        diff = abs(model - data)
        residual = cov_inv.dot(diff)
        sum = residual.dot(residual)
        return sum

    choleskyCov = np.linalg.cholesky(cov_matrix)

    for k in range(nboot):
        y = rho_resampled[k, :]
        # Plot the data
#        plt.errorbar(x, y, yerr=drho_central, fmt='o', label='Data', markersize=3.0, elinewidth=1.2)

        # Create lmfit Parameters object
        params = Parameters()
        params.add('amplitude_1', value=1e-7, min=0)
        params.add('mean_1', value=1.0, min=0.8, max=1.2)
        params.add('amplitude_2', value=5e-8, min=0)
        params.add('mean_2', value=2.0, min=1.0, max=4.0)

        if triple_fit == True:
            params.add('amplitude_3', value=1e-8, min=0)
            params.add('mean_3', value=3.0, min=2.0, max=4.0)
        if four_fit == True:
            params.add('amplitude_4', value=0.1e-7)
            params.add('mean_4', value=3.7, min=3.0)


        FITWrapper_corr = Minimizer(chisq_correlated, params, fcn_args=(x,y,choleskyCov))
        result = FITWrapper_corr.minimize()

        # Minimize the residual function
#        result = minimize(residual, params, args=(x, y, inv_cov))

        # Generate the fitted curve
        x_fit = np.linspace(min(x), max(x)+0.2, 1000)
        y_fit = double_gaussian(result.params, x_fit)


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

        '''
        # Plot the individual Gaussian components
        y_gaussian_1 = gaussian(x_fit, result.params['amplitude_1'], result.params['mean_1'])
        y_gaussian_2 = gaussian(x_fit, result.params['amplitude_2'], result.params['mean_2'])
        plt.plot(x_fit, y_gaussian_1, label='Gaussian 1', color='orange', linewidth=1.3, alpha=0.2)
        plt.plot(x_fit, y_gaussian_2, label='Gaussian 2', color='blue',linewidth=1.3, alpha=0.2)
        if triple_fit == True:
            y_gaussian_3 = gaussian(x_fit, result.params['amplitude_3'], result.params['mean_3'])
            plt.plot(x_fit, y_gaussian_3, label='Gaussian 3', color='gray', linewidth=1.3, alpha=0.2)
        if four_fit == True:
            y_gaussian_4 = gaussian(x_fit, result.params['amplitude_4'], result.params['mean_4'])
            plt.plot(x_fit, y_gaussian_4, label='Gaussian 4', color='brown', linewidth=1.3, alpha=0.2)
        '''
#    for k in range(10):
#        print('Amplitude1: ', amplitude_vals1[k], '\tMean1: ', mean_vals1[k])


    amplitude1 = np.average(amplitude_vals1)
    mean1 = np.average(mean_vals1)
    damplitude1 = np.std(amplitude_vals1)
    dmean1 = np.std(mean_vals1)

    amplitude2 = np.average(amplitude_vals2)
    mean2 = np.average(mean_vals2)
    damplitude2 = np.std(amplitude_vals2)
    dmean2 = np.std(mean_vals2)

    if triple_fit == True:
        amplitude3 = np.average(amplitude_vals3)
        mean3 = np.average(mean_vals3)
        damplitude3 = np.std(amplitude_vals3)
        dmean3 = np.std(mean_vals3)

    if four_fit == True:
        amplitude4 = np.average(amplitude_vals4)
        mean4 = np.average(mean_vals4)
        damplitude4 = np.std(amplitude_vals4)
        dmean4 = np.std(mean_vals4)

    x1 = np.linspace(min(x), max(x)+0.2, 1000)

    # Calculate the upper and lower error bands
    upper_band1 = abs(amplitude1 + damplitude1) * np.exp(-((x1 - abs(mean1 + dmean1)) ** 2) / (2 * (sigma) ** 2))

    lower_band1 = abs(amplitude1 - damplitude1) * np.exp(-((x1 - abs(mean1 - dmean1)) ** 2) / (2 * (sigma) ** 2))


    plt.plot(x1, gaussian(x1,amplitude1,mean1), color='blue')
    plt.fill_between(x1, lower_band1, upper_band1, color='blue', alpha=0.4, label='Error Bands')

    upper_band2 = abs(amplitude2 + damplitude2) * np.exp(-((x1 - abs(mean2 + dmean2)) ** 2) / (2 * (sigma) ** 2))

    lower_band2 = abs(amplitude2 - damplitude2) * np.exp(-((x1 - abs(mean2 - dmean2)) ** 2) / (2 * (sigma) ** 2))

    plt.plot(x1, gaussian(x1, amplitude2, mean2), color='green')
    plt.fill_between(x1, lower_band2, upper_band2, color='green', alpha=0.4, label='Error Bands')



    if triple_fit == True:
        # Calculate the upper and lower error bands
        upper_band4 = abs(amplitude3 + damplitude3) * np.exp(
            -((x1 - abs(mean3 + dmean3)) ** 2) / (2 * (sigma) ** 2))

        lower_band4 = abs(amplitude3 - damplitude3) * np.exp(
            -((x1 - abs(mean3 - dmean3)) ** 2) / (2 * (sigma) ** 2))

        plt.plot(x1, gaussian(x1, amplitude3, mean3), color='purple')
        plt.fill_between(x1, lower_band4, upper_band4, color='purple', alpha=0.4, label='Error Bands')

        if four_fit == True:
            # Calculate the upper and lower error bands
            upper_band5 = abs(amplitude4 + damplitude4) * np.exp(
                -((x1 - abs(mean4 + dmean4)) ** 2) / (2 * (sigma) ** 2))

            lower_band5 = abs(amplitude4 - damplitude4) * np.exp(
                -((x1 - abs(mean4 - dmean4)) ** 2) / (2 * (sigma) ** 2))

            plt.plot(x1, gaussian(x1, amplitude4, mean4, sigma), color='brown')
            plt.fill_between(x1, lower_band5, upper_band5, color='brown', alpha=0.4, label='Error Bands')

            plt.plot(x1, four_gaussian2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, amplitude4, mean4, x1), color='gray')

            upper_band3 = (amplitude1 + damplitude1) * np.exp(
                -((x1 - (mean1 + dmean1)) ** 2) / (2 * (sigma) ** 2)) + (
                                      amplitude2 + damplitude2) * np.exp(
                -((x1 - (mean2 + dmean2)) ** 2) / (2 * (sigma) ** 2)) + (
                                      amplitude3 + damplitude3) * np.exp(
                -((x1 - (mean3 + dmean3)) ** 2) / (2 * (sigma) ** 2)) + (
                                      amplitude4 + damplitude4) * np.exp(
                -((x1 - (mean4 + dmean4)) ** 2) / (2 * (sigma) ** 2))
            lower_band3 = (amplitude1 - damplitude1) * np.exp(
                -((x1 - abs(mean1 - dmean1)) ** 2) / (2 * (sigma) ** 2)) + abs(
                                      amplitude2 - damplitude2) * np.exp(
                -((x1 - abs(mean2 - dmean2)) ** 2) / (2 * (sigma) ** 2)) + abs(
                                      amplitude3 - damplitude3) * np.exp(
                -((x1 - abs(mean3 - dmean3)) ** 2) / (2 * (sigma) ** 2)) + abs(
                                      amplitude4 - damplitude4) * np.exp(
                -((x1 - abs(mean4 - dmean4)) ** 2) / (2 * (sigma) ** 2))

            plt.fill_between(x1, lower_band3, upper_band3, color='gray', alpha=0.4, label='Error Bands')
        else:
            plt.plot(x1, triple_gaussian2(amplitude1, mean1, amplitude2, mean2, amplitude3, mean3, x1), color='gray')

            upper_band3 = (amplitude1 + damplitude1) * np.exp(
                -((x1 - (mean1 + dmean1)) ** 2) / (2 * (sigma) ** 2)) + (amplitude2 + damplitude2) * np.exp(
                -((x1 - (mean2 + dmean2)) ** 2) / (2 * (sigma) ** 2)) + (amplitude3 + damplitude3) * np.exp(
                -((x1 - (mean3 + dmean3)) ** 2) / (2 * (sigma) ** 2))

            lower_band3 = abs(amplitude1 - damplitude1) * np.exp(
                -((x1 - abs(mean1 - dmean1)) ** 2) / (2 * (sigma) ** 2)) + abs(amplitude2 - damplitude2) * np.exp(
                -((x1 - abs(mean2 - dmean2)) ** 2) / (2 * (sigma) ** 2)) + abs(amplitude3 - damplitude3) * np.exp(
                -((x1 - abs(mean3 - dmean3)) ** 2) / (2 * (sigma) ** 2))

            plt.fill_between(x1, lower_band3, upper_band3, color='gray', alpha=0.4, label='Error Bands')

    else:
        plt.plot(x1, double_gaussian2(amplitude1, mean1, amplitude2, mean2, x1), color='gray')

        upper_band3 = (amplitude1 + damplitude1) * np.exp(-((x1 - (mean1 + dmean1)) ** 2) / (2 * (sigma) ** 2)) + (amplitude2 + damplitude2) * np.exp(-((x1 - (mean2 + dmean2)) ** 2) / (2 * (sigma) ** 2))
        lower_band3 = (amplitude1 - damplitude1) * np.exp(-((x1 - (mean1 - dmean1)) ** 2) / (2 * (sigma) ** 2)) + (amplitude2 - damplitude2) * np.exp(-((x1 - (mean2 - dmean2)) ** 2) / (2 * (sigma) ** 2))

        plt.fill_between(x1, lower_band3, upper_band3, color='gray', alpha=0.4, label='Error Bands')

    # Customize the plot
    plt.xlabel('$E/m_{\pi}$')
    plt.ylabel('$\\rho_\sigma (E)$')
#    plt.legend()
    plt.grid()

    print(LogMessage(),'E0: ', mean1, '+-', dmean1, '\t', '(', mean1*mpi, '+-', dmean1*mpi,')')
    print(LogMessage(),'E1: ', mean2, '+-', dmean2, '\t', '(', mean2*mpi, '+-', dmean2*mpi,')')

    if triple_fit == True:
        print(LogMessage(),'E2: ', mean3, '+-', dmean3, '\t', '(', mean3*mpi, '+-', dmean3*mpi,')')
    if four_fit == True:
        print(LogMessage(),'E3: ', mean4, '+-', dmean4, '\t', '(', mean4*mpi, '+-', dmean4*mpi,')')

    print(LogMessage(), 'E1/E0: ', mean2 / mean1, '+-',
          np.sqrt((dmean2 / mean1) ** 2 + (dmean1 * mean2 / mean1 ** 2) ** 2))

    print(LogMessage(), '--- Fit parameters --- ')
    print(LogMessage(), 'Amplitude1: ', amplitude1, '+-', damplitude1, "\t", 'Mean1: ', mean1, '+-', dmean1)
    print(LogMessage(), 'Amplitude2: ', amplitude2, '+-', damplitude2, "\t", 'Mean2: ', mean2, '+-', dmean2)

    if triple_fit == True:
        print(LogMessage(), 'Amplitude3: ', amplitude3, '+-', damplitude3, "\t", 'Mean3: ', mean3, '+-', dmean3)
    if four_fit == True:
        print(LogMessage(), 'Amplitude4: ', amplitude4, '+-', damplitude4, "\t", 'Mean4: ', mean4, '+-', dmean4)
    # Display the plot
    plt.show()


    # Estimation of chi square
    flag_chi2 = True
    if triple_fit == True:
        if four_fit == True:
            if len(x) > 8:
                chi_square_red = correlated_residual_four(amplitude1,mean1,amplitude2,mean2,amplitude3,mean3,amplitude4,mean4, x, rho_central, choleskyCov) / (len(x) - 8)
            else:
                print('Cannot compute Chi square!')
                flag_chi2 = False
        else:
            if len(x) > 6:
                chi_square_red = correlated_residual_three(amplitude1,mean1,amplitude2,mean2,amplitude3,mean3,x, rho_central, choleskyCov) / (len(x) - 6)
            else:
                print('Cannot compute Chi square!')
                flag_chi2 = False

    else:
        if len(x) > 4:
            chi_square_red = correlated_residual(amplitude1,mean1,amplitude2,mean2, x, rho_central, choleskyCov) / (len(x) - 4)
        else:
            print('Cannot compute Chi square!')
            flag_chi2 = False

    if flag_chi2:
        print('Reduced Chi Square (with correlation): ', chi_square_red)
        chi_square_red = 0.0
        model = double_gaussian2(amplitude1,mean1,amplitude2,mean2,x)
        for i in range(len(x)):
            chi_square_red += ((rho_central[i] - model[i]) **2 ) / drho_central[i]**2
        print('Reduced Chi Square (no correlation): ', chi_square_red)

if __name__ == "__main__":
    main()
