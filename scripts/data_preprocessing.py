import numpy as np
import pandas as pd
import math
import sys
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial

def processData(path):
    #replace with the path on your pc
    data = pd.read_csv(path)
    dataset = data.to_numpy()

    healthy = dataset[dataset[:,-1]==0]
    left = dataset[dataset[:,-1]==1]
    right = dataset[dataset[:,-1]==2]

    healthy = np.reshape(healthy, (-1, 270, 4))
    left = np.reshape(left, (-1, 270, 4))
    right = np.reshape(right, (-1, 270, 4))

    eta = 0.025
    noise_functions = []

    for i in range(healthy.shape[0]):
        run = np.reshape(healthy[i, :, :], (-1, 4))
        noise_function = cleanData(run, eta)
        noise_functions.append(np.concatenate((noise_function, 0), axis=None))

    for j in range(left.shape[0]):
        run = np.reshape(left[j, :, :], (-1, 4))
        noise_function = cleanData(run, eta)
        noise_functions.append(np.concatenate((noise_function, 1), axis=None))

    for k in range(right.shape[0]):
        run = np.reshape(right[k, :, :], (-1, 4))
        noise_function = cleanData(run, eta)
        noise_functions.append(np.concatenate((noise_function, 2), axis=None))

    # run = []
    # noise_functions = []
    # last_iteration = 0.0
    # label = 0
    # count = 0
    # data = []
    #
    # for i in range(dataset.shape[0]):
    #     if dataset[i, 1] < last_iteration or i == dataset.shape[0] - 1:
    #         run = np.array(run)
    #         run = np.reshape(run, (-1, 7))
    #
    #         # TODO: clean run and calculate statistics
    #         eta = 0.05
    #         noise_function = cleanData(run, eta)
    #         noise_functions.append(np.concatenate((noise_function, label), axis=None))
    #         print(noise_function)
    #
    #         if label == 0:
    #             data.append(run)
    #         elif label == 1:
    #             data.append(run)
    #         else:
    #             data.append(run)
    #
    #         count += 1
    #         if count % 4 == 0:
    #             label += 1
    #
    #         last_iteration = dataset[i, 1]
    #         run = []
    #         run.append(dataset[i, :])
    #     else:
    #         run.append(dataset[i, :])
    #         last_iteration = dataset[i, 1]

    noise_functions = np.array(noise_functions)
    print(noise_functions)
    healthy_distribution, left_distribution, right_distribution = computeDistribution(noise_functions)
    distribution_data = np.array([healthy_distribution, left_distribution, right_distribution])
    print(distribution_data.shape)

    # data = np.array(data)
    # print(data.shape)
    # for i in range(noise_functions.shape[0]):
    #     run = data[i]
    #     print(np.reshape(run, (-1, 7)).shape)
    #     y_hat = func(run[:, 2], *noise_functions[i, :-1])
    #     plt.scatter(run[:, 2], run[:, 3], color='red', marker='p')
    #     plt.scatter(run[:, 2], y_hat, color='blue', marker='p')
    #     plt.show()

    # f = open('/home/ace/catkin_ws/src/network_faults/data/noise_functions.csv', 'w')
    # np.savetxt(f, noise_functions, delimiter=",")
    #
    # f = open('/home/ace/catkin_ws/src/network_faults/data/distributions.csv', 'w')
    # np.savetxt(f, distribution_data, delimiter=",")

def func(x, a, b, c):
    return a*np.exp(b*x)+c

def norm(x, x_hat):
    norm_vector = np.empty((x.shape[0], 1))
    for i in range(norm_vector.shape[0]):
        norm_vector[i] = np.linalg.norm(x[i, 1] - x_hat[i])

    return norm_vector

def zscoreOutliers(data, norm, iter):
    z = np.abs(stats.zscore(norm))
    data_out = data[(z < (3-iter*0.005)).all(axis=1)]

    return data_out

def iqrOutliers(data, norm, pct):
    q3, q1 = np.percentile(norm, [100-pct, pct])
    iqr = q3 - q1

    data_out = data[((norm > (q1 - 1.5*iqr)) | (norm < (q3 + 1.5*iqr))).all(axis=1)]
    # data_out = data_out[(norm > (q25 - 1.5*iqr)).all(axis=1)]

    return data_out

def cleanData(data, eta):
    print("New Run")
    run_norm = 100.0
    x = data
    iteration = 0
    max_iter = 10
    range = 3.0
    initial_guess = np.array([0.1, 0.01, 0.01])
    upper_bounds = range*initial_guess
    lower_bounds = -range*initial_guess
    count = 0.0

    while run_norm >= eta and iteration != max_iter and x.shape[0] > 50:
        # range = range + range*(iteration*0.5)
        upper_bounds = initial_guess + range*np.fabs(initial_guess)
        lower_bounds = initial_guess - range*np.fabs(initial_guess)
        print("range: %s" % range)
        print("initial guess")
        print(initial_guess)
        print("upper bounds")
        print(upper_bounds)
        print("lower bounds")
        print(lower_bounds)
        # print("count: %s" % count)
        # print("run norm: %s" % run_norm)

        try:
            popt, pcov = curve_fit(func, x[:, 0], x[:, 1], bounds=(lower_bounds, upper_bounds))
            y_hat = func(x[:, 0], *popt)
            # plt.scatter(x[:, 2], x[:, 3], color='red', marker='p')
            # plt.scatter(x[:, 2], y_hat, color='blue', marker='p')

            norm_vector = norm(x, y_hat)

            data_zscore = zscoreOutliers(x, norm_vector, iteration)
            data_iqr = iqrOutliers(x, norm_vector, 25)

            zscore_popt, zscore_pcov = curve_fit(func, data_zscore[:, 0], data_zscore[:, 1], bounds=(lower_bounds, upper_bounds))
            zscore_y_hat = func(data_zscore[:, 0], *zscore_popt)

            iqr_popt, iqr_pcov = curve_fit(func, data_iqr[:, 0], data_iqr[:, 1], bounds=(lower_bounds, upper_bounds))
            iqr_y_hat = func(data_iqr[:, 0], *iqr_popt)

            zscore_norm_vector = norm(data_zscore, zscore_y_hat)
            iqr_norm_vector = norm(data_iqr, iqr_y_hat)

            zscore_norm = np.linalg.norm(data_zscore[:, 1] - zscore_y_hat)
            iqr_norm = np.linalg.norm(data_iqr[:, 1] - iqr_y_hat)
            # print("Zscore norm: %s" % zscore_norm)
            # print("IQR norm: %s" % iqr_norm)

            print(x.shape)
            print(data_zscore.shape)
            print(data_iqr.shape)
            print(zscore_y_hat.shape)
            print(iqr_y_hat.shape)

            if zscore_norm < iqr_norm:
                plt.scatter(data_zscore[:, 0], data_zscore[:, 1], color='red', label="sample data: iteration %s" % iteration)
                plt.scatter(data_zscore[:, 0], zscore_y_hat, color='blue', label="fitted data: iteration %s" % iteration)
                plt.title("Zscore Filtered Data")
                plt.legend()

                run_opt = zscore_popt
                initial_guess = run_opt
                run_norm = zscore_norm
                x = data_zscore
                iteration += 1
            else:
                plt.scatter(data_iqr[:, 0], data_iqr[:, 1], color='red', label="sample data: iteration %s" % iteration)
                plt.scatter(data_iqr[:, 0], iqr_y_hat, color='blue', label="fitted data: iteration %s" % iteration)
                plt.title("IQR Filtered Data")
                plt.legend()

                run_opt = iqr_popt
                initial_guess = run_opt
                run_norm = iqr_norm
                x = data_iqr
                iteration += 1

            count += 1

            plt.show()
        except RuntimeError:
            print("Error - curve_fit failed")

    return run_opt

def computeDistribution(x):
    healthy_data = x[(x[:, -1] == 0)]
    left_data = x[(x[:, -1] == 1)]
    right_data = x[(x[:, -1] == 2)]

    print(np.std(healthy_data[:, 0], dtype=np.float64))

    healthy_distribution = np.concatenate((np.mean(healthy_data[:, 0]), np.mean(healthy_data[:, 1]), np.mean(healthy_data[:, 2]),
                                    np.std(healthy_data[:, 0]), np.std(healthy_data[:, 1]), np.std(healthy_data[:, 2]), healthy_data[0, -1]), axis=None)
    left_distribution = np.concatenate((np.mean(left_data[:, 0]), np.mean(left_data[:, 1]), np.mean(left_data[:, 2]),
                                    np.std(left_data[:, 0]), np.std(left_data[:, 1]), np.std(left_data[:, 2]), left_data[0, -1]), axis=None)
    right_distribution = np.concatenate((np.mean(right_data[:, 0]), np.mean(right_data[:, 1]), np.mean(right_data[:, 2]),
                                    np.std(right_data[:, 0]), np.std(right_data[:, 1]), np.std(right_data[:, 2]), right_data[0, -1]), axis=None)
    print("Healthy Distribution")
    print(healthy_distribution)
    print("Left Distribution")
    print(left_distribution)
    print("Right Distribution")
    print(right_distribution)

    return healthy_distribution, left_distribution, right_distribution

if __name__ == '__main__':
    path = "~/catkin_ws/src/network_faults/data/noise_data.csv"
    processData(path)
