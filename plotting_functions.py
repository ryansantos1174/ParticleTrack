import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import os
import numpy as np
from functions import *

def gauss_curve(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0)**2 / (2 * sigma ** 2))


def Plot_All_Results(model, valid_data_x, valid_data_y, max_values, dimension=3,  ntracks=10, save_dir=None, SAVE=False, bad_points=False):
   
  
    # Getting presdictions and unnormalizing the data
    prediction = model.predict(valid_data_x)
    prediction = unnormalize(prediction, max_values, dimension)
    valid_data_y = unnormalize(valid_data_y, max_values, dimension)
    valid_data_x = unnormalize(valid_data_x, max_values, dimension)

    # Calculating and formatting residuals
    print('length of valid_data_y \n', len(valid_data_y))
    residual = prediction - valid_data_y
    print('length of prediction \n', len(prediction))
    print('length of residual \n', len(residual))
    print('shape of predictions', prediction.shape)


    # Creating a numpy array of zeros to hold data which allows the dimesions of the data to be whatever
    empty_array = np.zeros((dimension, len(valid_data_y)))

    # This allows the code to format the residuals for data of any dimensions   
    for index, val in enumerate(residual):
        for coord in val:
            for i in range(dimension):
                empty_array[i][index] = coord[i]

                
    # This is supposed to find the highest residual value track and then give points that have comparable resdiual values
    for dim in range(dimension):
        if bad_points:
            for val in empty_array[dim]:
                if abs(val) > max(empty_array[dim], key=abs) * 0.9:
                    val_index = empty_array[dim].index(val)
                    print('\n These are the data points that cause high residual in x',
                      valid_data_x[val_index])


    # Formatting histogram
    for dim in range(dimension):
        plt.clf()
        n, bins, _ = plt.hist(empty_array[dim], bins=50)
        bins = bins[:-1]
        possible_values = ['x', 'y', 'z']
        plt.title(possible_values[dim])
        #Fitting gaussian curve over data
        popt, pcov = curve_fit(gauss_curve, bins, n)
        plt.plot(bins, gauss_curve(bins, *popt), 'r-')
        

        # Saving the data and saving the configuration of the model
        if ((SAVE) and (save_dir != None)):
            resid_dir = os.path.join(save_dir, f'{possible_values[dim]}_resid_hist.png')
            plt.savefig(resid_dir)
        plt.show()

    # Plotting Tracks
    valid_data_y = np.squeeze(valid_data_y)
    prediction = np.squeeze(prediction)

    #This necessarily needs at least two data points
    # If the dimension is less than two then the only thing I can graph is a histogram which is already plotted. 
    if dimension >= 2:
        for index, track in enumerate(valid_data_x):
            if index < ntracks:
                for hit in track:
                    plt.scatter(hit[0], hit[1], color='blue')
                    plt.scatter(valid_data_y[index][0],
                        valid_data_y[index][1], color='green')
                    plt.scatter(prediction[index][0],
                        prediction[index][1], color='red')
            if SAVE:
                plt.savefig(os.path.join(save_dir, f'track{index}.png'))
            plt.show()
