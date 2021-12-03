
import tensorflow as tf
import pandas as pd
import scipy
import random
import math
import numpy as np

#Function used to find the root mean squared
def rms(array):
    square = 0
    for i in array:
        square += i**2
    mean = square/len(array)
    root_mean = math.sqrt(mean)
    return root_mean
        
            
def spherical_conversion (df):
    # Getting the values that I need from the dataframe
    x_values = df['x']
    y_values = df['y']
    #New cylindrical values
    phi_values = []
    r_values = []
    # So that I can iterate through the two lists at the same time.
    coords = zip(x_values,y_values)
    for x, y in coords:
        phi = math.atan2(y, x)
        r = math.sqrt(x **2 + y**2)
        phi_values.append(phi)
        r_values.append(r)
    # Creating cylindrical copy of input dataframe
    df_cylindrical = df
    #Renaming columns and inputing the new values
    df_cylindrical = df_cylindrical.rename(columns={'x':'r', 'y':'phi'})
    df_cylindrical['r'] = r_values
    df_cylindrical['phi'] = phi_values
    return df_cylindrical


def sort_radius(df, dimension=3, cartesian_values=['x','y','z']):
    df_r = df
    # Creates arrays of zero with dimensions given by the length of the input dataframe
    holder = np.zeros(( len(df.index)))
    print(holder)
    for i in range(dimension):
        holder+= (df_r[cartesian_values[i]])**2
    df_r['radius'] = holder
    df_r.sort_values('radius', inplace=True)
    return df_r
  

#Mark cyl=True if your input is in cylindrical coordinates
# Currently do not use cylindrical function if dimensions are not 3
def train_test_split(df, train=0.90, grouper='particle_id', width=4, shift=1, cyl=False, dimension=3, variables=['x','y','z']):
    # This cuts out the particles that do not enough hits to match up with the width set
    df = track_length(df, width+1)

    #Gets the max absolute values within each column of the dataframe
    max_values = []
    for _ in range(dimension):
        max_values.append( df[variables[_]].abs().max())
    
        
    #This sorts the hits by radius then Creates groupby object grouping by item given in grouper which should still be sorted.
    if cyl:
        sorted_df = df.sort_values("r")
    else:
        sorted_df = sort_radius(df)

    #This normalizes the data between -1 and 1
    for dim in range(dimension):
        sorted_df[variables[dim]] = sorted_df[variables[dim]]/max_values[dim]
    grouped_df = sorted_df.groupby(grouper)
    print('\n Training on this many different groups: \n', grouped_df.ngroups)
    #Gets all the values inside the 'grouper' column and gets the unique values
    id_list = df[grouper].values
    unique_id = unique(id_list)
    n = len(unique_id)

    # This makes the randomness repeatable
    def seed():
        return 0.2
    #Shuffles the unique_ids using the seed to keep randomness consistent
    random.shuffle(unique_id, seed)


    #Empty lists to append to 
    train_data_x = []
    train_data_y = []
    test_data_x = []
    test_data_y = []

    # Creating Train,Test and Valid dataframes
    for i in range(0, int(n*train)):
        train_append = grouped_df.get_group(unique_id[i])
        #Drops every column except for the variables stated in variables
        train_append = train_append[variables]
        #Converts the dataframe to a numpy array 
        train_append = train_append.values
        # The input data is defined by width and the output is defined by shift
        train_append_x = train_append[:width]
        train_append_y = train_append[width:width+shift]

        #To prevent a bug in the code where not all tracks were the same length
        if len(train_append_x) == width:
            train_data_x.append(train_append_x)
            train_data_y.append(train_append_y)
        else:
            print(len(train_append_x))

    #Repeat from above to form train data
            
    for x in range(int(n*train), n):
        test_append = grouped_df.get_group(unique_id[x])
    
        test_append = test_append[variables]
        #test_append.drop('particle_id', inplace=True, axis=1)
        test_append = test_append.values
        test_append_x = test_append[:width]
        test_append_y = test_append[width:width+shift]
        if len(test_append_x) == width:
            test_data_x.append(test_append_x)
            test_data_y.append(test_append_y)
        else:
            print(len(test_append_x))
            
    return train_data_x, train_data_y, test_data_x, test_data_y, max_values


def residual_plot(actual_value, x_data=None, model=None, prediction=None):
    if (model != None and prediction == None):
        prediction = model(x_data)

    residual = prediction - actual_value
    x_resid = []
    y_resid = []
    z_resid = []

    for resid in residual:
        for resid_values in resid:
            x_resid.append(resid_values[0])
            y_resid.append(resid_values[1])
            z_resid.append(resid_values[2])

    max_x = x_resid.index(max(x_resid))
    max_y = y_resid.index(max(y_resid))
    max_z = z_resid.index(max(z_resid))
    min_x = x_resid.index(min(x_resid))
    min_y = y_resid.index(min(y_resid))
    min_z = z_resid.index(min(z_resid))
    plt.hist(x_resid, bins=50)
    plt.title("X_residuals")
    plt.show()

    plt.hist(y_resid, bins=50)
    plt.title('Y_residuals')
    plt.show()

    plt.hist(z_resid, bins=50)
    plt.title('Z_residuals')
    plt.show()

    return [[max_x, min_x], [max_y, min_y], [max_z, min_z]]




def unique(list1):
    # initialize a null list
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def unnormalize(array, max_values, dim):
    for track in array:
        for hit in track:
            for i in range(dim):
                hit[i] = hit[i] * max_values[i]
    return array


def track_length(df, length):
# Creates a dataframe that only has particles with tracks of given length
    grouped = df.groupby('particle_id')
    final = grouped.filter(lambda x: len(x) >= length)
    return final
