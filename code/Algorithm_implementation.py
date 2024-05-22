import scipy
from spectral import *
import numpy as np
import matplotlib.pyplot as plt
import spectral.io.envi as envi
from scipy.linalg import sqrtm, inv


mathod_chronochrome_use_0_for_normolize_y_Pfa = "initial"
mathod_chronochrome_use_0_for_normolize_y_Pd =  "initial"
mathod_equalization_use_0_for_normolize_y_Pfa = "initial"
mathod_equalization_use_0_for_normolize_y_Pd = "initaial"
mathod_chronochrome_use_m_for_normolize_y_inovation_pfa = "initaial"
mathod_chronochrome_use_m_for_normolize_y_inovation_Pd = "initaial"
mathod_equalization_use_m_for_normolize_y_inovation_Pfa = "initaial"
mathod_equalization_use_m_for_normolize_y_inovation_Pd = "initaial"

def m (data,row,col):
    """
    Calculate the average of neighboring elements for a specific position in a 3D matrix.

    This function handles boundary cases and adjusts the number of neighbors considered based on the location within the matrix (edges, corners, or center).

    Parameters:
        data (np.ndarray): A 3D numpy array from which to calculate the average.
        row (int): The row index of the element for which the average is calculated.
        col (int): The column index of the element for which the average is calculated.

    Returns:
        float: The average value of the neighboring elements.
    """
    data_int32 = data

    if (row == 0):
        if(col == 0) :
            return ((data_int32[row +1][col] + data_int32[row+1][col+1] +data_int32[row][col+1])/3) # for averge of data[0][0] we take only 3 nieghbour data[0][1] data[1][0] data[1][1]
        elif (col == data.shape[1] - 1): # data[0][49]
            return ((data_int32[row][col - 1] + data_int32[row + 1][col - 1] + data_int32[row+1][col]) / 3) #data[0][48] data[1][48] data[1][49]
        else: # row is zero and we run from 1 to col - 1
            return ((data_int32[row][col - 1]+data_int32[row][col + 1]+data_int32[row+1][col - 1]+data_int32[row+1][col]+data_int32[row+1][col+1])/5)
    elif(row == data.shape[0]-1): # mow we run in the last row
        if(col == 0): #data[49][1] data[48][0] data[48][1]
            return ((data_int32[row][col+1] + data_int32[row-1][col] + data_int32[row-1][col+1]) / 3)
        elif (col == data.shape[1] - 1): #data[49][48] data[48][49] data[48][48]
            return ((data_int32[row][col-1] + data_int32[row-1][col] + data_int32[row - 1][col-1]) / 3)
        else: # 49 (the last one )
            return ((data_int32[row-1][col - 1]+data_int32[row-1][col]+data_int32[row-1][col+1]+data_int32[row][col-1]+data_int32[row][col+1])/5)
    elif((col == 0 and row != 0 )or col == 0 and row != data.shape[0] -1): #now we run on the first col
        return ((data_int32[row-1][col]+data_int32[row-1][col+1]+data_int32[row][col+1]+data_int32[row+1][col]+data_int32[row+1][col+1])/5)

    elif ((col == data.shape[1] - 1 and row != 0) or (data.shape[1] - 1 == 0 and row != data.shape[0] - 1)):  # now we run on the last col
        return ((data_int32[row-1][col-1]+data_int32[row-1][col]+data_int32[row][col-1]+data_int32[row+1][col-1]+data_int32[row+1][col])/5)
    else :
        return ((data_int32[row-1][col-1]+data_int32[row-1][col]+data_int32[row-1][col+1]+data_int32[row][col-1]+data_int32[row][col+1]+data_int32[row+1][col-1]+data_int32[row+1][col]+data_int32[row+1][col+1])/8)

def get_m_matrix(data):
    """
    Generate a matrix where each element is the average of its neighbors based on the `m` function.

    This function iterates through each element of the input 3D matrix, applies the `m` function to get the average
    of the neighbors considering edge and corner cases, and stores the result in a new matrix.

    Parameters:
        data (np.ndarray): A 3D numpy array from which to compute the averages.

    Returns:
        np.ndarray: A new 3D numpy array with the average values at each position.
    """
    # Initialize the matrix to store the average values
    m_matrix = np.zeros((data.shape[0], data.shape[1],data.shape[2]))
    # Loop through each element in the matrix to calculate the average of its neighbors
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            m_matrix[i][j] = m(data,i,j)
    return m_matrix

def normalize_bodner(data):
    normalize_matrix = np.zeros((data.shape[0], data.shape[1],data.shape[2]))
    averg = np.zeros(data.shape[2])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            averg = averg +  data[i][j]
    averg = averg/(data.shape[0]*data.shape[1])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            normalize_matrix[i][j] = data[i][j] - averg

    return normalize_matrix

def normalize_noam(data):
    """
    Normalize each element in a 3D matrix by subtracting the average of its neighbors.

    This function applies normalization to each element of the input matrix by subtracting the average value
    of its neighbors, determined by the `m` function, which considers boundary conditions and computes the average accordingly.

    Parameters:
        data (np.ndarray): A 3D numpy array to be normalized.

    Returns:
        np.ndarray: A normalized 3D numpy array where each element has been adjusted by the average of its neighbors.
    """
    # Initialize the matrix to store the normalized values
    normalize_matrix = np.zeros((data.shape[0], data.shape[1],data.shape[2]))
    # Loop through each element in the matrix to apply normalization
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            normalize_matrix[i][j] = (data[i][j] - m(data,i,j))
    return normalize_matrix
def normalize_bodner(data):
    """
    Normalize a 3D dataset by subtracting the average value of each spectral band from each pixel.

    Parameters:
    data (numpy.ndarray): A 3D numpy array of shape (height, width, spectral bands).

    Returns:
    numpy.ndarray: A 3D numpy array with the same shape as `data`, containing the normalized values.

    Description:
    This function normalizes the input 3D dataset, which is typically an image or a collection of images
    with multiple spectral bands. The normalization process involves the following steps:

    1. Calculate the average value of each spectral band across all pixels.
    2. Subtract the average value from each pixel's corresponding spectral band value.

    The result is a new 3D array where each pixel's spectral band value is adjusted by removing the
    average value, effectively centering the data around zero for each spectral band.
    """
    normalize_matrix = np.zeros((data.shape[0], data.shape[1],data.shape[2]))
    averg = np.zeros(data.shape[2])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            averg = averg +  data[i][j]
    averg = averg/(data.shape[0]*data.shape[1])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            normalize_matrix[i][j] = data[i][j] - averg

    return normalize_matrix



def covariance(data):
    """
    Compute the covariance matrix for a 3D dataset by averaging the outer products of vectors at each position.

    This function computes the covariance matrix for a 3D dataset (e.g., image data with multiple bands). For each
    pixel or voxel, represented as a vector, it calculates the outer product with itself and accumulates these products
    across the entire dataset to obtain an average outer product matrix, which serves as the covariance matrix.

    Parameters:
        data (np.ndarray): A 3D numpy array where each (i, j) position contains a vector of features.

    Returns:
        np.ndarray: A 2D numpy array representing the covariance matrix of the vectors contained in the input data.
    """
    # Initialize the matrix to store the sum of outer products
    outer_product_matrix = np.zeros((data.shape[2], data.shape[2]))

    # Loop through each element (vector) in the matrix to compute the outer product and accumulate it
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            vec_x = data[i][j]
            outer_product_matrix += np.outer(vec_x, vec_x)

    # Normalize the outer product matrix by the total number of elements to get the covariance matrix
    return outer_product_matrix / (data.shape[0] * data.shape[1])

def covariance_x_y(matrix_x, matrix_y):
    """
    Compute the cross-covariance matrix between two sets of 3D data.

    This function calculates the cross-covariance matrix by taking the outer product of corresponding vectors
    from two datasets at each position (i, j) and averaging these products over all positions. It is useful for
    assessing the linear relationship between two datasets where each (i, j) index in the first two dimensions
    corresponds to a vector in the third dimension.

    Parameters:
        matrix_x (np.ndarray): A 3D numpy array representing the first dataset, where each (i, j) contains a vector.
        matrix_y (np.ndarray): A 3D numpy array representing the second dataset, where each (i, j) contains a vector.

    Returns:
        np.ndarray: A 2D numpy array representing the cross-covariance matrix between vectors of matrix_x and matrix_y.
    """
    # Initialize the matrix to store the sum of outer products
    outer_product_matrix = np.zeros((matrix_x.shape[2], matrix_x.shape[2]))

    # Loop through each element (vector) in the matrices to compute the outer product and accumulate it
    for i in range(matrix_x.shape[0]):
        for j in range(matrix_x.shape[1]):
            vec_x = matrix_x[i][j]
            vec_y = matrix_y[i][j]
            outer_product_matrix += np.outer(vec_y, vec_x)

    # Normalize the outer product matrix by the total number of elements to get the cross-covariance matrix
    return outer_product_matrix / (matrix_x.shape[0] * matrix_x.shape[1])


def L_covariance_Equalization(matrix_x, matrix_y):
    """
    Compute a transformation matrix L that transforms matrix X to approximate matrix Y using covariance equalization.

    This function calculates the matrix L by taking the square root of matrix Y and the inverse of the square root of matrix X,
    and then multiplying these two matrices. This technique is useful in statistical applications where one wishes to transform
    one covariance matrix to approximate another, often used in signal processing and image recognition.

    Parameters:
        matrix_x (np.ndarray): A 2D numpy array representing the covariance matrix of dataset X.
        matrix_y (np.ndarray): A 2D numpy array representing the covariance matrix of dataset Y.

    Returns:
        np.ndarray: A 2D numpy array representing the transformation matrix L that when applied to X, approximates Y.
    """
    # Compute the square root of matrix Y and the inverse of the square root of matrix X
    Y_sqrt = sqrtm(matrix_y)
    X_sqrt_inv = inv(sqrtm(matrix_x))

    # Compute the transformation matrix L
    L = Y_sqrt @ X_sqrt_inv
    return L


def L_CC(C, cove_X):
    """
    Compute the transformation matrix L using matrix C and the inverse of covariance matrix X.

    This function calculates the matrix L that transforms one dataset to align with another in terms of covariance.
    It is useful in various data processing applications where transformations based on covariance matrices are required.

    Parameters:
        C (np.ndarray): A 2D numpy array representing the cross-covariance matrix between two datasets.
        cove_X (np.ndarray): A 2D numpy array representing the covariance matrix of one dataset.

    Returns:
        np.ndarray: A 2D numpy array representing the transformation matrix L.
    """
    # Calculate the transformation matrix L
    L = C @ inv(cove_X)
    return L


def difference_use_chonchrome(matrix_x, matrix_y, L):
    """
    Compute a matrix based on the differences between transformed matrix_x and matrix_y, using transformation matrix L.

    This function calculates the difference vectors for each corresponding element of matrix_x and matrix_y, where matrix_x is
    transformed by matrix L. It computes the outer product of these difference vectors and averages the results to produce a matrix
    that quantifies the discrepancies between the transformed matrix_x and matrix_y.

    Parameters:
        matrix_x (np.ndarray): A 3D numpy array where each (i, j) position contains a vector of features.
        matrix_y (np.ndarray): A 3D numpy array where each (i, j) position contains a vector of features, corresponding to matrix_x.
        L (np.ndarray): A 2D numpy array representing the transformation matrix used to transform elements of matrix_x.

    Returns:
        np.ndarray: A 2D numpy array representing the averaged outer product matrix of the difference vectors.
    """
    # Initialize the matrix to store the sum of outer products of the difference vectors
    outer_product_matrix = np.zeros((matrix_x.shape[2], matrix_x.shape[2]))

    # Loop through each element (vector) in the matrices to compute the difference vectors and their outer products
    for i in range(matrix_x.shape[0]):
        for j in range(matrix_x.shape[1]):
            vec_x = matrix_x[i][j]
            vec_y = matrix_y[i][j]
            vec_e = vec_y - (L @ vec_x)  # Compute the difference vector
            outer_product_matrix += np.outer(vec_e, vec_e)  # Accumulate the outer product of the difference vector

    # Normalize the outer product matrix by the total number of elements to get the final matrix
    return outer_product_matrix / (matrix_x.shape[0] * matrix_x.shape[1])

def diffrence_use_equalization(matrix_x,matrix_y,L):
    """
    Compute a matrix based on the differences between transformed matrix_x and matrix_y, using transformation matrix L.

    This function calculates the difference vectors for each corresponding element of matrix_x and matrix_y, where matrix_x is
    transformed by matrix L. It then computes the outer product of these difference vectors and averages the results to produce a matrix
    that quantifies the discrepancies between the transformed matrix_x and matrix_y.

    Parameters:
        matrix_x (np.ndarray): A 3D numpy array where each (i, j) position contains a vector of features.
        matrix_y (np.ndarray): A 3D numpy array where each (i, j) position contains a vector of features, corresponding to matrix_x.
        L (np.ndarray): A 2D numpy array representing the transformation matrix used to transform elements of matrix_x.

    Returns:
        np.ndarray: A 2D numpy array representing the averaged outer product matrix of the difference vectors.
    """
    outer_product_matrix =np.zeros((matrix_x.shape[2], matrix_x.shape[2]))
    for i in range(matrix_x.shape[0]):
        for j in range(matrix_x.shape[1]):
            vec_x = matrix_x[i][j]
            vec_y = matrix_y[i][j]
            vec_e = vec_y - (L @ vec_x)

            outer_product_matrix = outer_product_matrix +  np.outer(vec_e, vec_e)

    return (outer_product_matrix/(matrix_x.shape[0]*matrix_x.shape[1]))




def add_target(matix, vec):
    """
    Add a target vector to each pixel in a 3D matrix.

    Parameters:
    matrix (numpy.ndarray): A 3D numpy array of shape (height, width, spectral bands).
    vec (numpy.ndarray): A 1D numpy array representing the target vector with length equal to the number of spectral bands.

    Returns:
    numpy.ndarray: A 3D numpy array with the same shape as `matrix`, with the target vector added to each pixel.

    Description:
    This function adds a scaled version of a target vector to each pixel in the input 3D matrix.
    The scaling factor used is 0.01. The process involves the following steps:

    1. Initialize a new matrix with the same shape as the input matrix.
    2. Scale the target vector by multiplying it by 0.01.
    3. Iterate over each pixel in the input matrix and add the scaled target vector to the pixel's values.
    4. Store the result in the new matrix and return it.

    Example:
    If the input `matrix` has dimensions (100, 100, 10) and `vec` has length 10, the function adds 0.01 times
    `vec` to each pixel vector in `matrix`, resulting in a new matrix with the same dimensions.
    """

    new_matrix = np.zeros((matix.shape[0], matix.shape[1], matix.shape[2]))
    target = vec * 0.01
    for i in range(matix.shape[0]):
        for j in range(matix.shape[1]):
            new_matrix[i][j] = matix[i][j] + target

    return new_matrix

""""
def Anomaly(matrix_x,matrix_y,matrix_x_target,L):
    anomaly =np.zeros((matrix_x.shape[0], matrix_x.shape[1]))
    anomaly_target = np.zeros((matrix_x.shape[0], matrix_x.shape[1]))
    E = difference_use_chonchrome(matrix_x, matrix_y, L)
    E = np.linalg.inv(E)
    print("finsh calc E and L")
    for i in range(matrix_x.shape[0]):
        for j in range(matrix_x.shape[1]):
            vec_x = matrix_x[i][j]
            vec_x_targit = matrix_x_target[i][j]
            vec_y = matrix_y[i][j]
            vec_e = vec_y - (L_chrom @ vec_x)
            vec_e_target = vec_y - (L_chrom @ vec_x_targit)
            anomaly[i][j] = vec_e.T@E@vec_e

            anomaly_target[i][j] = vec_e_target.T @ E @ vec_e_target
    return anomaly,anomaly_target
"""
def Anomaly_y(matrix_x,matrix_y,matrix_y_target,L):
    """
    Detect anomalies in the given matrices using the Chronochrome or equlazition methods.

    Parameters:
    matrix_x (numpy.ndarray): A 3D numpy array representing the reference image (height, width, spectral bands).
    matrix_y (numpy.ndarray): A 3D numpy array representing the test image (height, width, spectral bands).
    matrix_y_target (numpy.ndarray): A 3D numpy array representing the target image (height, width, spectral bands).
    L (numpy.ndarray): A transformation matrix used in the Chronochrome or equlazition methods.

    Returns:
    tuple: Two 2D numpy arrays of shape (height, width). The first array contains the anomaly scores for `matrix_y`,
    and the second array contains the anomaly scores for `matrix_y_target`.

    Description:
    This function performs anomaly detection on the given matrices using the Chronochrome method. It involves the following steps:

    1. Initialize the anomaly matrices with zeros.
    2. Compute the difference matrix `E` using the Chronochrome or equlazition methods and calculate its inverse.
    3. For each pixel, compute the residual vectors `vec_e` and `vec_e_target`.
    4. Calculate the anomaly scores for each pixel in `matrix_y` and `matrix_y_target`.

    The result is two 2D arrays containing the anomaly scores for the input matrices.
    """
    anomaly =np.zeros((matrix_x.shape[0], matrix_x.shape[1]))
    anomaly_target = np.zeros((matrix_x.shape[0], matrix_x.shape[1]))
    E = difference_use_chonchrome(matrix_x, matrix_y, L)
    E = np.linalg.inv(E)
    print("finsh calc E and L")
    for i in range(matrix_x.shape[0]):
        for j in range(matrix_x.shape[1]):
            vec_x = matrix_x[i][j]
            vec_y_target = matrix_y_target[i][j]
            vec_y = matrix_y[i][j]
            vec_e = vec_y - (L_chrom @ vec_x)
            vec_e_target = vec_y_target - (L_chrom @ vec_x)
            anomaly[i][j] = vec_e.T@E@vec_e

            anomaly_target[i][j] = vec_e_target.T @ E @ vec_e_target
    return anomaly,anomaly_target
"""""
def Anomaly_y_inovation(matrix_x,matrix_y,matrix_y_target,L,target):
    anomaly =np.zeros((matrix_x.shape[0], matrix_x.shape[1]))
    anomaly_target = np.zeros((matrix_x.shape[0], matrix_x.shape[1]))
    E = difference_use_chonchrome(matrix_x, matrix_y, L)
    E = np.linalg.inv(E)
    print("finsh calc E and L")
    for i in range(matrix_x.shape[0]):
        for j in range(matrix_x.shape[1]):
            vec_x = matrix_x[i][j]
            vec_y_target = matrix_y_target[i][j]
            vec_y = matrix_y[i][j]
            vec_e = vec_y - (L_chrom @ vec_x)
            vec_e_target = vec_y_target - (L_chrom @ vec_x)
            anomaly[i][j] = target.T@E@vec_e

            anomaly_target[i][j] = target.T @ E @ vec_e_target
    return anomaly,anomaly_target
"""""




def mathod_chronochrome_use_0_for_normolize_y(matrix_x,matrix_y,matrix_y_target,L):
    """
    Perform anomaly detection using the Chronochrome method and plot the results.

    Parameters:
    matrix_x (numpy.ndarray): A 3D numpy array representing the reference image (height, width, spectral bands).
    matrix_y (numpy.ndarray): A 3D numpy array representing the test image (height, width, spectral bands).
    matrix_y_target (numpy.ndarray): A 3D numpy array representing the target image (height, width, spectral bands).
    L (numpy.ndarray): A transformation matrix used in the Chronochrome method.

    Returns:
    None

    Description:
    This function performs anomaly detection on the given matrices using the Chronochrome method with normalization.
    The process involves the following steps:

    1. Compute the anomaly matrices for `matrix_y` and `matrix_y_target` using the `Anomaly_y` function.
    2. Flatten the anomaly matrices and compute histograms for both anomaly matrices.
    3. Plot the histograms of the anomaly matrices.
    4. Generate values for the matched filter (MF) and calculate the probability of detection (Pd) and probability of false alarm (Pfa).
    5. Plot the inverse cumulative probability distribution and the ROC curve.
    6. Calculate and print the area under the ROC curve for specified thresholds (0.1, 0.01, 0.001).

    This function provides a comprehensive analysis of the anomaly detection performance using visual plots and numerical metrics.
    """
    global mathod_chronochrome_use_0_for_normolize_y_Pfa
    global mathod_chronochrome_use_0_for_normolize_y_Pd

    print("mathod_chronochrome_use_0_for_normolize_y")

    no_targ, whit_targ = Anomaly_y(matrix_x, matrix_y, matrix_y_target,L)
    print("finsh anomly")

    no_targ = no_targ.flatten()
    whit_targ = whit_targ.flatten()
    nt_hist_values, nt_hist_edges = np.histogram(no_targ, bins=150, range=(0, 700))
    wt_hist_values, wt_hist_edges = np.histogram(whit_targ, bins=150, range=(0, 700))
    # Get the center of each bin
    nt_hist_centers = 0.5 * (nt_hist_edges[1:] + nt_hist_edges[:-1])
    wt_hist_centers = 0.5 * (wt_hist_edges[1:] + wt_hist_edges[:-1])
    # Create the plots
    plt.figure(figsize=(10, 6))
    plt.plot(nt_hist_centers, nt_hist_values, label='NT', linestyle='-', linewidth=2)
    plt.plot(wt_hist_centers, wt_hist_values, label='WT', linestyle='-', linewidth=2)
    plt.legend()
    plt.title('Generated Histograms,CC')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show(block = False)

    # Generate MF_val values
    MF_val = np.linspace(0, 500, 10000)
    Pd = []
    Pfa = []
    division  = (matrix_x.shape[0] * matrix_x.shape[1])
    # Calculate Pd and Pfa for each MF_val
    for val in MF_val:
        Pd.append(np.sum(whit_targ > val) /division )  # True Positive Rate
        Pfa.append(np.sum(no_targ > val) / division)

    # Plot the Inverse Cumulative Probability Distribution
    plt.figure()
    plt.plot(MF_val, Pfa, 'r')
    plt.title("Inverse Cumulative Probability Distribution")
    plt.legend(['Pfa'])
    plt.grid(True)
    #plt.hold(True)
    plt.plot(MF_val, Pd, 'b')
    plt.legend(['Pfa', 'Pd'])
    plt.grid(True)
    mathod_chronochrome_use_0_for_normolize_y_Pfa =Pfa

    mathod_chronochrome_use_0_for_normolize_y_Pd=Pd
    # Plot the ROC curve
    plt.figure()
    plt.plot(Pfa, Pd)
    plt.title("ROC Curve")
    plt.xlabel('Pfa (False Positive Rate)')
    plt.ylabel('Pd (True Positive Rate)')
    plt.xlim([0, 1])  # Limit x-axis from 0 to 0.1
    plt.grid(True)
    plt.show()

    # Ensure Pfa and Pd are numpy arrays, not lists


    Pfa = np.array(Pfa)
    Pd = np.array(Pd)

    A = np.zeros(3)
    sums = np.zeros(3)
    th = np.array([0.1, 0.01, 0.001])

    for i in range(3):
        # Indices where Pfa is less than or equal to the threshold
        indices = np.where(Pfa <= th[i])

        # Use the indices to index into Pfa and Pd, ensure they are numpy arrays
        # If Pfa and Pd are lists, this will cause the TypeError mentioned
        Pfa_subset = Pfa[indices]
        Pd_subset = Pd[indices]

        # Calculate the area using trapezoidal integration
        sums[i] = -np.trapz(Pd_subset, Pfa_subset)

         # Calculate A using the trapezoidal sum and the threshold
        A[i] = (sums[i] - 0.5 * th[i] ** 2) / (th[i] - 0.5 * th[i] ** 2)

        # Print the result
    print(A)


def mathod_chronochrome_use_m_for_normolize_y_inovation(matrix_x,matrix_y,matrix_y_target,L):
    """
    Perform anomaly detection using the Chronochrome method with innovation normalization and plot the results.

    Parameters:
    matrix_x (numpy.ndarray): A 3D numpy array representing the reference image (height, width, spectral bands).
    matrix_y (numpy.ndarray): A 3D numpy array representing the test image (height, width, spectral bands).
    matrix_y_target (numpy.ndarray): A 3D numpy array representing the target image (height, width, spectral bands).
    L (numpy.ndarray): A transformation matrix used in the Chronochrome method.

    Returns:
    None

    Description:
    This function performs anomaly detection on the given matrices using the Chronochrome method with innovation normalization.
    The process involves the following steps:

    1. Compute the anomaly matrices for `matrix_y` and `matrix_y_target` using the `Anomaly_y` function.
    2. Flatten the anomaly matrices and compute histograms for both anomaly matrices.
    3. Plot the histograms of the anomaly matrices.
    4. Generate values for the matched filter (MF) and calculate the probability of detection (Pd) and probability of false alarm (Pfa).
    5. Plot the inverse cumulative probability distribution and the ROC curve.
    6. Calculate and print the area under the ROC curve for specified thresholds (0.1, 0.01, 0.001).

    This function provides a comprehensive analysis of the anomaly detection performance using visual plots and numerical metrics.
    """
    global mathod_chronochrome_use_m_for_normolize_y_inovation_Pd
    global mathod_chronochrome_use_m_for_normolize_y_inovation_pfa
    print("mathod_chronochrome_use_m_for_normolize_y_inovation")


    no_targ, whit_targ = Anomaly_y(matrix_x, matrix_y, matrix_y_target,L)
    # Flatten the matrices
    print("finsh anomly")

    no_targ = no_targ.flatten()
    whit_targ = whit_targ.flatten()
    nt_hist_values, nt_hist_edges = np.histogram(no_targ, bins=150, range=(0, 1000))
    wt_hist_values, wt_hist_edges = np.histogram(whit_targ, bins=150, range=(0, 1000))
    # Get the center of each bin
    nt_hist_centers = 0.5 * (nt_hist_edges[1:] + nt_hist_edges[:-1])
    wt_hist_centers = 0.5 * (wt_hist_edges[1:] + wt_hist_edges[:-1])
    # Create the plots
    plt.figure(figsize=(10, 6))
    plt.plot(nt_hist_centers, nt_hist_values, label='NT', linestyle='-', linewidth=2)
    plt.plot(wt_hist_centers, wt_hist_values, label='WT', linestyle='-', linewidth=2)
    plt.legend()
    plt.title('Generated Histograms, CC inovation')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show(block = False)

    # Generate MF_val values
    MF_val = np.linspace(0, 1000, 10000)
    Pd = []
    Pfa = []
    division  = (matrix_x.shape[0] * matrix_x.shape[1])
    # Calculate Pd and Pfa for each MF_val
    for val in MF_val:
        Pd.append(np.sum(whit_targ > val) /division )  # True Positive Rate
        Pfa.append(np.sum(no_targ > val) / division)

    # Plot the Inverse Cumulative Probability Distribution
    plt.figure()
    plt.plot(MF_val, Pfa, 'r')
    plt.title("Inverse Cumulative Probability Distribution")
    plt.legend(['Pfa'])
    plt.grid(True)
    #plt.hold(True)
    plt.plot(MF_val, Pd, 'b')
    plt.legend(['Pfa', 'Pd'])
    plt.grid(True)

    # Plot the ROC curve
    mathod_chronochrome_use_m_for_normolize_y_inovation_Pd = Pd
    mathod_chronochrome_use_m_for_normolize_y_inovation_pfa = Pfa
    plt.figure()
    plt.plot(Pfa, Pd)
    plt.title("ROC Curve")
    plt.xlabel('Pfa (False Positive Rate)')
    plt.ylabel('Pd (True Positive Rate)')
    plt.xlim([0, 1])  # Limit x-axis from 0 to 0.1
    plt.grid(True)
    plt.show()

    # Ensure Pfa and Pd are numpy arrays, not lists


    Pfa = np.array(Pfa)
    Pd = np.array(Pd)

    A = np.zeros(3)
    sums = np.zeros(3)
    th = np.array([0.1, 0.01, 0.001])

    for i in range(3):
        # Indices where Pfa is less than or equal to the threshold
        indices = np.where(Pfa <= th[i])

        # Use the indices to index into Pfa and Pd, ensure they are numpy arrays
        # If Pfa and Pd are lists, this will cause the TypeError mentioned
        Pfa_subset = Pfa[indices]
        Pd_subset = Pd[indices]

        # Calculate the area using trapezoidal integration
        sums[i] = -np.trapz(Pd_subset, Pfa_subset)
        print(sums[i])

         # Calculate A using the trapezoidal sum and the threshold
        A[i] = (sums[i] - 0.5 * th[i] ** 2) / (th[i] - 0.5 * th[i] ** 2)

        # Print the result
    print(A)




def mathod_equalization_use_0_for_normolize_y(matrix_x,matrix_y,matrix_y_target,L):
    """
        Perform anomaly detection using the Covariance Equalization method and plot the results.

        Parameters:
        matrix_x (numpy.ndarray): A 3D numpy array representing the reference image (height, width, spectral bands).
        matrix_y (numpy.ndarray): A 3D numpy array representing the test image (height, width, spectral bands).
        matrix_y_target (numpy.ndarray): A 3D numpy array representing the target image (height, width, spectral bands).
        L (numpy.ndarray): A transformation matrix used in the Covariance Equalization method.

        Returns:
        None

        Description:
        This function performs anomaly detection on the given matrices using the Covariance Equalization method with normalization.
        The process involves the following steps:

        1. Compute the anomaly matrices for `matrix_y` and `matrix_y_target` using the `Anomaly_y` function.
        2. Flatten the anomaly matrices and compute histograms for both anomaly matrices.
        3. Plot the histograms of the anomaly matrices.
        4. Generate values for the matched filter (MF) and calculate the probability of detection (Pd) and probability of false alarm (Pfa).
        5. Plot the inverse cumulative probability distribution and the ROC curve.
        6. Calculate and print the area under the ROC curve for specified thresholds (0.1, 0.01, 0.001).

        This function provides a comprehensive analysis of the anomaly detection performance using visual plots and numerical metrics.
        """
    global mathod_equalization_use_0_for_normolize_y_Pfa
    global mathod_equalization_use_0_for_normolize_y_Pd
    print("mathod_equalization_use_0_for_normolize_y")

    #for i in range (norm_matrix_taget.shape[0]):
    #    for j in range (norm_matrix_taget.shape[1]):
    #        print((norm_matrix_taget[i][j]-norm_matrix_X[i][j]))
    no_targ, whit_targ = Anomaly_y(matrix_x, matrix_y, matrix_y_target,L)
    # Flatten the matrices
    print("finsh anomly")

    no_targ = no_targ.flatten()
    whit_targ = whit_targ.flatten()
    nt_hist_values, nt_hist_edges = np.histogram(no_targ, bins=150, range=(0, 700))
    wt_hist_values, wt_hist_edges = np.histogram(whit_targ, bins=150, range=(0, 700))
    # Get the center of each bin
    nt_hist_centers = 0.5 * (nt_hist_edges[1:] + nt_hist_edges[:-1])
    wt_hist_centers = 0.5 * (wt_hist_edges[1:] + wt_hist_edges[:-1])
    # Create the plots
    plt.figure(figsize=(10, 6))
    plt.plot(nt_hist_centers, nt_hist_values, label='NT', linestyle='-', linewidth=2)
    plt.plot(wt_hist_centers, wt_hist_values, label='WT', linestyle='-', linewidth=2)
    plt.legend()
    plt.title('Generated Histograms, CE')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show(block = False)

    # Generate MF_val values
    MF_val = np.linspace(0, 500, 10000)
    Pd = []
    Pfa = []
    division  = (matrix_x.shape[0] * matrix_x.shape[1])
    # Calculate Pd and Pfa for each MF_val
    for val in MF_val:
        Pd.append(np.sum(whit_targ > val) /division )  # True Positive Rate
        Pfa.append(np.sum(no_targ > val) / division)

    # Plot the Inverse Cumulative Probability Distribution
    plt.figure()
    plt.plot(MF_val, Pfa, 'r')
    plt.title("Inverse Cumulative Probability Distribution")
    plt.legend(['Pfa'])
    plt.grid(True)
    #plt.hold(True)
    plt.plot(MF_val, Pd, 'b')
    plt.legend(['Pfa', 'Pd'])
    plt.grid(True)

    mathod_equalization_use_0_for_normolize_y_Pfa = Pfa
    mathod_equalization_use_0_for_normolize_y_Pd = Pd
    # Plot the ROC curve
    plt.figure()
    plt.plot(Pfa, Pd)
    plt.title("ROC Curve")
    plt.xlabel('Pfa (False Positive Rate)')
    plt.ylabel('Pd (True Positive Rate)')
    plt.xlim([0, 1])  # Limit x-axis from 0 to 0.1
    plt.grid(True)
    plt.show()

    # Ensure Pfa and Pd are numpy arrays, not lists


    Pfa = np.array(Pfa)
    Pd = np.array(Pd)

    A = np.zeros(3)
    sums = np.zeros(3)
    th = np.array([0.1, 0.01, 0.001])

    for i in range(3):
        # Indices where Pfa is less than or equal to the threshold
        indices = np.where(Pfa <= th[i])

        # Use the indices to index into Pfa and Pd, ensure they are numpy arrays
        # If Pfa and Pd are lists, this will cause the TypeError mentioned
        Pfa_subset = Pfa[indices]
        Pd_subset = Pd[indices]

        # Calculate the area using trapezoidal integration
        sums[i] = -np.trapz(Pd_subset, Pfa_subset)

         # Calculate A using the trapezoidal sum and the threshold
        A[i] = (sums[i] - 0.5 * th[i] ** 2) / (th[i] - 0.5 * th[i] ** 2)

        # Print the result
    print(A)


def mathod_equalization_use_m_for_normolize_y_inovation(matrix_x,matrix_y,matrix_y_target,L):
    global mathod_equalization_use_m_for_normolize_y_inovation_Pd
    global mathod_equalization_use_m_for_normolize_y_inovation_Pfa
    print("mathod_equalization_use_m_for_normolize_y_inovation")


    no_targ, whit_targ = Anomaly_y(matrix_x, matrix_y, matrix_y_target,L)
    # Flatten the matrices
    print("finsh anomly")

    no_targ = no_targ.flatten()
    whit_targ = whit_targ.flatten()
    nt_hist_values, nt_hist_edges = np.histogram(no_targ, bins=150, range=(0, 1000))
    wt_hist_values, wt_hist_edges = np.histogram(whit_targ, bins=150, range=(0, 1000))
    # Get the center of each bin
    nt_hist_centers = 0.5 * (nt_hist_edges[1:] + nt_hist_edges[:-1])
    wt_hist_centers = 0.5 * (wt_hist_edges[1:] + wt_hist_edges[:-1])
    # Create the plots
    plt.figure(figsize=(10, 6))
    plt.plot(nt_hist_centers, nt_hist_values, label='NT', linestyle='-', linewidth=2)
    plt.plot(wt_hist_centers, wt_hist_values, label='WT', linestyle='-', linewidth=2)
    plt.legend()
    plt.title('Generated Histograms, equlization inovation')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show(block = False)

    # Generate MF_val values
    MF_val = np.linspace(0, 1000, 10000)
    Pd = []
    Pfa = []
    division  = (matrix_x.shape[0] * matrix_x.shape[1])
    # Calculate Pd and Pfa for each MF_val
    for val in MF_val:
        Pd.append(np.sum(whit_targ > val) /division )  # True Positive Rate
        Pfa.append(np.sum(no_targ > val) / division)

    # Plot the Inverse Cumulative Probability Distribution
    plt.figure()
    plt.plot(MF_val, Pfa, 'r')
    plt.title("Inverse Cumulative Probability Distribution")
    plt.legend(['Pfa'])
    plt.grid(True)
    #plt.hold(True)
    plt.plot(MF_val, Pd, 'b')
    plt.legend(['Pfa', 'Pd'])
    plt.grid(True)

    # Plot the ROC curve
    mathod_equalization_use_m_for_normolize_y_inovation_Pfa = Pfa
    mathod_equalization_use_m_for_normolize_y_inovation_Pd = Pd
    plt.figure()
    plt.plot(Pfa, Pd)
    plt.title("ROC Curve")
    plt.xlabel('Pfa (False Positive Rate)')
    plt.ylabel('Pd (True Positive Rate)')
    plt.xlim([0, 1])  # Limit x-axis from 0 to 0.1
    plt.grid(True)
    plt.show()

    # Ensure Pfa and Pd are numpy arrays, not lists


    Pfa = np.array(Pfa)
    Pd = np.array(Pd)

    A = np.zeros(3)
    sums = np.zeros(3)
    th = np.array([0.1, 0.01, 0.001])

    for i in range(3):
        # Indices where Pfa is less than or equal to the threshold
        indices = np.where(Pfa <= th[i])

        # Use the indices to index into Pfa and Pd, ensure they are numpy arrays
        # If Pfa and Pd are lists, this will cause the TypeError mentioned
        Pfa_subset = Pfa[indices]
        Pd_subset = Pd[indices]

        # Calculate the area using trapezoidal integration
        sums[i] = -np.trapz(Pd_subset, Pfa_subset)
        print(sums[i])

         # Calculate A using the trapezoidal sum and the threshold
        A[i] = (sums[i] - 0.5 * th[i] ** 2) / (th[i] - 0.5 * th[i] ** 2)

        # Print the result
    print(A)





header_self = "self_test_rad.hdr"  # Change it to the path to the header, both cube and header should be
# in the same directory.
hdr_self = envi.read_envi_header(header_self)
data_self = open_image(header_self)  # hold extra data, not just the image itself
data_self = envi.open(header_self)
cube_self = data_self.load().copy()  # load the cube to a variable.
scipy.io.savemat('D1_F12_H1_Cropped_des.mat', {'D1_F12_H1_Cropped_des': cube_self})

header_blind = "blind_test_refl.hdr"  # Change it to the path to the header, both cube and header should be
# in the same directory.
hdr_blind = envi.read_envi_header(header_blind)
data_blind = open_image(header_blind)  # hold extra data, not just the image itself
data_blind = envi.open(header_blind)
cube_blind = data_blind.load().copy()  # load the cube to a variable.
scipy.io.savemat('D1_F12_H1_Cropped_des.mat', {'D1_F12_H1_Cropped_des': cube_blind})
print(data_blind.shape)
""" create X Y C """
""" Choronochrome we need 
-variance x 
-covreance xy
-L = coverance(xy) @ inv(variance(x) 
-e = normliaze(y) - L@normolize(x) 
-A(x,y) = e.T@<inv(e@e.T))@e = e.T@inv(E)@e
"""
"Choronochrome using M "
norm_matrix_X_M = normalize_noam(cube_self)
norm_matrix_Y_M = normalize_noam(cube_blind)
norm_matrix_x_0 = normalize_bodner(cube_self)
norm_matrix_y_0 = normalize_bodner(cube_blind)
var_x_0 = covariance(norm_matrix_x_0)
var_y_0 = covariance(norm_matrix_y_0)
var_x = covariance(norm_matrix_X_M)
var_y = covariance(norm_matrix_Y_M)
C = covariance_x_y(norm_matrix_X_M, norm_matrix_Y_M)
C_0 = covariance_x_y(norm_matrix_x_0,norm_matrix_y_0)
L_chrom = L_CC(C, var_x)
L_chrom_0 = L_CC(C_0,var_x_0)

norm_matrix_taget_M  = add_target(norm_matrix_X_M, cube_self[5][5])
norm_matrix_taget_0 = add_target(norm_matrix_x_0,cube_self[5][5])
norm_matrix_taget_M_y=add_target(norm_matrix_Y_M,cube_self[5][5])
"Choronochrome using avarge 0   when y is the target"
norm_matrix_taget_0_y = add_target(norm_matrix_y_0,cube_self[5][5])
mathod_chronochrome_use_0_for_normolize_y(norm_matrix_x_0, norm_matrix_y_0, norm_matrix_taget_0_y,L_chrom_0)

"Choronochrome using M  when y is the target Inovation"
norm_matrix_taget_M_y = add_target(norm_matrix_Y_M,cube_self[5][5])
mathod_chronochrome_use_m_for_normolize_y_inovation(norm_matrix_X_M, norm_matrix_Y_M, norm_matrix_taget_M_y,L_chrom)




"covariance equlazition using 0 "
L_CE_0 = L_covariance_Equalization(var_x_0,var_y_0)
mathod_equalization_use_0_for_normolize_y(norm_matrix_x_0, norm_matrix_y_0, norm_matrix_taget_0_y,L_CE_0)
"covariance equlazition using m inovation"
L_CE = L_covariance_Equalization(var_x, var_y)
mathod_equalization_use_m_for_normolize_y_inovation(norm_matrix_X_M, norm_matrix_Y_M, norm_matrix_taget_M_y,L_CE)




# Plot the ROC curve
plt.figure()
plt.plot(mathod_chronochrome_use_0_for_normolize_y_Pfa, mathod_chronochrome_use_0_for_normolize_y_Pd,label='Chronochrome')
plt.plot(mathod_equalization_use_0_for_normolize_y_Pfa,mathod_equalization_use_0_for_normolize_y_Pd,label='Equalization')
plt.title("Combined ROC Curve")
plt.xlabel('False Positive Rate (Pfa)')
plt.ylabel('True Positive Rate (Pd)')
plt.xlim([0, 1])  # Limit x-axis from 0 to 0.1
plt.legend()
plt.grid(True)
plt.show()

# Plot the ROC curve
plt.figure()
plt.plot(mathod_chronochrome_use_m_for_normolize_y_inovation_pfa, mathod_chronochrome_use_m_for_normolize_y_inovation_Pd,label='Chronochrome')
plt.plot(mathod_equalization_use_m_for_normolize_y_inovation_Pfa,mathod_equalization_use_m_for_normolize_y_inovation_Pd,label='Equalization')
plt.title("Combined ROC Curve")
plt.xlabel('False Positive Rate (Pfa)')
plt.ylabel('True Positive Rate (Pd)')
plt.xlim([0, 1])  # Limit x-axis from 0 to 0.1
plt.legend()
plt.grid(True)
plt.show()

# Plot the ROC curve
plt.figure()
plt.plot(mathod_chronochrome_use_0_for_normolize_y_Pfa, mathod_chronochrome_use_0_for_normolize_y_Pd,label='Chronochrome')
plt.plot(mathod_equalization_use_0_for_normolize_y_Pfa,mathod_equalization_use_0_for_normolize_y_Pd,label='Equalization')
plt.plot(mathod_chronochrome_use_m_for_normolize_y_inovation_pfa, mathod_chronochrome_use_m_for_normolize_y_inovation_Pd,label='Chronochrome_inovation')
plt.plot(mathod_equalization_use_m_for_normolize_y_inovation_Pfa,mathod_equalization_use_m_for_normolize_y_inovation_Pd,label='Equalization_inovation')
plt.title("Combined ROC Curve")
plt.xlabel('False Positive Rate (Pfa)')
plt.ylabel('True Positive Rate (Pd)')
plt.xlim([0, 1])  # Limit x-axis from 0 to 0.1
plt.legend()
plt.grid(True)
plt.show()








