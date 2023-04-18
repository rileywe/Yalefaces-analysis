# Yalefaces-analysis
Correlation, Eigenvector, and SVD analysis of the yale faces dataset for machine learning

##### Author: 
Riley Estes

### Abstract
This project uses the dataset of pictures of peoples faces with different lighting know as Yale Faces and creates correlation matrices, performs eigenvector analysis, and applies single value decomposition to the images. This is done in order to find the feature space and important aspects of each image in identifying the faces and matching them to people. The image processing in this lab can help optimize a machine learning algorithm and will yield a better outcome error than simply giving the algorithm the raw data alone would. 

### Theoretical Background
There are three theoretical topics explored in this procedure: 
<ol>
  <li>Correlation</li>
  <li>Eigenvectors</li>
  <li>Single Value Decomposition (SVD)</li>
</ol>

1. Correlation between images is the measure of how similar on a pixel by pixel basis two images are to each other. By making a 2x2 matrix where each data point in the matrix is the variance between both of the images, and the axis are image indices, the correlations between each image and each other image can easily be visualized. This is useful because it can be used to classify images together and determine if they are images of the same thing or not. Note that it is important to first normalize the images to have mean 0 and variance 1 so that the variance of an image pair can be directly compared to the variance of another image pair. This can be confirmed by seeing if the correlation of an image with itself is 1. If the variance of each image is normalized to 1, the variance between an image and itself should also be 1.

2. Eigenvectors and corresponding eigenvalues may also be used in data processing to determine which data variables have the largest effect on the variance of each image, and thus which variables are most important to consider when classifying data. This can be done by finding which eigenvectors have the greatest eigenvalues, or which parts of the input image have the greatest effect on whether two images are similar. Eigenvector analysis is closely related to principle component analysis in that it can find which components of the data are most unique to each sample, and thus which components should be looked at most closely when classifying the data. 

3. Single Value Decomposition is a method by which a matrix (such as a data set) is decomposed into 3 composite matrices U, Sigma, and V^T that, when the dot product of them in that order is taken, will solve to be the original matrix. The matrix U contains the singular vectors that describe the variation in the rows of the input data set matrix, and the martix V contains the singular vectors corresponding to variance in the columns of the input matrix. Given a data set where each image is contained in a column, the matrix U describes the varaince across all images on a per-pixel basis, and the matrix V describes the variance between each pixel within an image. In order to find feature spaces with this method, the first 6 vectors in U corresponding the greatest percentages of the total dimensional variance can be taken to represent the most important aspects when classifying the image. The first 6 values in the diagonal Sigma matrix correspond to the standard deviations of each of these vectors. These vectors are effectively the same as the eigenvectors with the greatest eigenvalues. 

### Algorithm Implementation and Development
After the yale face image data (compressed to 32x32 pixel images and normalized to greyscale values between 0 and 1) is loaded as a 1024x2414 matrix X, it is normalized to have mean 0 and variance 1:
```
means = np.mean(X, axis=0)
stddevs = np.std(X, axis=0)
X = (X - means) / stddevs
```

Then to create a correlation matrix for the first 100 images, the dot product of the resulting matrix with a transpose of itself is taken and normalized to have values between -1 and 1:
```
X100 = X[:, :100]
c = np.dot(X100.T, X100) / 1024
```
The displayed correlation matrix for these first 100 images is shown here: 
<br> <img src="https://github.com/rileywe/Yalefaces-analysis/blob/main/Output%20Images/big%20corr.png?raw=true" width="400"/>


