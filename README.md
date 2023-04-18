# Yalefaces-analysis
Correlation, Eigenvector, and SVD analysis of the yale faces dataset for machine learning

##### Author: 
Riley Estes

### Abstract
This project uses the dataset of pictures of peoples faces with different lighting know as Yale Faces and creates correlation matrices, performs eigenvector analysis, and applies single value decomposition to the images. This is done in order to find the feature space and important aspects of each image in identifying the faces and matching them to people. The image processing in this lab can help optimize a machine learning algorithm and will yield a better outcome error and be more efficient than simply giving the algorithm the raw data alone would. 

### Introduction and Overview
The Yale Faces dataset is a set of 2414 images of different people's faces in different lighting conditions often used to train machine learning models on facial recognition. The particular set used here is compressed to 32x32 image sizes, made greyscale, and normalized to have pixel values between 0 and 1. Letting a machine learning algorithm train on raw data such as this would cost much more and take much longer than it needs to. So, instead of taking this approach, an alternative is offered in this project. By exploring the utility of correlation matrices, eigenvector analysis, and SVD analysis, the facial recognition problem can be much more efficiently and accurately implemented. The variance between images can be used to classify images together, and feature spaces can be found with eigenvector and SVD analysis in order to perform Principle Component Analysis or dimension reduction to make the program faster and cheaper, or to select certain feature spaces that are more important in facial recognition, making the program more accurate. 

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
##### Correlation:
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

In order to find the most and least correlated images, the diagonal of the matrix (which is always 1) is removed from consideration so that the first and second image in a pair can be taken from max and min using max/100 and max%100 with the following code:
```
cNoDiag = np.copy(c)
np.fill_diagonal(cNoDiag, 0.5)
max = np.argmax(cNoDiag)
min = np.argmin(cNoDiag)
```

Using a select 10 images, a new 10x10 correlation matrix is made with the following code and image indices:
```
indices = np.array([0, 312, 511, 4, 2399, 112, 1023, 86, 313, 2004])
X10 = X[:, indices]
c10 = np.dot(X10.T, X10) / 1024
```
Note that the image index at 0 is labeled as image 1 in the dataset, so for example, index 2004 holds image number 2005.


##### Eigenvectors:
The following code creates a 1024x6 array of the 6 most important eigenvectors based on the magnitude of their eigenvalues:
```
Y = X.dot(X.T)
eigvals, eigvecs = np.linalg.eig(Y)
greatest = np.flip(np.argsort(np.abs(eigvals)))[:6]
greatestEVecs = eigvecs[:, greatest]
```
argsort sorts values from least to greatest, so the array is flipped and the first 6 values are taken as the indices of the 6 greatest eigenvalues. The corresponding eigenvectors are at the same indices in the eigenvector array taken from np.linalg.eig(Y).

##### Single Value Decomposition:
The SVD of the faces array (still normalized from the first block in the correlation section) is taken and the six greatest principle components are extracted with the following code:
```
u, s, vt = np.linalg.svd(X)
PCDirs = vt[:6, :]
```
The matrices are already sorted in order of greatest variance to least, so the first 6 rows in vt correspond to the 6 greatest varianced principle component directions. 

To see the difference between the SVD analysis and the eigenvector analysis, the normalized difference (linear error) is calculated between the first eigenvector and first SVD mode:
```
diff_norm = np.linalg.norm(np.abs(eigvecs[:, 0]) - np.abs(u[:, 0]))
```

Finally, the last analysis finds the percent of the total variance each of the 6 greatest SVD modes correpsonds to:
```
totalVar = np.sum(s**2)
varPercents = (s**2 / totalVar)[:6] * 100
```
This uses the fact that s (Sigma) is a diagonal matrix with the standard deviations of each mode in order from greatest standard deviation to least. Therefore, the variance of each vector alone can be compared to the total variance in the whole matrix in order to find the percentage of the matrix variance that corresponds to each mode. 

### Computational Results

##### Correlation:
The displayed correlation matrix for the first 100 images is shown here: 
<br> <img src="https://github.com/rileywe/Yalefaces-analysis/blob/main/Output%20Images/big%20corr.png?raw=true" width="400"/>

Notice that all values along the diagonal are 1 due to normalization. The matrix is also symmetric along the diagonal, which makes sense as a compared to be is the same as b compared to a. There is a wide range of variances captured in this matrix, but most values are near 0 or 0.5. However, a few image pairs exhibit near 1 correlation, and some exhibit nearly -0.6 correlation. 
After removing the diagonal from consideration, the maximum correlation is between images  6  and  63  with a value of  0.9710984631450812, while the minimum correlation is between images  16  and  82  with a value of  -0.7840280171025154. 

The most correlated image pair is shown here:
<br> <img src="https://github.com/rileywe/Yalefaces-analysis/blob/main/Output%20Images/most%20corr.png?raw=true" width="400"/>

And the least correlated pair is here:
<br> <img src="https://github.com/rileywe/Yalefaces-analysis/blob/main/Output%20Images/least%20corr.png?raw=true" width="400"/>

These results make sense, seeing as the most correlated pair is the same face and only very slightly different lighting. The images in the least correlated are complete opposites being a male face and a female face with opposite lighting. 

The correlation matrix for the select 10 faces is shown here:
<br> <img src="https://github.com/rileywe/Yalefaces-analysis/blob/main/Output%20Images/small%20corr.png?raw=true" width="400"/>

Much like the first correlation matrix, this one shows a zoomed in perspective. It's much eaiser now to see which images are well or not well correlated with each other. For example, images at indices 4 and 6 have a high correlation, and so are likely very similar and probably the same person's face, while images at indices 3 and 6 are inversely correlated, which likely means it's the same face but with opposite lighting. Image pairs such as the one at indices 5 and 8 have nearly 0 correlation, and are likely 2 different faces with different lighting. 

##### Eigenvectors and SVD:
The normalized difference of the absolute values of the first eigenvector in Y (the correlation matrix) and the first mode from the SVD of X is 8.806729017063308e-16.
Seeing as this value is very close to 0 (being 10^-16), it is reasonable to assume that the eigenvector and SVD analysis strategies essentially perform the same task and yield very similar if not identical outputs. This theoretically makes sense as both approaches aim to find which aspects across all of the images in the data correspond to the highest variance among the images. 
To look into this relationship further, the variance of the first 6 (and most significant) SVD modes were calculated as percents of the total variance:
SVD mode 0 is 39.159% of the total variance.
SVD mode 1 is 18.854% of the total variance.
SVD mode 2 is 7.704% of the total variance.
SVD mode 3 is 4.889% of the total variance.
SVD mode 4 is 2.342% of the total variance.
SVD mode 5 is 2.035% of the total variance.
This shows that the first few modes hold most of the variance and therefore most of what is important to consider when classifying data. This means that the data could be put through Principle Component Analysis (PCA) to look at only the first 6 modes, and still have almost as good of an idea of how to classify the data as it would have if it considered all of the principle component directions, except that it would take a very small fraction of the computing power and time. 
To visually represent which feature spaces these modes represent, they have been displayed here:
<br> <img src="https://github.com/rileywe/Yalefaces-analysis/blob/main/Output%20Images/eigfaces.png?raw=true" width="800"/>

As is shown, each mode corresponds to a body part (such as the nose or eyes) or to the lighting condition of the face. It is concerning to see that the first mode with almost 40% of the total variance is a feature space of the lighting on the face because this would mkae facial recognition in different light conditions very challenging, and the program would need to resort to other modes to get the correct facial recognition data it would need. 

### Summary and Conclusions
The Yale Faces data set of 2414 images of faces in different lighting conditions proves a challenge to classify with a machine learning algorithm. Image preprocessing and analysis such as correlation mapping, eigenvector and SVD analysis can work wonders in improving the error, runtime, and computing power neccesary to properly classify each face to its owner. The correlation approach gives insights to how similar an image is to another on a per-pixel basis simply by looking at the variance in pixel values between images. This is a method that might work for a classification algorithm, but a smarter approach by first finding the feature spaces using eigenvector analysis or SVD analysis would be more efficient by far. This approach finds the feature spaces the program is looking for in each image based on the variance of each feature space, and shows which parts of the image the program weighs most when it's classifying the data. By only considering the most important feature spaces, you can not only make the program more efficient, but by also removing unwanted feature spaces such as lighting differences, you could improve the classification error of the program as well. 
