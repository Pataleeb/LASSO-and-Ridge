# LASSO-and-Ridge
Medical Image Reconstruction with LASSO and Ridge Regression

We begin witht the true image of dimesion 50 * 50. In this image 2084 of its pixels have a value of 0, while 416 pixels have a value of 1. 
We measure n=1300 linear combinations with the weights in the linear combination being ranom and independently distributed as N(0,1).
These measurements are given by the entries of the vector y=AX+e
where y=R(1300) and A=R(1300*2500). Error term is normally distributed as N(0,25*I(1300)). 

We can model y as a linear combination of the columns of x to recover some coefficient vector that is close to the true image.

Although the n=1300 is smaller than the dimesion p=2500, the true image is sparse. Therefore, we can recover the sparse image using few measurements exploiting its structure. This is called "compressed sensing".

This image recovery can be done using lasso and ridge regression techniques. 

First we use lasso to recover the image and select optimal lambda using 10-fold cross-validation (CV).

Next, we use ridge regression to recover the image with same CV.




