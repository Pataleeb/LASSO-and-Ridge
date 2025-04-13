import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

data = scipy.io.loadmat('cs.mat')
print(data.keys())

if 'img' in data:
    x = data['img'].reshape(50, 50)
    plt.imshow(x, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
   # plt.show()

print(f"Number of 1s: {np.sum(x == 1)}")
print(f"Number of 0s: {np.sum(x == 0)}")

x =x.flatten()
np.random.seed(6740)
A=np.random.randn(1300,2500)
epsilon=np.random.normal(0,np.sqrt(25),size=1300)

y=A@x + epsilon
lasso_img= LassoCV(cv=10,alphas=np.logspace(-4,1,50),random_state=100).fit(A,y)

optimal_lambda=lasso_img.alpha_
print(f"Optimal lambda value: {optimal_lambda}")

img_recovered=lasso_img.coef_.reshape(50,50)

plt.figure(figsize=(8, 5))
plt.semilogx(lasso_img.alphas_, np.mean(lasso_img.mse_path_, axis=1), marker='o')
plt.axvline(optimal_lambda, linestyle='--', color='red', label=f"Optimal λ: {optimal_lambda:.4f}")
plt.xlabel("Lambda Value")
plt.ylabel("MSE")
plt.title("LASSO Cross-Validation Error")
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(x.reshape(50, 50), cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(img_recovered, cmap='gray')
axes[1].set_title("Recovered Image")
axes[1].axis("off")

plt.show()

###Q2
alphas = np.logspace(-4,1,50)
ridge_reg = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_reg.fit(A,y)

optimal_lambda_ridge = ridge_reg.alpha_
print(f"Optimal lambda for Ridge: {optimal_lambda_ridge}")

img_recovered_ridge = ridge_reg.coef_.reshape(50, 50)
error_ridge=np.mean(ridge_reg.cv_values_, axis=0)

plt.figure(figsize=(8, 5))
plt.semilogx(alphas, error_ridge, marker='o', label="CV Error")
plt.axvline(optimal_lambda_ridge, linestyle='--', color='red', label=f"Optimal λ: {optimal_lambda_ridge:.4f}")
plt.xlabel("Lambda")
plt.ylabel("MSE")
plt.title("Ridge Regression Cross-Validation Error Curve")
plt.legend()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(x.reshape(50, 50), cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(img_recovered_ridge, cmap='gray')
axes[1].set_title("Recovered Image (Ridge Regression)")
axes[1].axis("off")

plt.show()