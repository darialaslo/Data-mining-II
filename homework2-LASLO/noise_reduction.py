import imageio
import numpy.linalg as linalg
import numpy as np

'''
Function for performing the noise reduction through SVD.
This is a simple implementation where the reduction of
the noise is performed via the so-called nullification 
of singular values (and corresponding singular vectors 
after the multiplication) below a fixed threshold.

Input
----------------
img:	    noisy image
threshold:  singular values threshold

Output
----------------
img_dn:		denoised image
'''
def NoiseReduction(img, threshold):
	# 1. Perform SVD on the noisy image, i.e., img

	#obtain svd for the image 
	L, D, R = linalg.svd(img)

	# 2. Nullify the singular values lower than the 
	#    threshold, i.e., threshold
	D[np.where(D<threshold)]=0

	#define the matrix of the singular values 
	S = np.zeros((L.shape[1], R.shape[0]))
	S[:D.shape[0], :D.shape[0]] = np.diag(D)

	# 3. Reconstruct the image using the modified 
	#    singular values

	reconstr_img = np.dot(L, np.dot(S, R))
	

	return reconstr_img
