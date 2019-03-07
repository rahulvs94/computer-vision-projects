# computer vision projects

PERSPECTIVE TRANSFORMATION 

Approach 1:
	
	- Implemented using findContours() function from OpenCV
	- Works when length approx variable is 4
	- Adjusted the filter size in gaussian function to remove noise
	- cv2.CHAIN_APPROX_SIMPLE used to save memory (only provides 4 corner points)
	
reference - https://www.pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/


Approach 2:

	- Implemented using Harris corner detector method
	- Needs to adjust input parameters in cornerHarris() function to get corner positions
	- Adjusted the points' position in oldPoints variable to get proper transformed image


