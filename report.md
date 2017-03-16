## Writeup Report

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/output_5_0.png "Chessboard"
[image2]: ./output_images/output_6_0.png "Undistorted Case1"
[image3]: ./output_images/output_6_1.png "Undistorted Case2"
[image4]: ./output_images/output_9_0.png "Road Transformed"
[image5]: ./output_images/output_12_0.png "Comb Thresh Case1"
[image6]: ./output_images/output_12_1.png "Comb Thresh Case2"
[image7]: ./output_images/output_13_0.png "Color Thresh Case1"
[image8]: ./output_images/output_13_1.png "Color Thresh Case2"
[image9]: ./output_images/output_16_0.png "Perspective Transform"
[image10]: ./output_images/output_17_1.png "Binary Perspective Transform"
[image11]: ./output_images/output_20_1.png "Histogram"
[image12]: ./output_images/output_20_3.png "Sliding Windows"
[image13]: ./output_images/output_20_4.png "Next frame"
[image14]: ./output_images/output_21_0.png "Convolution"
[image15]: ./output_images/curve_formula.png "Curvature"
[image16]: ./output_images/output_27_1.png "Output"
[image17]: ./output_images/output_45_0.png "Fail identified"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading the report now!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Here is the step for Camera Calibration:    
First, loaded all the images under camera_cal folder.    
Second, created object points and image points.    
Third, find the chessboard corners and add object points and image points once the corners are found.    
Fourth, Using the object points and image points to computed the camera matrix and distortion coefficients via cv2.calibrateCamera function.    

```python
# prepare object points
nx = 9 #the number of inside corners in x
ny = 6 #the number of inside corners in y

# Read in an image
images = glob.glob('camera_cal/calibration*.jpg')

def cal_camera(images, nx, ny):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            plt.imshow(img)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2], None, None)
    return mtx, dist

mtx, dist = cal_camera(images, nx, ny)
```
    
![alt text][image1]    
To verify the Camera Calibration step, I use the mtx and dist which return from cal_camera to distortion-corrected image.
```python
def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
```	    
![alt text][image2]    
![alt text][image3]    

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Now we use the same mtx and dist to the road lane image to verify it.    
![alt text][image4]

#### 2. Describe how you used color transforms, gradients or other methods to create a thresholded binary image.
I had tried several combinations to decide which is the best way.
1. Combination the gradient measurements with x, y, magnitude and direction:

```python
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output
def comb_thresh(img):
    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined
```
Let's see the result.	
![alt text][image5]
It looks working. How about the yellow line?
![alt text][image6]
Looks like is fail on recognize the yellow.
Let's try different method.

HLS Color and Gradient:

```python
def comb_color_thresh(img, s_thresh=(90, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary
```	
![alt text][image7]
![alt text][image8]
It works on both white and yellow line.
I decided to use HLS Color and Gradient to create the thresholded binary image

####3. Describe how you performed a perspective transform and provide an example of a transformed image.

Thers is the code I perspective transform image which included the source and destination region.

```python
def perspective_transform(img):
    HEIGHT, WIDTH = img.shape[:2]
    offset = 40
    src = np.array([[WIDTH*0.375, HEIGHT*0.67], [WIDTH*0.625, HEIGHT*0.67], [0, HEIGHT*0.95], [WIDTH, HEIGHT*0.95]], np.float32)
    dst = np.array([[0, 0], [WIDTH, 0], [0, HEIGHT], [WIDTH, HEIGHT]], np.float32)    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
    return warped, Minv
```	
This resulted in the following source and destination points:

| Source                         | Destination   | 
|:------------------------------:|:-------------:| 
| WIDTH*(3/8), HEIGHT*(2/3)      | 0, 0          | 
| WIDTH*(5/8), HEIGHT*(2/3)      | WIDTH, 0      |
| 0, HEIGHT*(0.95)               | 0, HEIGHT     |
| WIDTH, HEIGHT*(0.95)           | WIDTH, HEIGHT |

Here is the result after I apply the src and dst for the perspective transform original image.
![alt text][image9]
And this is how it looks like after perspective transform binary image.
![alt text][image10]

#### 4. Describe how you identified lane-line pixels and fit their positions with a polynomial?

At the first time, I use histogram and sliding windows on the perspective transform binary image to identify the lane-lines pixels and fit the 2nd order polynomial.
And then, I use the result which gets from sliding windows as base to fit the 2nd order polynomial for the next frame.

```python
def sliding_windows(binary_warped, nwindows = 9, margin = 100, minpix = 50, plot = False):
    '''
    Sliding the windows to find the lane lines
    '''
    binary_warped = np.copy(binary_warped)
    # Already created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if plot:
        #Visualization
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.title('Lane line from sliding windows')
        plt.show()
    return  left_fit, right_fit
```
Using the histogram to identify the lane-line pixels.    
![alt text][image11]    
Applied the result from histogram on sliding windows.    
![alt text][image12]    
Using the result from sliding windows as base on next frame.    
![alt text][image13]    

I also tried convolution method. But it didn't works well, I decide to drop this method.    
![alt text][image14]    

#### 5. Describe how you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I use the following formula to get the curvature.    
![alt text][image15]     
```python
def curvature(left_fit, right_fit):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image    
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    return left_curverad, right_curverad
```	
And we need to transform the curvature from pixels to real world distance and also apply the effect from perspective transform.
```python
def curvature_real_world(left_fit, right_fit):
    '''
    Calculate curvature for left and right lane line and transfer to real world distance
    '''
    #using 20 instead of 30 here since the perspective transform is 1.5x much longer than original image
    ym_per_pix = 20/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Generate x and y values from second order polynomial
    ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
    y_eval = np.max(ploty)
    leftx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    rightx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad
```
And use the 2nd order polynomial fit for left and right to find out the offset from the center of the lane and car position.
```python
def get_x_position(fit, y):
    '''
    Get x position from the polyfit
    '''
    return fit[0]*y**2 + fit[1]*y + fit[2]

def get_direction(left_fit, right_fit, img):
    '''
    Get position of center location
    '''
    xm_per_pix = 3.7/700
    ploty = img.shape[0]
    left_pos = get_x_position(left_fit, ploty)
    right_pos = get_x_position(right_fit, ploty)
    offest = (left_pos + right_pos)/2 - img.shape[0]
    direction = "left" if offest < 0 else "right"
    return abs(offest*xm_per_pix), direction
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After highlight the region between left and right lines, we use the Minv to revert the image from Perspective to original.
```python
def draw_lines(img, warped, left_fit, right_fit, Minv):
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    left_pts = np.array(np.transpose(np.vstack([left_fitx, ploty]))).reshape(-1,1,2)
    right_pts = np.array(np.flipud(np.transpose(np.vstack([right_fitx, ploty]))))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.polylines(color_warp, np.int_([left_pts]), True, (255, 0, 0), 50)
    cv2.polylines(color_warp, np.int_([right_pts]), True, (0, 0, 255), 50)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    return result
```

![alt text][image16]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Pipeline the above steps can success implementation lane line finder on the project video.
However, it fail on challenge part.
Since the lane line are dash line, some frame fail on identified the lane-line pixels.    
![alt text][image17]    
I implement line class to store the previous polynomial fit and add sanity checker can improve a lots.
But it need more work on tweak the sanity checker to make is more smooth and update at the right timing.

