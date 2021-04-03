''' This file is a package contains helper functions for opencv and imgs '''

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import imutils
import natsort
import os


def PLOT_IMG_MAT(img, figure_num=1, show=True):
    '''
    Show img matplotlib style

    Parameters
    ------------
    img : Numpy array
        img to be shown
    figure_num : (Optional) int, default : 1
        Figure number
    show : (Optional) bool, default : True
        If true shows O/P immediately, else you need to type plt.show()

    Returns
    ------------
    None
    '''
    plt.figure(figure_num)
    plt.imshow(img)
    plt.axis("off")  # Don't show coordinate axis

    if (show is True):  # Show img if asked
        plt.show()
    return None


def GINPUT_ROUTINE(img, num_pts=-1, first_col='r', show_maximised=False, message=None):
    '''
    Get coordinates of points in img by clicking left mouse buttton, click middle click to end
    If num_pts == -1, then choose points indefinately until middle click of mouse

    Parameters
    ------------
    img : Numpy array
        Img where points are to be selected
    num_pts : (Optional) int, default = -1
        Number of points to be selected. If <= zero, then select point indefinately and exit using middle mouse click
    first_col : (Optional) String, default = 'r'
        Indicates the coordinates in the first column.
        If 'r', then first column of matrix indicates row coordinates.
        If 'c', then first column of matrix indicates col coordinates.
    show_maximized : Bool
        If true, then shows the maximised window

    Returns
    ------------
    coordinates : Numpy array
        Numpy array containing the coordinates of the points (row,col) format

    Notes
    -----------
    (0,0)---------------> +ve X axis / rows cordinates
        |
        |
        |
        |
        |
        +ve Y axis / columns cordinates
    '''
    Nrows, Ncol = img.shape[0], img.shape[1]
    plt.figure()

    # maximise the window
    if (show_maximised):
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.maximize()

    plt.imshow(imutils.opencv2matplotlib(img))
    plt.tight_layout()
    # Instruction of what to do
    if num_pts <= 0:
        if(message is None):
            plt.text(int(Ncol * 0.3), -50, "Select any number of points")  # First argument is coulum and second argument is row-
        elif(message is not None):
            plt.text(int(Ncol * 0.3), -50,str(message))
    else:
        if(message is None):
            plt.text(int(Ncol * 0.3), -50, ("Select " + str(num_pts) + " points"   ))  # First argument is coulm and second argument is row
        elif(message is not None):
            plt.text(int(Ncol * 0.3), -50,str(message))
    coordinates = plt.ginput(  n=num_pts, timeout=0   )  # timeout : time(sec) to wait until termination, if input not given
    coordinates = np.array( coordinates)  # Currenlty first column contains column coordinates

    # Exchange col1 and col2 to get in the form (row_coordinate, col_coordinate) using advance slicing
    if (first_col == 'r'):
        coordinates[:, [0, 1]] = coordinates[:, [1, 0]]
    elif (first_col == 'c'):
        coordinates = coordinates.copy()

    coordinates = np.floor(coordinates)  # Floor to make them integers

    # close the window
    plt.close()

    return coordinates.astype(int)  # Return coordinates as integers


def RESIZE_IMG(img, fx1, fy1):
    '''
    Function to resize img

    Parameters
    ------------
    img1 : Numpy array
        img to be resized
    fx1 : float
        Horizontal stretch (>0.1)
    fy1 : float
        Vertical stretch (>0.1)

    Returns
    ------------
    Resized img
    '''
    if (img != None):
        return cv2.resize(img, (0, 0), fx=fx1, fy=fy1)
    else:
        print("img incorrect/img is NONE")


def PLOT_IMG_CV(img, wait_time=0, window_name="name"):
    '''
    Show the img in OpenCV format

    Parameters
    ------------
    img : Numpy array
        img to show
    window_name : (Optional) String
        Window name
    wait_time : (Optional) int
        Wait time before closng the window automatically. If zero, then waits indefinately for user input

    Returns
    ------------
    k : str
        It is the waitkey pressed
    '''
    cv2.imshow(window_name, img)
    k = cv2.waitKey(wait_time) & 0xFF
    return k


def PLOT_COLOR_HISTOGRAM(img, show=True, color=('b', 'g', 'r')):
    '''
    Plot color histogram of img

    Parameters
    ------------
    img : Numpy array
        img whose histogram is to be calculated
    show : (Optional) bool, default : True
        If true shows O/P immediately, else you need to type plt.show()
    color : (Optional) Tuple of strings, default : ('b', 'g', 'r')
        Colors to be used

    Returns
    ------------
    None
    '''
    color = ('blue', 'green', 'red')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col, label=str(color[i]))
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        max_pixel_intensity = max(
            np.max(img[:, :, 0]), np.max(img[:, :, 1]), np.max(img[:, :, 2]))
        min_pixel_intensity = max(
            np.min(img[:, :, 0]), np.min(img[:, :, 1]), np.min(img[:, :, 2]))
        plt.xlim([min_pixel_intensity, max_pixel_intensity])
        plt.legend()
    if (show is True):
        plt.show()


def PLOT_GRAY_HISTOGRAM(img, show=True):
    '''
    Plot color histogram of img

    Parameters
    ------------
    img : Numpy array
        img whose histogram is to be calculated
    show : (Optional) bool
        If true shows O/P immediately, else you need to type plt.show()
    color : (Optional) Tuple of strings
        Colors to be used

    Returns
    ------------
    None
    '''
    color = ('k')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col, label='Frequency count')
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        max_pixel_intensity = max(0, np.max(img[:, :]))
        min_pixel_intensity = min(0, np.min(img[:, :]))
        plt.xlim([min_pixel_intensity, max_pixel_intensity])
        plt.legend()
    if (show is True):
        plt.show()


def SOBEL(gimg):
    '''
    Return `Sobel` edge derivative image

    Parameters
    ------------
    gimg : 8 bit image
        8 bit gray scale image or 8 bit single channel image

    Returns
    ------------
    GMag : Numpy Array
        Numpy array of Magnitude of resultant X and Y gradient
    GMag_Norm : Numpy array
        Numpy array of Normalized Magnitude of resultant X and Y gradient, for viewing purpose.
     '''

    scale = 1
    delta = 0
    ddepth = cv2.CV_32F
    # Computing the X- and Y-Gradients, using the Sobel kernel
    grad_x = cv2.Sobel(
        gimg,
        ddepth,
        1,
        0,
        ksize=3,
        scale=scale,
        delta=delta,
        borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(
        gimg,
        ddepth,
        0,
        1,
        ksize=3,
        scale=scale,
        delta=delta,
        borderType=cv2.BORDER_DEFAULT)

    # Absolute Gradient for Display purposes --- Remove in future, not needed
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Gradient magnitude computation  --- Magnitude of the field --- Also for display
    g1 = grad_x * grad_x
    g2 = grad_y * grad_y
    GMag = np.sqrt(g1 + g2)  # Actual magnitude of the gradient

    # Normalized gradient 0-255 scale, and 0-1 scale --- For Display
    GMag_Norm = np.uint8(GMag * 255.0 / Gmag.max())  # Magnitude, for display

    return GMag, GMag_Norm


class TIMERS:
    '''
    Timer class to time functions. Interface same as matlab (TIC and TOC)

    Attributes
    ------------
    _start_time : Starting time
    _end_time : Ending time

    Useage
    ------------
    import helper_functions as hf
    hf.TIMERS t1;

    t1.TIC()
    .
    ... Your CODE HERE....
    .
    t1.TOC()

    Notes
    ------------
    The _varname in attributes indicates the variable is private and should not
    be accessed

    '''

    def __init__(self):
        self._start_time = None
        self._end_time = None
        pass

    def TIC(self):
        '''
        Starts the timer

        Parameters
        ------------
        None

        Returns
        ------------
        None
        '''
        self._start_time = time.time()
        self._end_time = None
        self.time_recorded = None

    def TOC(self, show_time=True):
        '''
        Stops the timer and optionally prints the time elapsed between TIC and TOC

        Parameters
        ------------
        show_time : bool
            If true, prints the time. Else you need to explicitly call self.PRINT_TIME()

        Returns
        ------------
        None
        '''
        if (self._start_time is None):
            print(
                "\n--- Timer not started. Use self.TIC() to start the timer ---\n"
            )
            # break
        else:
            self._end_time = time.time()
            if (show_time):
                if (self._end_time - self._start_time <
                        0.001):  # if time is less than 0.001 Sec, then show time in milllsec
                    print("Time : " + str((self._end_time - self._start_time) *
                                          1000) + "  Milli-seconds")
                else:
                    print("Time : " + str(self._end_time - self._start_time) +
                          "  seconds")
            self.time_recorded = self._end_time - self._start_time

    def GET_TIME(self):
        '''
        Returns the time calculated between TIC and TOC in seconds

        Parameters
        ------------
        None

        Returns
        ------------
        None
        '''
        if (self._start_time is None):
            print("Timer not started. Use self.TIC() to start the timer")
        elif (self._end_time is None):
            print("Timer not ended. Use self.TOC() to end the timer")
        else:
            return (self._end_time - self._start_time
                    )  # return time in seconds


def PUT_TXT_IMG_CV(img,
                   message,
                   location=None,
                   font_sz='s',
                   colour=(0, 0, 255)):
    '''
    Display text on image

    Parameters
    ------------
    img : Numpy array
        Image on which text is to be put
    message : String
        Text to be put on the image
    location : (Optional) Tuple of length 2, default : None
        Location of the message in (col, row) form. If not supplied, then puts message in top left corner
    font_sz : (Optional) String, default : "s" (small)
        Indicate small, medium, large font. Use 's' or 'm' or 'l' (small, medium, large)
    colour : (Optional) Tuple of ints , default : (0,0,255) (red color)
        Indicates the color of message, Tuple of length 3 for color image, tuple of length 1 for grayscale

    Returns
    ------------
    None
    '''
    font_sizes = {'s': 0.35, 'm': 0.75, 'l': 1.2}
    if (len(img.shape) ==
            3):  # if color image, then use Red color to write the text
        if (location is None):
            r, c, colors = img.shape
            cv2.putText(
                img,
                text=message,
                org=(int(0.1 * r), int(0.1 * c)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_sizes[font_sz],
                color=(0, 0, 255),
                thickness=2,
                lineType=8)
        else:
            cv2.putText(
                img,
                text=message,
                org=location,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_sizes[font_sz],
                color=colour,
                thickness=1,
                lineType=4)
    else:  # if grayscale image, then use Gray color to write the text
        if (location is None):
            r, c = img.shape
            cv2.putText(
                img,
                text=message,
                org=(int(0.1 * r), int(0.1 * c)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_sizes[font_sz],
                color=(150),
                thickness=2,
                lineType=5)
            pr
        else:
            cv2.putText(
                img,
                text=message,
                org=location,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_sizes[font_sz],
                color=(150),
                thickness=2,
                lineType=8)


def SKIP_FRAMES(video_source, num_skip):
    '''
    Skips frames from video

    Parameters
    ------------
    video_source : OpenCV VideoCapture object
        Video where frames are to be skipped
    num_skip : int
        Number of frames to skip

    Returns
    ------------
    None
    '''
    for i in range(num_skip):  # num_skip incicates number of frames
        ret, old_frame = video_source.read()


def ORDER_POINTS(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def FOUR_POINT_TRANSFORM(img, pts):
    '''
    Obtain perspective transform of any image

    Parameters
    ------------
    img : Numpy array
        Image which is to be transformed

    pts : Numpy array, shape of array is (4,2)
        Numpy array containing coordinates of 4 points arranged in (col, row) form.
        The coordinates should be arranged clockwise, in following order ONLY
        top-left --> top-right --> bottom-right --> bottom-left

    Returns
    ------------
    warped : Numpy matrix
        Returns a transformed image

    See Also
    ------------
    helper_functions.GINPUT_ROUTINE() to obtain points from image
    '''
    # obtain a consistent order of the points and unpack them
    # individually
    rect = ORDER_POINTS(pts)
    (tl, tr, br, bl) = rect
    print(tl, tr, br, bl)

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
    heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1],
         [0, maxHeight - 1]],
        dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    print(M.shape)
    print(M)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


# def GET_FILES_IN_FOLDER(folder_name, do_sort=True, abs_path=False):
#     '''
#     Returns a list of files in folder. Does not recurisively scan directories inside
#     the given folder_name

#     Paramters
#     ------------
#     folder_name : str
#         The input folder where files are to be found
#     do_sort : (Optional) Bool
#         If true, then sorts the list of files naturally
#     abs_path : (Optional) Bool
#     	If true, return the absolute path of the file
#     	else, returns only the list of file in the given folder name

#     Returns
#     ------------
#     only_files_in_folder : list
#         The list contining the file names in the folder

#     Note
#     ------------
#     Naturally sorted means --
#         ['Image1.jpg', 'image1.jpg', 'image3.jpg', 'image12.jpg', 'image15.jpg']
#     '''
#     # read files from a folder
#     if (not abs_path):# return file names WITHOUT absolute path 
#         only_files_in_folder = [
#             f for f in os.listdir(folder_name)
#             if os.path.isfile(os.path.join(folder_name, f))
#         ]
#     else: # return the list file WITH ABSOLUTE of path
#         only_files_in_folder = [
#             os.path.abspath(f) for f in os.listdir(folder_name)
#             if os.path.isfile(os.path.join(folder_name, f))
#         ]

#     if (do_sort):
#         # sort the images natural sort order
#         only_files_in_folder = natsort.natsorted(only_files_in_folder)

#     return only_files_in_folder


def GET_FILES_IN_FOLDER(folder_name, do_sort=True, path_type='absolute'):
    '''
    Returns a list of files in folder. Does not recurisively scan directories inside
    the given folder_name

    Paramters
    ------------
    folder_name : str
        The input folder where files are to be found
    do_sort : (Optional) Bool
        If true, then sorts the list of files naturally
    path_type : (Optional) String, default = 'abs'
        There are 3 options for path type,
        'absolute' : returns files with their absolute path
        'relative' : returns files with relative path to the given folder name
        'no_path'  : reuturns only the file list without the path name

    Returns
    ------------
    only_files_in_folder : list
        The list contining the file names

    Note
    ------------
    Naturally sorted means --
        ['Image1.jpg', 'image1.jpg', 'image3.jpg', 'image12.jpg', 'image15.jpg']
    '''
    # read files from a folder
    if (path_type is "relative"):# return file names WITHOUT absolute path 
        only_files_in_folder = [
            os.path.join(folder_name, f) for f in os.listdir(folder_name)
            if os.path.isfile(os.path.join(folder_name, f))
        ]
    elif(path_type is "absolute"): # return the list file WITH ABSOLUTE of path
        only_files_in_folder = [
            os.path.abspath(f) for f in os.listdir(folder_name)
            if os.path.isfile(os.path.join(folder_name, f))
        ]
    elif(path_type is "no_path"):
        only_files_in_folder = [
            f for f in os.listdir(folder_name)
            if os.path.isfile(os.path.join(folder_name, f))
        ]

    if (do_sort):
        # sort the images natural sort order
        only_files_in_folder = natsort.natsorted(only_files_in_folder)

    return only_files_in_folder


def POINTS_TO_MASK(im, pts_list, is_close=False):
    """
    This function returns a mask from set of points

    paramters
    -----------
    im : Numpy array, single or multichannel
        The image from which mask is to be applied
    pts_list : List of List
        List of points in the form of [  [col, row], [col, row] .... ]
    is_close : Bool
        If the last point in the pts_list is same as 1st point, them
        put True, else put false

    returns
    ------------
    mask_bw : Numpy array, single or multichannel
        The black and white mask of the selected points
    im_segmented : Numpy array, single or multichannel
        The segmented image, where mask is applied to the given im
    cropped_mask_bw : Numpy array, single or multichannel
        The b/w mask image with the bounding box of the mask
    cropped_mask_color : Numpy array, single or multichannel
        The color mask image with the bounding box of the mask

    """
    mask_bw = np.zeros_like(im)

    if(len(im.shape)==2):
        img_is_grayscale = True
        color = 255
    else:
        color = (255,255,255)
        img_is_grayscale = False

    # now create a mask out of points
    if(is_close == False):
        # then append the first point in the list
        # to make a closed polygon
        # pts_list.append(pts_list[0])
        cv2.fillConvexPoly(mask_bw, np.int32([pts_list.append(pts_list[0])]) , color)
    else:
        cv2.fillConvexPoly(mask_bw, np.int32([pts_list]) , color)

    print(pts_list)


    # apply mask to image
    img_segmented = np.bitwise_and(mask_bw, im)
    
    # now get the cropped mask
    # if(img_is_grayscale)
    x,y,w,h = cv2.boundingRect(cv2.convexHull(np.int32([pts_list])))

    cropped_mask_bw = mask_bw[y:y+h, x:x+w]
    cropped_mask_color = img_segmented[y:y+h, x:x+w]
    
    return mask_bw, img_segmented, cropped_mask_bw, cropped_mask_color

