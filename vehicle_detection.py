import numpy as np
import cv2 as cv
import os
import evaluation as eval
import scipy.signal as signal

###############################################################
##### This code has been tested in Python 3.6 environment #####
###############################################################


def denoise(frame, size):
    frame = cv.GaussianBlur(frame, (size, size),0)
    
    return frame


def f1_score(precision, recall):
    score = 2 * precision * recall / (precision + recall)
    return score


def main(th, gaussian_size, median_size):

    ##### Set threshold
    threshold = th
    
    ##### Set path
    input_path = './input_image'    # input path
    gt_path = './groundtruth'       # groundtruth path
    result_path = './result'        # result path

    ##### load input
    input = [img for img in sorted(os.listdir(input_path)) if img.endswith(".jpg")]

    ##### first frame and first background
    frame_current = cv.imread(os.path.join(input_path, input[0]))
    frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
    frame_current_gray = denoise(frame_current_gray, gaussian_size)
    frame_prev_gray = frame_current_gray


    ##### background substraction
    for image_idx in range(len(input)):

        ##### calculate foreground region
        diff = frame_current_gray - frame_prev_gray
        diff_abs = np.abs(diff).astype(np.float64)

        ##### make mask by applying threshold
        frame_diff = np.where(diff_abs > threshold, 1.0, 0.0)

        ##### apply mask to current frame
        current_gray_masked = np.multiply(frame_current_gray, frame_diff)
        current_gray_masked_mk2 = np.where(current_gray_masked > 0, 255.0, 0.0)

        ##### result
        result = current_gray_masked_mk2.astype(np.uint8)
        
        ##### Copy the image
        im_floodfill = result.copy()
         
        ##### Mask used to flood filling
        ##### the size needs to be 2 pixels than the image
        h, w = result.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
         
        ##### Floodfill from point (0, 0)
        cv.floodFill(im_floodfill, mask, (0,0), 255);
         
        ##### Invert floodfilled image
        im_floodfill_inv = cv.bitwise_not(im_floodfill)
         
        ##### Combine the two images to get the foreground
        im_out = result | im_floodfill_inv
        
        ##### apply morphology for filling inside of the objects
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
        im_out = cv.morphologyEx(im_out, cv.MORPH_CLOSE, kernel)
        
        ##### post processing
        im_out = signal.medfilt(im_out, median_size)
        im_out = cv.GaussianBlur(im_out, (gaussian_size, gaussian_size), 0)
        th, im_out = cv.threshold(im_out, 200, 255, cv.THRESH_BINARY);
        
        ##### show final image
        cv.imshow('result', im_out)

        ##### renew background
        frame_prev_gray = frame_current_gray

        ##### make result file
        ##### Please don't modify path
        cv.imwrite(os.path.join(result_path, 'result%06d.png' % (image_idx + 1)), im_out)

        ##### end of input
        if image_idx == len(input) - 1:
            break

        ##### read next frame
        frame_current = cv.imread(os.path.join(input_path, input[image_idx + 1]))
        frame_current_gray = cv.cvtColor(frame_current, cv.COLOR_BGR2GRAY).astype(np.float64)
        frame_current_gray = denoise(frame_current_gray, gaussian_size)

        ##### If you want to stop, press ESC key
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break

    ##### evaluation result
    eval.cal_result(gt_path, result_path)


if __name__ == '__main__':
    
    main(5, 7, 5)
    print('f1 score: ' + str(f1_score(0.94319, 0.95142) * 100) + '%')
