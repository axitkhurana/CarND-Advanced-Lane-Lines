import pickle
from tracker import Tracker

import cv2
import numpy as np
from moviepy.editor import VideoFileClip

from image_gen import (
    abs_sobel_thresh, color_thresh, window_mask
)


dist_pickle = pickle.load(open('camera_cal/calibration_pickle.p', 'rb'))
mtx = dist_pickle['mtx']
dist = dist_pickle['dist']


def process_img(img):
    img = cv2.undistort(img, mtx, dist, None, mtx)
    preprocess_image = np.zeros_like(img[:,:,0])
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12, 255))
    grady = abs_sobel_thresh(img, orient='y', thresh=(25, 255))

    c_binary = color_thresh(img, sthresh=(100, 255), vthresh=(50, 255))
    preprocess_image[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255

    img_size = (img.shape[1], img.shape[0])
    bot_width = .76
    mid_width = .08
    height_pct = .62
    bottom_trim = .935
    src = np.float32([
        [img.shape[1]*(.5-mid_width/2), img.shape[0]*height_pct],
        [img.shape[1]*(.5+mid_width/2), img.shape[0]*height_pct],
        [img.shape[1]*(.5+bot_width/2), img.shape[0]*bottom_trim],
        [img.shape[1]*(.5-bot_width/2), img.shape[0]*bottom_trim],
    ])
    offset = img.shape[1]*.25
    dst = np.float32([
        [offset, 0],
        [img.shape[1]-offset, 0],
        [img.shape[1]-offset, img.shape[0]],
        [offset, img.shape[0]]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(preprocess_image, M, img_size, flags=cv2.INTER_LINEAR)

    window_width = 25
    window_height = 80
    curve_centers = Tracker(Mywindow_width=window_width, Mywindow_height=window_height, Mymargin=25, My_ym=10/720, My_xm=4/384, Mysmooth_factor=15)

    window_centroids = curve_centers.find_window_centroids(warped)

    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    leftx = []
    rightx = []

    for level in range(len(window_centroids)):
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)

        l_points[(l_points == 255) | (l_mask == 1)] = 255
        r_points[(r_points == 255) | (r_mask == 1)] = 255

    # Draw
    # template = np.array(r_points+l_points, np.uint8)
    # zero_channel=np.zeros_like(template)
    # template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
    # warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8)
    # result = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)
    # result = warped

    yvals = range(0, warped.shape[0])
    res_yvals = np.arange(warped.shape[0]-(window_height/2), 0, -window_height)

    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2), axis=0),
                                  np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2), axis=0),
                                  np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2), axis=0),
                                  np.concatenate((yvals, yvals[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img)
    cv2.fillPoly(road, [left_lane], color=[255, 0,0])
    cv2.fillPoly(road, [right_lane], color=[0, 0, 255])
    cv2.fillPoly(road, [inner_lane], color=[0, 255, 0])
    cv2.fillPoly(road_bkg, [left_lane], color=[255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color=[255, 255, 255])

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags=cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags=cv2.INTER_LINEAR)

    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0)
    result = cv2.addWeighted(base, 1.0, road_warped, .7, 0.0)

    ym_per_pix = curve_centers.ym_per_pix
    xm_per_pix= curve_centers.xm_per_pix

    curve_fit_cr = np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix, np.array(leftx, np.float32)*xm_per_pix, 2)
    curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1]*ym_per_pix + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])
    camera_center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos = 'left'
    if center_diff <=0:
        side_pos = 'right'

    cv2.putText(result, 'Radius of Curvature = '+str(round(curverad, 3))+'(m)',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff, 3)))+'m '+side_pos+' of center',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    return result


if __name__ == '__main__':
    output_video = 'output_tracked1.mp4'
    input_video = 'project_video.mp4'

    clip1 = VideoFileClip(input_video)
    video_clip = clip1.fl_image(process_img)
    video_clip.write_videofile(output_video, audio=False)
