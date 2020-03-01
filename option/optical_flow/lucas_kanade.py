import cv2
import numpy as np
import argparse
import glob
import os

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-image', type=str)

    args = parser.parse_args()
    args.ft_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    args.lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    return args

def main(args):

    # get the name of image files
    name_file = os.path.join(args.dir_image, '*')
    list_file = glob.glob(name_file)
    list_file.sort()

    for i in range(len(list_file)):
        image_original = cv2.imread(list_file[i])
        mask = np.zeros_like(image_original)
        image_earlier = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        image_earlier = cv2.resize(image_earlier, (32, 32))
        ft_1 = cv2.goodFeaturesToTrack(image_earlier, mask=None, **args.ft_params)

        image_later = cv2.imread(list_file[i + 1])
        image_later = cv2.cvtColor(image_later, cv2.COLOR_BGR2GRAY)
        image_later = cv2.resize(image_later, (32, 32))
        ft_2, status, err = cv2.calcOpticalFlowPyrLK(image_earlier, image_later, ft_1, None, **args.lk_params)

        good_1 = ft_1[status == 1]
        good_2 = ft_2[status == 1]

        # 特徴点とオプティカルフローをフレーム・マスクに描画
        for i, (pt2, pt1) in enumerate(zip(good_2, good_1)):
            x1, y1 = pt1.ravel()
            x2, y2 = pt2.ravel()
            mask = cv2.line(mask, (x2, y2), (x1, y1), [0, 0, 200], 1)
            # image_original = cv2.circle(image_original, (x2, y2), 5,  [0, 0, 200], -1)

        # フレームとマスクの論理積（合成）
        img = cv2.add(image_original, mask)
        cv2.imwrite('mask.png', img) # ウィンドウに表示

if __name__ == "__main__":
    args = argument()
    main(args)
