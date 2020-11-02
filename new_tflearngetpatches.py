import cv2
import numpy as np
import random
from contrast_stretch import contrast_enhance


class patches:
    def __init__(self,rgb_path,thermal_path,rgb,thermal,npatch):
        self.rgb_path = rgb_path
        self.thermal_path = thermal_path
        self.rgb = rgb
        self.thermal = thermal
        self.npatch = npatch

    def getpatches(self):
        indices = list(zip(self.rgb, self.thermal))
        random.shuffle(indices)
        self.rgb, self.thermal = zip(*indices)
        all_patch_match = []
        ll = 15
        ul = 17
        ref = np.empty((ul + ll, ul + ll, 3)).shape
        g_truth = []

        ll = 15
        ul = 17
        ref = np.empty((ul + ll, ul + ll, 3)).shape

        for i in range(0, self.npatch):

            # reading images
            img_rgb1 = cv2.imread(self.rgb_path + str(self.rgb[i]))
            img_rgb1 = cv2.cvtColor(img_rgb1, cv2.COLOR_BGR2RGB)

            img_thermal1 = cv2.imread(self.thermal_path + str(self.thermal[i]))

            if np.std(img_thermal1)<40:
                c_s = contrast_enhance(img_thermal1, img_thermal1)
                img_thermal1, img_thermal2 = c_s.CE()

            # keypoints
            sift = cv2.xfeatures2d.SIFT_create()

            kp, descriptors_1 = sift.detectAndCompute(img_rgb1, None)

            list_kp = []

            for idx in range(0, len(kp)):
                x = round(kp[idx].pt[0])
                y = round(kp[idx].pt[1])
                if x > ul + ll and y > ul + ll:
                    list_kp.append([x, y])

            for j in range(0, len(list_kp)):

                T_patch1 = img_rgb1[round(list_kp[j][1] - ll):round(list_kp[j][1] + ul),
                           round(list_kp[j][0] - ll):round(list_kp[j][0] + ul)]
                T_patch2 = img_thermal1[round(list_kp[j][1] - ll):round(list_kp[j][1] + ul),
                           round(list_kp[j][0] - ll):round(list_kp[j][0] + ul)]
                random_pixel = random.randint(7, 500)
                T_patch3 = img_thermal1[random_pixel - ll:random_pixel + ul, random_pixel - ll:random_pixel + ul]

                val1 = np.reshape(T_patch1[:, :, 0], -1)
                val2 = np.reshape(T_patch2[:, :, 0], -1)

                if np.std(img_thermal1) < 40:
                    if np.std(val1) > 40 and np.std(val2) > 29:
                        # all matches stored in tuple within a list
                        if ref == T_patch2.shape and ref == T_patch3.shape:
                            all_patch_match.append((T_patch2,float(1)))
                            all_patch_match.append((T_patch3,float(0)))
                            # g_truth.extend([1, 0])

                elif np.std(img_thermal1) > 40:
                    if np.std(val1) > 50 and np.std(val2) > 25:
                        # all matches stored in tuple within a list
                        if ref == T_patch2.shape and ref == T_patch3.shape:
                            all_patch_match.append((T_patch2,float(1)))
                            all_patch_match.append((T_patch3,float(0)))
                            # g_truth.extend([1, 0])



        return all_patch_match, None