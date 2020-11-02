import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
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
        descript_list = []
        final_dl = []
        ll = 15
        ul = 17
        ref = np.empty((ul + ll, ul + ll, 3)).shape

        for i in range(0, self.npatch):

            # reading images
            img_rgb1 = cv2.imread(self.rgb_path + str(self.rgb[i]))
            img_rgb1 = cv2.cvtColor(img_rgb1, cv2.COLOR_BGR2RGB)

            img_thermal1 = cv2.imread(self.thermal_path + str(self.thermal[i]))

            # if np.std(img_thermal1)<40:
            #     c_s = contrast_enhance(img_thermal1, img_thermal1)
            #     img_thermal1, img_thermal2 = c_s.CE()

            # keypoints
            sift = cv2.xfeatures2d.SIFT_create()

            kp, dp = sift.detectAndCompute(img_rgb1, None)

            list_kp = []
            for idx in range(0, len(kp)):
                x = round(kp[idx].pt[0])
                y = round(kp[idx].pt[1])
                if x > 7 and y > 7:
                    list_kp.append([x, y])
                    descript_list.append(dp[idx])
            # plt.imshow(img_thermal1)
            # plt.show()
            # print(np.std(img_thermal1))
            for j in range(0, len(list_kp)):

                T_patch1 = img_rgb1[round(list_kp[j][1] - ll):round(list_kp[j][1] + ul),
                           round(list_kp[j][0] - ll):round(list_kp[j][0] + ul)]
                T_patch2 = img_thermal1[round(list_kp[j][1] - ll):round(list_kp[j][1] + ul),
                           round(list_kp[j][0] - ll):round(list_kp[j][0] + ul)]
                random_pixel = random.randint(33, 500)
                T_patch3 = img_thermal1[random_pixel - ll:random_pixel + ul, random_pixel - ll:random_pixel + ul]

                val1 = np.reshape(T_patch1[:, :, 0], -1)
                val2 = np.reshape(T_patch2[:, :, 0], -1)

                # if abs(np.std(val1) - np.std(val2)) > 10:

                if np.std(img_thermal1) < 40:
                    if np.std(val1) > 40 and np.std(val2) > 29:
                        # all matches stored in tuple within a list
                        if ref == T_patch1.shape and ref == T_patch2.shape and ref == T_patch3.shape:
                            all_patch_match.append((T_patch1, T_patch2, T_patch3))
                            # res = cv2.hconcat([T_patch1, T_patch2])
                            # print((np.std(val1),np.std(val2)))
                            # imS = cv2.resize(res, (128, 64))
                            # cv2.imshow('image', imS)
                            # cv2.waitKey(5000)
                            # cv2.destroyAllWindows()
                            final_dl.append(descript_list[j])

                elif np.std(img_thermal1) > 40:
                    if np.std(val1) > 50 and np.std(val2) > 25:
                        # all matches stored in tuple within a list
                        if ref == T_patch1.shape and ref == T_patch2.shape and ref == T_patch3.shape:
                            all_patch_match.append((T_patch1, T_patch2, T_patch3))
                            # res = cv2.hconcat([T_patch1, T_patch2])
                            # print((np.std(val1), np.std(val2)))
                            # imS = cv2.resize(res, (128, 64))
                            # cv2.imshow('image', imS)
                            # cv2.waitKey(5000)
                            # cv2.destroyAllWindows()
                            final_dl.append(descript_list[j])


        return all_patch_match, final_dl






# import cv2
# import numpy as np
# import random
#
# class patches:
#     def __init__(self,rgb_path,thermal_path,rgb,thermal,npatch):
#         self.rgb_path = rgb_path
#         self.thermal_path = thermal_path
#         self.rgb = rgb
#         self.thermal = thermal
#         self.npatch = npatch
#
#     def getpatches(self):
#
#         # indices = list(zip(self.rgb, self.thermal))
#         # random.shuffle(indices)
#         # self.rgb, self.thermal = zip(*indices)
#         all_patch_match = []
#         descript_list = []
#         final_dl = []
#         ll = 15
#         ul = 17
#         ref = np.empty((ul + ll, ul + ll, 3)).shape
#
#
#         for i in range(0, self.npatch-1):
#             k = 0
#             # reading images
#             img_rgb1 = cv2.imread(self.rgb_path + str(self.rgb[i]))
#             img_rgb1 = cv2.cvtColor(img_rgb1, cv2.COLOR_BGR2RGB)
#             img_rgb2 = cv2.imread(self.rgb_path + str(self.rgb[i+1]))
#             img_rgb2 = cv2.cvtColor(img_rgb2, cv2.COLOR_BGR2RGB)
#
#             img_thermal1 = cv2.imread(self.thermal_path + str(self.thermal[i]))
#             img_thermal2 = cv2.imread(self.thermal_path + str(self.thermal[i+1]))
#
#             # keypoints
#             sift = cv2.xfeatures2d.SIFT_create()
#
#             kp1, d1 = sift.detectAndCompute(img_rgb1, None)
#             kp2, d2 = sift.detectAndCompute(img_rgb2, None)
#
#             bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#             matches = bf.match(d1, d2)
#
#             # Sort them in the order of their distance.
#             matches = sorted(matches, key=lambda x: x.distance)
#             matches = matches[:100]
#
#             # Initialize lists
#             list_kp1 = []
#             list_kp2 = []
#
#             # For each match extract a corresponding patch
#             for mat in matches:
#                 img1_idx = mat.queryIdx
#                 img2_idx = mat.trainIdx
#                 (x1, y1) = kp1[img1_idx].pt
#                 (x2, y2) = kp2[img2_idx].pt
#                 list_kp1.append([x1, y1])
#                 list_kp2.append([x2, y2])
#
#
#             r = np.random.randn(len(list_kp2))
#             norm = r / np.linalg.norm(r)
#             shift = np.int_(np.round(norm * 20))
#
#             shift = np.array([shift, shift]).reshape(-1, 2)
#             rand = (random.randint(1, 6)) ** 2
#             list_kp2 = np.array(list_kp2) + shift
#
#             list_kp3 = (list_kp2 + shift * rand)
#             list_kp3[list_kp3 < 0] = list_kp2[list_kp3 < 0]
#             list_kp2 = list_kp2.tolist()
#             list_kp3 = list_kp3.tolist()
#
#             for j in range(0, len(list_kp3)):
#
#                 T_patch1 = img_thermal1[round(list_kp1[j][1] - ll):round(list_kp1[j][1] + ul),
#                            round(list_kp1[j][0] - ll):round(list_kp1[j][0] + ul)]
#                 T_patch2 = img_thermal2[round(list_kp2[j][1] - ll):round(list_kp2[j][1] + ul),
#                            round(list_kp2[j][0] - ll):round(list_kp2[j][0] + ul)]
#                 # random_pixel = random.randint(65, 400)
#                 T_patch3 = img_thermal2[round(list_kp3[j][1] - ll):round(list_kp3[j][1] + ul),
#                            round(list_kp3[j][0] - ll):round(list_kp3[j][0] + ul)]
#
#                 #
#                 # val1 = np.reshape(T_patch2[:, :, 0], -1)
#                 # val2 = np.reshape(T_patch3[:, :, 0], -1)
#
#                 # if abs(np.std(val1) - np.std(val2)) > 10:
#                 # all matches stored in tuple within a list
#                 if ref == T_patch1.shape and ref == T_patch2.shape and ref == T_patch3.shape:
#                     res = cv2.hconcat([T_patch1, T_patch2])
#                     imS = cv2.resize(res, (128, 64))
#                     cv2.imshow('image', imS)
#                     cv2.waitKey(5000)
#                     cv2.destroyAllWindows()
#                     all_patch_match.append((T_patch1, T_patch2, T_patch3))
#
#                     k += 1
#                     if k == 100:
#                         break
#
#         return all_patch_match, None
