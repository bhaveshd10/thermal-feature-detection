import numpy as np
from PIL import Image
from scipy.special import expit, logit

class Data_loader():

    def __init__(self, patch_list, transform):

        self.patch_list = patch_list
        self.transform = transform

    def __getitem__(self, index):

        (query,kpt) = self.patch_list[index]
        query = query[:, :, ::-1]

        query = Image.fromarray(query.astype(np.uint8))
        # kp_list= self.kpl[index]

        if self.transform is not None:
            query = self.transform(query)

        return (query,kpt)

    def __len__(self):
        return len(self.patch_list)
