import numpy as np
from PIL import Image

class Data_loader():

    def __init__(self, patch_list,dl, transform):

        self.patch_list = patch_list
        self.dl = dl
        self.transform = transform

    def __getitem__(self, index):

        (query,positive,negative) = self.patch_list[index]
        query = query[:, :, ::-1]
        positive = positive[:, :, ::-1]
        negative = negative[:, :, ::-1]
        query = Image.fromarray(query.astype(np.uint8))
        positive = Image.fromarray(positive.astype(np.uint8))
        negative = Image.fromarray(negative.astype(np.uint8))
        descript = self.dl[index]
        descript = (descript - np.min(descript))/np.ptp(descript)

        if self.transform is not None:
            query = self.transform(query)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return query,positive,negative,descript

    def __len__(self):
        return len(self.patch_list)