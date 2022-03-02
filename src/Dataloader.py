"""Dataloader for Flare Transformer"""

import torch
import numpy as np
from torch.utils.data import Dataset


class TrainDataloader(Dataset):
    def __init__(self, split, params, image_type="magnetogram", path="data/data_"):
        self.path = path
        self.window_size = params["window"]
        year_split = params["year_split"]

        # get x
        self.img = self.get_multiple_year_image(year_split[split], image_type)
        # self.img = self.img[:, np.newaxis] # without fancy index

        # get label
        self.label = self.get_multiple_year_data(year_split[split], "label")

        # get feat
        self.feat = self.get_multiple_year_data(year_split[split],
                                                "feat")[:, :90]

        # get window
        self.window = self.get_multiple_year_window(
            year_split[split], "window_48")[:, :self.window_size]
        self.window = np.asarray(self.window, dtype=int)

        print("img: {}".format(self.img.shape),
              "label: {}".format(self.label.shape),
              "feat: {}".format(self.feat.shape),
              "window: {}".format(self.window.shape))

    def __len__(self):
        """
        Returns:
            [int]: [length of sample]
        """
        return len(self.label)

    def __getitem__(self, idx):
        """
            get sample
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #  fancy index
        mul_x = self.img[np.asarray(self.window[idx][:self.window_size],
                                    dtype=int)]
        mul_feat = self.feat[np.asarray(self.window[idx][:self.window_size],
                                        dtype=int)]
        sample = ((mul_x - self.mean) / self.std,
                  self.label[idx],
                  mul_feat)

        return sample

    def get_multiple_year_image(self, year_dict, image_type):
        """
            concatenate data of multiple years [image]
        """
        for i, year in enumerate(range(year_dict["start"],
                                 year_dict["end"]+1)):
            data_path = self.path + str(year) + "_" + image_type + ".npy"
            print(data_path)
            image_data = np.load(data_path)
            if i == 0:
                result = image_data
            else:
                result = np.concatenate([result, image_data], axis=0)
            # print(result.shape)
        return result

    def get_multiple_year_data(self, year_dict, data_type):
        """
            concatenate data of multiple years [feat/label]
        """
        for i, year in enumerate(range(year_dict["start"],
                                 year_dict["end"]+1)):
            data_path = self.path + str(year) + "_" + data_type + ".csv"
            print(data_path)
            data = np.loadtxt(data_path, delimiter=',')
            if i == 0:
                result = data
            else:
                result = np.concatenate([result, data], axis=0)
        return result

    def get_multiple_year_window(self, year_dict, data_type):
        """
            concatenate data of multiple years [window]
        """
        num_data = 0
        for i, year in enumerate(range(year_dict["start"],
                                 year_dict["end"]+1)):
            data_path = self.path + str(year) + "_" + data_type + ".csv"
            print(data_path)
            data = np.loadtxt(data_path, delimiter=',')
            data += num_data
            if i == 0:
                result = data
            else:
                result = np.concatenate([result, data], axis=0)
            num_data += data.shape[0]
        return result

    def calc_mean(self):
        """
            calculate mean and std of images
        """
        bs = 1000000000
        ndata = np.ravel(self.img)
        mean = np.mean(ndata)
        std = 0
        for i in range(ndata.shape[0] // bs + 1):
            tmp = ndata[bs*i:bs*(i+1)] - mean
            tmp = np.power(tmp, 2)
            std += np.sum(tmp)
            print("Calculating std : ", i, "/", ndata.shape[0] // bs)
        std = np.sqrt(std / len(ndata))
        return mean, std

    def set_mean(self, mean, std):
        """
            Set self.mean and self.std
        """
        self.mean = mean
        self.std = std
