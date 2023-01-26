from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import os
from datasets import load_dataset


class NNDataLoader():
    train_data = None
    train_label = None
    val_data = None
    val_label = None
    test_data = None
    test_labels = None

    train = None
    val = None
    test = None

    def __init__(self) -> None:
        pass

    def load_csv(self, path):
        '''
        Load the CSV form of MNIST data without any external library
        :param path: the path of the csv file
        :return:
            data: A list of list where each sub-list with 28x28 elements
                corresponding to the pixels in each image
            labels: A list containing labels of images
        '''
        data = []
        labels = []
        with open(path, 'r') as fp:
            images = fp.readlines()
            images = [img.rstrip() for img in images]

            for img in images:
                img_as_list = img.split(',')
                y = int(img_as_list[0]) # first entry as label
                x = img_as_list[1:]
                x = [int(px) / 255 for px in x]
                data.append(x)
                labels.append(y)
        return data, labels
    
    
    def load_trainval_from_path(self, path, random_state=42, test_size=0.15):
        """
        Load training data with labels
        :return:
            train_data: A list of list containing the training data
            train_label: A list containing the labels of training data
            val_data: A list of list containing the validation data
            val_label: A list containing the labels of validation data
        """
        # Load training data
        print("Loading training data...")
        data, label = self.load_csv(path)
        assert len(data) == len(label)
        print("Training data loaded with {count} images".format(count=len(data)))

        self.train_data, self.train_labels, self.val_data, self.val_label = train_test_split(data, label, test_size = test_size, random_state = random_state, shuffle=True)
    
    
    def load_test_from_path(self, test_path):
        """
            Load testing data with labels
            :return:
                data: A list of list containing the testing data
                label: A list containing the labels of testing data
            """
        # Load training data
        print("Loading testing data...")
        self.test_data, self.test_label = self.load_csv(test_path)
        assert len(self.test_data) == len(self.test_label)
        print("Testing data loaded with {count} images".format(count=len(self.test_data)))

    def fasion_mnist(self, random_state=42, test_size=0.15):
        download = os.path.isdir("data/FasionMNIST")
        training_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=download,
            transform=ToTensor()
        )

        self.test = datasets.FashionMNIST(
            root="data",
            train=False,
            download=download,
            transform=ToTensor()
        )

        self.train, self.val = train_test_split(training_data, test_size = test_size, random_state = random_state, shuffle=True)
    
    # def imagenet(self, random_state=42, test_size=0.15):
    #     download = os.path.isdir("data/imagenet")
    #     training_data = datasets.ImageNet(
    #         root="data",
    #         train=True,
    #         download=download,
    #         transform=ToTensor()
    #     )

    #     self.test = datasets.ImageNet(
    #         root="data",
    #         train=False,
    #         download=download,
    #         transform=ToTensor()
    #     )

    #     self.train, self.val = train_test_split(training_data, test_size = test_size, random_state = random_state, shuffle=True)


    def tiny_imagenet(self, random_state=42, val_size = 0.15, test_size = 0.05):
        # download = os.path.isdir("data/imagenet")
        data = load_dataset('Maysee/tiny-imagenet', split='train')

        # testing = load_dataset('Maysee/tiny-imagenet', split='valid')
        # self.test = load_dataset('Maysee/tiny-imagenet', split='test')

        self.train, self.val = data.train_test_split(test_size = val_size, shuffle=True)
        _, self.test = data.train_test_split(test_size = test_size, shuffle=True)
        print(self.train)


    def create_dataloader(self, custom = False, batch_size=32):
        if not custom:
            train_dataloader = DataLoader(self.train, batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader(self.val, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(self.test, batch_size=batch_size, shuffle=True)

        else:
            train_dataloader = DataLoader([[self.train_data[i], self.train_label[i]] for i in range(len(self.train_label))], batch_size=batch_size, shuffle=True)
            val_dataloader = DataLoader([[self.val_data[i], self.val_label[i]] for i in range(len(self.val_label))], batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader([[self.test_data[i], self.test_label[i]] for i in range(len(self.test_label))], batch_size=batch_size, shuffle=True)

        return train_dataloader, val_dataloader, test_dataloader
