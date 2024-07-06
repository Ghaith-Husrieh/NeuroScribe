import os
import pickle
import tarfile
from collections import OrderedDict

import numpy as np

import neuroscribe as ns
from neuroscribe.core._utils._data import (_decompress_file, _request_file,
                                           _save_file)
from neuroscribe.utils.data.transforms import Compose
from neuroscribe.utils.data.utils import read_data

__all__ = ['Dataset', 'MNIST', 'FashionMNIST']


class Dataset:
    def __init__(self):
        self._device = 'cpu'
        super().__setattr__('_parameters', OrderedDict())

    def __setattr__(self, name, value):
        if isinstance(value, list):
            for val in value:
                if name in self._parameters:
                    if isinstance(self._parameters[name], list):
                        self._parameters[name].append(ns.tensor(val))
                    else:
                        self._parameters[name] = [
                            self._parameters[name], ns.tensor(val)]
                else:
                    self._parameters[name] = ns.tensor(val)

        elif isinstance(value, np.ndarray):
            if '_parameters' not in self.__dict__:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            if name in self._parameters:
                if isinstance(self._parameters[name], list):
                    self._parameters[name].append(ns.tensor(value))
                else:
                    self._parameters[name] = [
                        self._parameters[name], ns.tensor(value)]
            else:
                self._parameters[name] = ns.tensor(value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def to(self, device):
        self._device = device
        for name, param in self._parameters.items():
            if isinstance(param, list):
                self._parameters[name] = [item.to(device) for item in param]
            else:
                self._parameters[name] = param.to(device)


class _MNISTDataset(Dataset):
    def __init__(self, images_file_path, labels_file_path, transform=None):
        super().__init__()
        self.images = read_data(images_file_path)
        self.labels = read_data(labels_file_path)
        self.transform = Compose(transform) if isinstance(
            transform, list) else transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        if self.transform:
            x = ns.tensor(
                self.transform(x.asnumpy()),
                requires_grad=False,
                device=self._device
            )

        return x, y

    @staticmethod
    def _download_and_cache_mnist(root, name, base_url, urls):
        mnist_dir = os.path.join(root, name)
        os.makedirs(mnist_dir, exist_ok=True)
        for key, relative_path in urls.items():
            url = os.path.join(base_url, relative_path)
            file_name = relative_path.split('/')[-1]
            file_path = os.path.join(mnist_dir, file_name)
            decompressed_file_path = file_path[:-
                                               3] if file_path.endswith('.gz') else file_path

            if not os.path.exists(decompressed_file_path):
                if not os.path.exists(file_path):
                    response = _request_file(url)
                    _save_file(response, file_path)
                _decompress_file(file_path)

    @staticmethod
    def _get_dataset(root, dataset_name, base_url, urls, train, download, transform):
        if download:
            _MNISTDataset._download_and_cache_mnist(
                root, dataset_name, base_url, urls)

        subdir = 'MNIST' if dataset_name == 'mnist' else 'fashion_mnist'
        if train:
            images_file_path = os.path.join(
                root, subdir, "train-images-idx3-ubyte")
            labels_file_path = os.path.join(
                root, subdir, "train-labels-idx1-ubyte")
        else:
            images_file_path = os.path.join(
                root, subdir, "t10k-images-idx3-ubyte")
            labels_file_path = os.path.join(
                root, subdir, "t10k-labels-idx1-ubyte")

        return _MNISTDataset(images_file_path, labels_file_path, transform=transform)


class _CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.transform = Compose(transform) if isinstance(
            transform, list) else transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx.item()]
        y = self.labels[idx.item()]

        if self.transform:
            x = ns.tensor(
                self.transform(x.asnumpy()),
                requires_grad=False,
                device=self._device
            )

        return x, y

    @staticmethod
    def _download_and_cache_cifar10(root, base_url, file_name):
        cifar10_dir = os.path.join(root, 'cifar10')
        os.makedirs(cifar10_dir, exist_ok=True)
        file_path = os.path.join(cifar10_dir, file_name)

        if not os.path.exists(file_path):
            response = _request_file(base_url + file_name)
            _save_file(response, file_path)

        with tarfile.open(file_path, 'r:gz') as tar:
            tar.extractall(path=cifar10_dir)

    @staticmethod
    def _load_data_from_batch(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict[b'data'], dict[b'labels']

    @staticmethod
    def _load_data(images_file_path, labels_file_path):
        with open(images_file_path, 'rb') as fo:
            data = np.load(fo, allow_pickle=True)
        with open(labels_file_path, 'rb') as fo:
            labels = np.load(fo, allow_pickle=True)
        data = data.reshape(-1, 3, 32, 32).astype(np.float32)
        return data, labels

    @staticmethod
    def _save_data_as_npy(images_file_path, labels_file_path, data, labels):
        np.save(images_file_path, data)
        np.save(labels_file_path, labels)

    @staticmethod
    def _get_dataset(root, dataset_name, base_url, file_name, train, download, transform):
        if download:
            _CIFAR10Dataset._download_and_cache_cifar10(
                root, base_url, file_name)

        cifar10_dir = os.path.join(root, 'cifar10')
        if train:
            data, labels = [], []
            for batch_num in range(1, 6):
                batch_file = os.path.join(
                    cifar10_dir, f'cifar-10-batches-py/data_batch_{batch_num}')
                batch_data, batch_labels = _CIFAR10Dataset._load_data_from_batch(
                    batch_file)
                data.append(batch_data)
                labels.append(batch_labels)
            data = np.vstack(data).reshape(-1, 3, 32, 32).astype(np.float32)
            labels = np.hstack(labels)
        else:
            test_batch_file = os.path.join(
                cifar10_dir, 'cifar-10-batches-py/test_batch')
            data, labels = _CIFAR10Dataset._load_data_from_batch(
                test_batch_file)
            data = data.reshape(-1, 3, 32, 32).astype(np.float32)

        images_file_path = os.path.join(cifar10_dir, 'images.npy')
        labels_file_path = os.path.join(cifar10_dir, 'labels.npy')
        _CIFAR10Dataset._save_data_as_npy(
            images_file_path, labels_file_path, data, labels)

        return _CIFAR10Dataset(data, labels, transform=transform)


def CIFAR10(root, train=True, download=True, transform=None):
    _BASE_URL = "https://www.cs.toronto.edu/~kriz/"
    _FILE_NAME = "cifar-10-python.tar.gz"
    return _CIFAR10Dataset._get_dataset(root, 'cifar10', _BASE_URL, _FILE_NAME, train, download, transform)


def MNIST(root, train=True, download=True, transform=None):
    _BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    _URLS = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    return _MNISTDataset._get_dataset(root, 'mnist', _BASE_URL, _URLS, train, download, transform)


def FashionMNIST(root, train=True, download=True, transform=None):
    _BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
    _URLS = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    return _MNISTDataset._get_dataset(root, 'fashion_mnist', _BASE_URL, _URLS, train, download, transform)
