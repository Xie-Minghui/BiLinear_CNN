# -*- coding: utf-8 -*
import re
import os
import pickle

import numpy as np
import PIL.Image

class CUB200():
    """CUB200 dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _train_data, list of np.ndarray.
        _train_labels, list of int.
        _test_data, list of np.ndarray.
        _test_labels, list of int.
    """

    def __init__(self, root, train=True,download=False):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        self._train = train

        if self._checkIntegrity():
            print('Files already downloaded and verified.')
        elif download:
            url = ('http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/'
                   'CUB_200_2011.tgz')
            self._download(url)
            # print("Already downloaded...")
            self._extract()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')

        # Now load the picked data.
        if self._train:
            self._train_data, self._train_labels = pickle.load(open(
                os.path.join(self._root, 'processed/train.pkl').replace(r'\\', '/'), 'rb'))
            assert (len(self._train_data) == 5994
                    and len(self._train_labels) == 5994)
        else:
            self._test_data, self._test_labels = pickle.load(open(
                os.path.join(self._root, 'processed/test.pkl').replace(r'\\', '/'), 'rb'))
            assert (len(self._test_data) == 5794
                    and len(self._test_labels) == 5794)

    def __getitem__(self, index):
        """
        Args:
            index, int: Index.

        Returns:
            image, PIL.Image: Image of the given index.
            target, str: target of the given index.
        """
        if self._train:
            image, target = self._train_data[index], self._train_labels[index]
        else:
            image, target = self._test_data[index], self._test_labels[index]
        # Doing this so that it is consistent with all other datasets.
        image = PIL.Image.fromarray(image)

        return image, target

    def __len__(self):
        """Length of the dataset.

        Returns:
            length, int: Length of the dataset.
        """
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _checkIntegrity(self):
        """Check whether we have already processed the data.

        Returns:
            flag, bool: True if we have already processed the data.
        """
        return (
                os.path.isfile(re.sub('\\\\', '/', os.path.join(self._root, 'processed/train.pkl')))
                and os.path.isfile(re.sub('\\\\', '/', os.path.join(self._root, 'processed/test.pkl'))))

    def _download(self, url):
        """Download and uncompress the tar.gz file from a given URL.

        Args:
            url, str: URL to be downloaded.
        """
        import six.moves
        import tarfile

        raw_path = re.sub('\\\\', '/', os.path.join(self._root, 'raw'))
        processed_path = re.sub('\\\\', '/', os.path.join(self._root, 'processed').replace(r'\\', '/'))

        if not os.path.isdir(raw_path):
            os.mkdir(raw_path, mode=0o775)
        if not os.path.isdir(processed_path):
            os.mkdir(processed_path, mode=0x775)

        # Downloads file.
        fpath = re.sub('\\\\', '/', os.path.join(self._root, 'raw/CUB_200_2011.tgz'))

        '''
        try:
            print('Downloading ' + url + ' to ' + fpath)
            six.moves.urllib.request.urlretrieve(url, fpath)
        except six.moves.urllib.error.URLError:
            if url[:5] == 'https:':
                self._url = self._url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.')
                print('Downloading ' + url + ' to ' + fpath)
                six.moves.urllib.request.urlretrieve(url, fpath)
        '''

        # Extract file.
        cwd = os.getcwd()
        # print("menu: ", fpath)
        # tar = tarfile.open(fpath, 'r:gz')
        os.chdir(re.sub('\\\\', '/', os.path.join(self._root, 'raw')))
        # tar.extractall()
        # tar.close()
        os.chdir(cwd)
        print('Current working directory is ', os.getcwd())

    def _extract(self):
        """Prepare the data for train/test split and save onto disk."""
        # print('Start extracting...')
        image_path = re.sub('\\\\', '/', os.path.join(self._root, 'raw/CUB_200_2011/images/'))
        # Format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(re.sub('\\\\', '/', os.path.join(
            self._root, 'raw/CUB_200_2011/images.txt')), dtype=str)
        # Format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(re.sub('\\\\', '/', os.path.join(
            self._root, 'raw/CUB_200_2011/train_test_split.txt')), dtype=int)

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []
        for id_ in range(id2name.shape[0]):
            image = PIL.Image.open(re.sub('\\\\', '/', os.path.join(image_path, id2name[id_, 1])))
            label = int(id2name[id_, 1][:3]) - 1  # Label starts with 0

            # Convert gray scale image to RGB image.
            if image.getbands()[0] == 'L':
                image = image.convert('RGB')
            image_np = np.array(image)
            image.close()

            if id2train[id_, 1] == 1:
                train_data.append(image_np)
                train_labels.append(label)
            else:
                test_data.append(image_np)
                test_labels.append(label)

        pickle.dump((train_data, train_labels),
                    open(re.sub('\\\\', '/', os.path.join(self._root, 'processed/train.pkl')), 'wb'))
        pickle.dump((test_data, test_labels),
                    open(re.sub('\\\\', '/', os.path.join(self._root, 'processed/test.pkl')), 'wb'))
