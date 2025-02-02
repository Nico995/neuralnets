
import warnings

from neuralnets.util.tools import sample_unlabeled_input, sample_synchronized, normalize, set_seed
from neuralnets.util.augmentation import split_segmentation_transforms
from neuralnets.data.base import *

from skimage.measure import regionprops, label


MAX_SAMPLING_ATTEMPTS = 20


def _orient(data, orientation=0):
    """
    This function essentially places the desired orientation axis to that of the original Z-axis
    For example:
          (Z, Y, X) -> (Y, Z, X) for orientation=1
          (Z, Y, X) -> (X, Y, Z) for orientation=2
    Note that applying this function twice corresponds to the identity transform

    :param data: assumed to be of shape (Z, Y, X), nothing will be done for None types
    :param orientation: 0, 1 or 2 (respectively for Z, Y or X axis)
    :return: reoriented data sample
    """
    if data is not None:
        if orientation == 1:
            return np.transpose(data, axes=(1, 0, 2))
        elif orientation == 2:
            return np.transpose(data, axes=(2, 1, 0))
    return data


def _validate_shape(input_shape, data_shape, orientation=0, in_channels=1, levels=4):
    """
    Validates an input for propagation through a U-Net by taking into account the following:
        - Sampling along different orientations
        - Sampling multiple adjacent slices as channels
        - Maximum size that can be sampled from the data

    :param input_shape: original shape of the sample (Z, Y, X)
    :param data_shape: shape of the data to sample from (Z, Y, X)
    :param orientation: orientation to sample along
    :param in_channels: sample multiple adjacent slices as channels
    :param levels: amount of pooling layers in the network
    :return: the validated input shape
    """

    # make sure input shape can be edited
    input_shape = list(input_shape)

    # sample adjacent slices if necessary
    is2d = input_shape[0] == 1
    if is2d and in_channels > 1:
        input_shape[0] = in_channels

    # transform the data shape and input shape according to the orientation
    if orientation == 1:  # (Z, Y, X) -> (Y, Z, X)
        input_shape = [input_shape[1], input_shape[0], input_shape[2]]
    elif orientation == 2:  # (Z, Y, X) -> (X, Y, Z)
        input_shape = [input_shape[2], input_shape[1], input_shape[0]]

    # make sure the input shape fits in the data shape: i.e. find largest k such that n of the form n=k*2**levels
    for d in range(3):
        if not (is2d and d == orientation) and input_shape[d] > data_shape[d]:
            # 2D case: X, Y - 3D case: X, Y, Z
            # note we assume that the data has a least in_channels elements in each dimension
            input_shape[d] = int((data_shape[d] // (2 ** levels)) * (2 ** levels))

    # and return as a tuple
    return tuple(input_shape)


def _select_labels(y, frac=1.0, mode='random_slices', dim=0, seed=None):
    """
    Selects a fraction of the labels in a dataset.

    :param y: labels
    :param frac: fraction to be sampled
    :param mode: mode of the label selection
        - random_slices: randomly selects slices from the labels (default)
        - balanced: for each class a number of slices is selected so that the class frequency remains the same
    :param dim: dimension to select slices (if mode == 'random_slices')
    :param seed: seed for randomization
    :return: a subset of the labels
    """

    def _find_optimum(x, target, n, labels, dim):

        # compute amount of labeled pixels
        x_ = int(labels.shape[dim] * x)
        if dim == 0:
            v = np.sum(labels[:x_, :, :])
        elif dim == 1:
            v = np.sum(labels[:, :x_, :])
        else:
            v = np.sum(labels[:, :, :x_])

        # check if we can still progress
        if 2 ** (-n) < (1 / labels.shape[dim]):
            return int(labels.shape[dim] * x)

        # check if target value is smaller or higher
        if v < target:
            return _find_optimum(x + 2 ** (-n), target, n + 1, labels, dim)
        elif v > target:
            return _find_optimum(x - 2 ** (-n), target, n + 1, labels, dim)
        else:
            return int(labels.shape[dim] * x)

    def _select_balanced(y, frac):

        # get the frequency of all unique class labels
        u, cnts = np.unique(y, return_counts=True)
        # select the classes
        coi = u[u != 255]
        cnts = cnts[u != 255]
        # get the target counts
        target_cnts = (cnts * frac).astype(int)
        # initialize the new labels
        y_ = np.zeros_like(y) + 255
        for i, c in enumerate(coi):  # for each class of interest
            # find the optimal slice split index
            yc = np.asarray(y == c)
            r = _find_optimum(0.5, target_cnts[i], 2, yc, 0)
            # make sure it is at least 1
            r = max(1, r)
            # select the class pixels
            y_[:r][y[:r] == c] = c

        return y_

    def _select_random(y, frac, dim):

        # initialize
        y_ = np.zeros_like(y) + 255

        # find amount of slices
        sz = y.shape
        n = int(np.ceil(sz[dim] * frac))
        if n == 0:
            raise ValueError('A fraction of %f and a labeled volume of shape %s results in no slices' % (frac, str(sz)))

        # select slices
        np.random.seed(seed)
        ns = np.random.permutation(np.arange(sz[dim]))[:n]
        for d in range(sz[dim]):
            if d in ns:
                if dim == 0:
                    y_[d, :, :] = y[d, :, :]
                elif dim == 1:
                    y_[:, d, :] = y[:, d, :]
                else:
                    y_[:, :, d] = y[:, :, d]

        return y_

    # set seed for randomization
    if seed is not None:
        set_seed(seed)

    if frac == 0:
        return np.zeros_like(y) + 255
    elif frac == 1:
        return y

    # sample selection
    if mode == 'balanced':
        return _select_balanced(y, frac)
    else:
        return _select_random(y, frac, dim)


def _map_cois(y, coi):
    """
    Maps the classes of interest to consecutive label indices

    :param y: labels
    :param coi: classes of interest
    :return: reindexed labels
    """
    coi_ = list(coi)
    coi_.sort()
    y_ = np.zeros_like(y)
    y_[y == 255] = 255
    for i, c in enumerate(coi_):
        y_[y == c] = i

    return y_


def _unmap_cois(y, coi):
    """
    Unmaps the classes of interest to the original labels

    :return: original class labels
    """

    coi_ = list(coi)
    coi_.sort()
    y_ = np.zeros_like(y)
    y_[y == 255] = 255
    for i, c in enumerate(coi_):
        y_[y == i] = c

    return y_


def _label_stats(labels, coi):
    stats = []
    for j, labels_j in enumerate(labels):
        if labels_j is not None:
            tmp = np.zeros((len(coi)))
            for i, c in enumerate(coi):
                tmp[i] = np.sum(labels_j == i)
            tmp = tmp / np.sum(tmp)
            stats.append([])
            for i, c in enumerate(coi):
                stats[j].append((c, tmp[i]))
            stats[j].append((255, np.sum(labels_j == 255) / labels_j.size))
        else:
            stats.append(None)
    return stats


def _balance_weights(labels, type=None, label_stats=None):

    if type == 'inverse_class_balancing':
        weights = np.ones_like(labels, dtype=float)
        for i, ls in enumerate(label_stats):
            c, f = ls
            w_c = 1 / f
            weights[labels == i] = w_c
    elif type == 'inverse_size_balancing':
        weights = np.ones_like(labels, dtype=float)
        for i, ls in enumerate(label_stats):
            l_cc = label(labels == i)
            props = regionprops(l_cc)
            for k, p in enumerate(props):
                sz = p.area
                weights[l_cc == k] = np.sum(labels == i) / sz / len(props)
    else:
        weights = np.ones_like(labels, dtype=float)

    return weights


class LabeledStandardDataset(StandardDataset):
    """
    Strongly labeled dataset of N 2D images and pixel-wise labels

    :param data_path: path to the dataset
    :param label_path: path to the labels
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional data_dtype: type of the data (typically uint8)
    :param optional label_dtype: type of the labels (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    """

    def __init__(self, data_path, label_path, scaling=None, type='tif3d', data_dtype='uint8', label_dtype='uint8',
                 coi=(0, 1), norm_type='unit'):
        super().__init__(data_path, scaling=scaling, type=type, dtype=data_dtype, norm_type=norm_type)

        self.label_path = label_path
        self.coi = coi

        # load labels
        self.labels = read_volume(label_path, type=type, dtype=label_dtype)

        # rescale the dataset if necessary
        if scaling is not None:
            target_size = np.asarray(np.multiply(self.labels.shape, scaling), dtype=int)
            self.labels = \
                F.interpolate(torch.Tensor(self.labels[np.newaxis, np.newaxis, ...]), size=tuple(target_size),
                              mode='area')[0, 0, ...].numpy()

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # get random sample
        input = normalize(self.data[i], type=self.norm_type)
        target = self.labels[i]

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            input, target = input[np.newaxis, ...], target[np.newaxis, ...]

        if len(np.intersect1d(np.unique(target),
                              self.coi)) == 0:  # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
            return self.__getitem__(i)
        else:
            return input, target


class UnlabeledStandardDataset(StandardDataset):
    """
    Unlabeled dataset of N 2D images

    :param data_path: path to the dataset
    :param optional scaling: tuple used for rescaling the data, or None
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    """

    def __init__(self, data_path, scaling=None, type='tif3d', dtype='uint8', norm_type='unit'):
        super().__init__(data_path, scaling=scaling, type=type, dtype=dtype, norm_type=norm_type)

        self.mu, self.std = self._get_stats()

    def __getitem__(self, i):

        # get random sample
        input = normalize(self.data[i], type=self.norm_type)

        if input.shape[0] > 1:
            # add channel axis if the data is 3D
            return input[np.newaxis, ...]
        else:
            return input


class LabeledVolumeDataset(VolumeDataset):
    """
    Dataset for pixel-wise labeled volumes

    :param data: input data, possible formats
            - path to the dataset (string)
            - preloaded 3D volume (numpy array)
            - list of paths to multiple datasets (list of strings)
            - list of preloaded 3D volumes (list of numpy arrays)
    :param labels: path to the labels or a 3D volume that has already been loaded, possible formats:
            - path to the dataset (string)
            - preloaded 3D volume (numpy array)
            - list of paths to multiple datasets (list of strings)
            - list of preloaded 3D volumes (list of numpy arrays)
                  if None is provided within a list, it is assumed that these labels are not available
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: 3-tuple used for rescaling the data, or a list of 3-tuples in case of multiple datasets
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional coi: list or sequence of the classes of interest
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional data_dtype: type of the data (typically uint8)
    :param optional label_dtype: type of the labels (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional transform: augmenter object
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1), or a list of
                                 2-tuples in case of multiple datasets
    :param optional range_dir: orientation of the slicing or a list of orientations in case of multiple datasets
    :param optional resolution: list of 3-tuples specifying the pixel resolution of the data
    :param optional match_resolution_to: match the resolution of all data to a specific dataset
    :param optional sampling_type: type of sampling in case of multiple datasets
            - joint: the dataset will generate random samples in each dataset and return all of them
            - single: the dataset will generate a random sample from a randomly selected dataset and return that
    :param optional return_domain: return the domain id during iterating
    :param optional partial_labels: fraction of the labels that should be selected (default: 1)
    :param optional weight_balancing: balance classes, we currently support
            - inverse_class_balancing: class frequencies are balanced
            - inverse_size_balancing: object size is balanced
    :param optional seed: seed for consequent partial labeling
    """

    def __init__(self, data, labels, input_shape=None, scaling=None, len_epoch=None, type='tif3d', coi=(0, 1),
                 in_channels=1, orientations=(0,), batch_size=1, data_dtype='uint8', label_dtype='uint8',
                 norm_type='unit', transform=None, range_split=None, range_dir=None, resolution=None,
                 match_resolution_to=None, sampling_type='joint', return_domain=False, partial_labels=1,
                 weight_balancing=None, seed=None):
        super().__init__(data, input_shape, scaling=scaling, len_epoch=len_epoch, type=type,
                         in_channels=in_channels, orientations=orientations, batch_size=batch_size, dtype=data_dtype,
                         norm_type=norm_type, range_split=range_split, range_dir=range_dir, resolution=resolution,
                         match_resolution_to=match_resolution_to, sampling_type=sampling_type,
                         return_domain=return_domain)

        if isinstance(labels, str) or isinstance(labels, np.ndarray):
            self.labels = [load_data(labels, data_type=type, dtype=label_dtype)]
        elif isinstance(labels, list) or isinstance(labels, tuple):  # list of data
            self.labels = []
            all_none = True
            for labels_i in labels:
                if labels_i is None:
                    self.labels.append(None)
                else:
                    self.labels.append(load_data(labels_i, data_type=type, dtype=label_dtype))
                    all_none = False
            if all_none:
                raise ValueError('None was provided for all datasets, please use an UnlabeledVolumeDataset')
        elif labels is None:
            raise ValueError('None was provided for all datasets, please use an UnlabeledVolumeDataset')
        else:
            raise ValueError('LabeledVolumeDataset requires labels in str, np.ndarray or list format')
        self.coi = coi
        self.transform = transform
        if transform is not None:
            self.shared_transform, self.x_transform, self.y_transform = split_segmentation_transforms(transform)
        self.partial_labels = partial_labels
        self.weight_balancing = weight_balancing
        self.seed = seed

        # select a subset of slices of the data
        for i in range(len(self.labels)):
            if self.labels[i] is not None:
                if isinstance(range_dir, list) or isinstance(range_dir, tuple):
                    self.labels[i] = slice_subset(self.labels[i], range_split[i], range_dir[i])
                else:
                    self.labels[i] = slice_subset(self.labels[i], range_split, range_dir)

        # rescale the dataset if necessary
        for i in range(len(self.labels)):
            if (self.resolution is not None and self.match_resolution_to is not None) or self.scaling is not None \
                    and self.labels[i] is not None and self.labels[self.match_resolution_to] is not None:
                if self.resolution is not None and self.match_resolution_to is not None:
                    scale_factor = np.divide(self.labels[self.match_resolution_to].shape, self.labels[i].shape)
                else:
                    if isinstance(self.scaling, list) or isinstance(self.scaling, tuple):
                        scale_factor = self.scaling[i]
                    else:
                        scale_factor = self.scaling
                labels_it = torch.Tensor(self.labels[i]).unsqueeze(0).unsqueeze(0)
                self.labels[i] = F.interpolate(labels_it, scale_factor=scale_factor, mode='bilinear')[0, 0, ...].numpy()

        # select a subset of the labels if necessary
        for i in range(len(self.labels)):
            if self.labels[i] is not None:
                if isinstance(self.partial_labels, list) or isinstance(self.partial_labels, tuple):
                    self.labels[i] = _select_labels(self.labels[i], frac=self.partial_labels[i], seed=seed)
                else:
                    self.labels[i] = _select_labels(self.labels[i], frac=self.partial_labels, seed=seed)

        # relabel classes of interest
        self.labels = [_map_cois(l, self.coi) if l is not None else None for l in self.labels]

        # print label stats
        self.label_stats = _label_stats(self.labels, self.coi)

        # setup weight balancing if necessary
        if weight_balancing is not None:
            print_frm('Precomputing balancing weights...')
            self.weights = [_balance_weights(l, type=self.weight_balancing, label_stats=self.label_stats[i][:-1])
                            if l is not None else None for i, l in enumerate(self.labels)]

    def _sample_xy(self, data_index):

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data[data_index].shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        x, y = sample_synchronized([self.data[data_index], self.labels[data_index]], input_shape)
        x = normalize(x, type=self.norm_type)
        if y is not None:
            y = y.astype(float)

        # reorient sample
        x = _orient(x, orientation=self.orientation)
        y = _orient(y, orientation=self.orientation)

        # add channel axis if the data is 3D
        if self.input_shape[0] > 1:
            x = x[np.newaxis, ...]
            if y is not None:
                y = y[np.newaxis, ...]

        # select middle slice if multiple consecutive slices
        if self.in_channels > 1:
            c = self.in_channels // 2
            if y is not None:
                y = y[c:c + 1]

        # augment sample
        if self.transform is not None:
            if y is not None:
                data = self.shared_transform(np.concatenate((x, y), axis=0))
                p = x.shape[0]
                x = self.x_transform(data[:p])
                y = self.y_transform(data[p:])
            else:
                x = self.shared_transform(x)
                x = self.x_transform(x)

        # transform to tensors
        x = torch.from_numpy(x).float()
        if y is not None:
            y = torch.from_numpy(y).long()

        return x, y

    def _sample_xyw(self, data_index):

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data[data_index].shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        x, y, w = sample_synchronized([self.data[data_index], self.labels[data_index], self.weights[data_index]],
                                      input_shape)
        x = normalize(x, type=self.norm_type)
        if y is not None:
            y = y.astype(float)
            w = w.astype(float)

        # reorient sample
        x = _orient(x, orientation=self.orientation)
        y = _orient(y, orientation=self.orientation)
        w = _orient(w, orientation=self.orientation)

        # add channel axis if the data is 3D
        if self.input_shape[0] > 1:
            x = x[np.newaxis, ...]
            if y is not None:
                y, w = y[np.newaxis, ...], w[np.newaxis, ...]

        # select middle slice if multiple consecutive slices
        if self.in_channels > 1:
            c = self.in_channels // 2
            if y is not None:
                y, w = y[c:c + 1], w[c:c + 1]

        # augment sample
        if self.transform is not None:
            if y is not None:
                data = self.shared_transform(np.concatenate((x, y, w), axis=0))
                p = x.shape[0]
                q = y.shape[0]
                x = self.x_transform(data[:p])
                y = self.y_transform(data[p:p+q])
                w = data[p+q:]
            else:
                x = self.shared_transform(x)
                x = self.x_transform(x)

        # transform to tensors
        x = torch.from_numpy(x).float()
        if y is not None:
            y = torch.from_numpy(y).long()
            w = torch.from_numpy(w).float()

        return x, y, w

    def __getitem__(self, i, attempt=0):

        # reorient when we start a new batch
        if i % self.batch_size == 0:
            self._select_orientation()

        if self.sampling_type == 'single':

            # randomly select a dataset
            r = np.random.randint(len(self.data))

            # select a sample from dataset r
            if self.weight_balancing is not None:
                x, y, w = self._sample_xyw(r)
            else:
                x, y = self._sample_xy(r)

            # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
            if len(np.intersect1d(torch.unique(y).numpy(), self.coi)) == 0 and not self.warned:
                if attempt < MAX_SAMPLING_ATTEMPTS:
                    if self.weight_balancing is not None:
                        x, y, w = self.__getitem__(i, attempt=attempt + 1)
                    else:
                        x, y = self.__getitem__(i, attempt=attempt + 1)
                else:
                    warnings.warn("No labeled pixels found after %d sampling attempts! " % attempt)
                    self.warned = True

        else:  # joint sampling

            xs = []
            ys = []
            if self.weight_balancing is not None:
                ws = []

            for r in range(len(self.data)):

                # select a sample from dataset r
                if self.weight_balancing is not None:
                    x, y, w = self._sample_xyw(r)
                else:
                    x, y = self._sample_xy(r)

                xs.append(x)
                ys.append(y)
                if self.weight_balancing is not None:
                    ws.append(w)

            if len(self.data) == 1:
                x = xs[0]
                y = ys[0]
                if self.weight_balancing is not None:
                    w = ws[0]
            else:
                x = xs
                y = ys
                if self.weight_balancing is not None:
                    w = ws

            r = np.arange(len(self.data), dtype=int)

            # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
            if np.sum([len(np.intersect1d(torch.unique(y_).numpy(), self.coi)) if y_ is not None else 1 for y_ in ys]) == 0 and not self.warned:
                if attempt < MAX_SAMPLING_ATTEMPTS:
                    if self.weight_balancing is not None:
                        x, y, w = self.__getitem__(i, attempt=attempt + 1)
                    else:
                        x, y = self.__getitem__(i, attempt=attempt + 1)
                else:
                    warnings.warn("No labeled pixels found after %d sampling attempts! " % attempt)
                    self.warned = True

        # dataloader does not support None types, send empty array instead
        for j in range(len(y)):
            if y[j] is None:
                y[j] = np.zeros(0)
                if self.weight_balancing is not None:
                    w[j] = np.zeros(0)

        # return sample
        if self.return_domain:
            if self.weight_balancing is not None:
                return r, x, y, w
            else:
                return r, x, y
        else:
            if self.weight_balancing is not None:
                return x, y, w
            else:
                return x, y

    def get_original_labels(self):
        """
        Unmaps the classes of interest to the original labels

        :return: original class labels
        """

        return [_unmap_cois(labels, self.coi) for labels in self.labels]


class UnlabeledVolumeDataset(VolumeDataset):
    """
    Dataset for unlabeled volumes

    :param data: input data, possible formats
            - path to the dataset (string)
            - preloaded 3D volume (numpy array)
            - list of paths to multiple datasets (list of strings)
            - list of preloaded 3D volumes (list of numpy arrays)
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: 3-tuple used for rescaling the data, or a list of 3-tuples in case of multiple datasets
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional transform: augmenter object
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1), or a list of
                                 2-tuples in case of multiple datasets
    :param optional range_dir: orientation of the slicing or a list of orientations in case of multiple datasets
    :param optional resolution: list of 3-tuples specifying the pixel resolution of the data
    :param optional match_resolution_to: match the resolution of all data to a specific dataset
    :param optional sampling_type: type of sampling in case of multiple datasets
            - joint: the dataset will generate random samples in each dataset and return all of them
            - single: the dataset will generate a random sample from a randomly selected dataset and return that
    """

    def __init__(self, data, input_shape=None, scaling=None, len_epoch=None, type='tif3d', in_channels=1,
                 orientations=(0,), batch_size=1, dtype='uint8', norm_type='unit', transform=None, range_split=None,
                 range_dir=None, resolution=None, match_resolution_to=None, sampling_type='joint'):
        super().__init__(data, input_shape, scaling=scaling, len_epoch=len_epoch, type=type,
                         in_channels=in_channels, orientations=orientations, batch_size=batch_size, dtype=dtype,
                         norm_type=norm_type, range_split=range_split, range_dir=range_dir, resolution=resolution,
                         match_resolution_to=match_resolution_to, sampling_type=sampling_type)

        self.transform = transform

    def _sample(self, data_index):

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data[data_index].shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        x = sample_unlabeled_input(self.data[data_index], input_shape)
        x = normalize(x, type=self.norm_type)

        # reorient sample
        x = _orient(x, orientation=self.orientation)

        # add channel axis if the data is 3D
        if self.input_shape[0] > 1:
            x = x[np.newaxis, ...]

        # augment sample
        if self.transform is not None:
            x = self.transform(x)

        # transform to tensors
        x = torch.from_numpy(x).float()

        return x

    def __getitem__(self, i):

        # reorient when we start a new batch
        if i % self.batch_size == 0:
            self._select_orientation()

        if self.sampling_type == 'single':

            # randomly select a dataset
            r = np.random.randint(len(self.data))

            # select a sample from dataset r
            x = self._sample(r)

        else:  # joint sampling

            xs = []

            for r in range(len(self.data)):

                # select a sample from dataset r
                x = self._sample(r)

                xs.append(x)

            if len(self.data) == 1:
                x = xs[0]
            else:
                x = xs

        # return sample
        return x


class LabeledSlidingWindowDataset(SlidingWindowDataset):
    """
    Dataset for pixel-wise labeled volumes with a sliding window

    :param data: input data, possible formats
            - path to the dataset (string)
            - preloaded 3D volume (numpy array)
            - list of paths to multiple datasets (list of strings)
            - list of preloaded 3D volumes (list of numpy arrays)
    :param labels: path to the labels or a 3D volume that has already been loaded, possible formats:
            - path to the dataset (string)
            - preloaded 3D volume (numpy array)
            - list of paths to multiple datasets (list of strings)
            - list of preloaded 3D volumes (list of numpy arrays)
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: 3-tuple used for rescaling the data, or a list of 3-tuples in case of multiple datasets
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional coi: list or sequence of the classes of interest
    :param optional batch_size: size of the sampling batch
    :param optional data_dtype: type of the data (typically uint8)
    :param optional label_dtype: type of the labels (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional transform: augmenter object
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1), or a list of
                                 2-tuples in case of multiple datasets
    :param optional range_dir: orientation of the slicing or a list of orientations in case of multiple datasets
    :param optional resolution: list of 3-tuples specifying the pixel resolution of the data
    :param optional match_resolution_to: match the resolution of all data to a specific dataset
    :param optional return_domain: return the domain id during iterating
    :param optional partial_labels: fraction of the labels that should be selected (default: 1)
    :param optional weight_balancing: balance classes, we currently support
            - inverse_class_balancing: class frequencies are balanced
            - inverse_size_balancing: object size is balanced
    """

    def __init__(self, data, labels, input_shape=None, scaling=None, type='tif3d', in_channels=1, orientations=(0,),
                 coi=(0, 1), batch_size=1, data_dtype='uint8', label_dtype='uint8', norm_type='unit', transform=None,
                 range_split=None, range_dir=None, resolution=None, match_resolution_to=None, return_domain=False,
                 partial_labels=1, weight_balancing=None):
        super().__init__(data, input_shape, scaling=scaling, type=type, in_channels=in_channels,
                         orientations=orientations, batch_size=batch_size, dtype=data_dtype, norm_type=norm_type,
                         range_split=range_split, range_dir=range_dir, resolution=resolution,
                         match_resolution_to=match_resolution_to, return_domain=return_domain)

        if isinstance(labels, str) or isinstance(labels, np.ndarray):
            self.labels = [load_data(labels, data_type=type, dtype=label_dtype)]
        elif isinstance(labels, list) or isinstance(labels, tuple):  # list of data
            self.labels = []
            for labels_i in labels:
                self.labels.append(load_data(labels_i, data_type=type, dtype=label_dtype))
        else:
            raise ValueError('LabeledSlidingWindowDataset requires labels in str, np.ndarray or list format')
        self.coi = coi
        self.transform = transform
        if transform is not None:
            self.shared_transform, self.x_transform, self.y_transform = split_segmentation_transforms(transform)
        self.partial_labels = partial_labels
        self.weight_balancing = weight_balancing

        # select a subset of slices of the data
        for i in range(len(self.labels)):
            if isinstance(range_dir, list) or isinstance(range_dir, tuple):
                self.labels[i] = slice_subset(self.labels[i], range_split[i], range_dir[i])
            else:
                self.labels[i] = slice_subset(self.labels[i], range_split, range_dir)

        # rescale the dataset if necessary
        for i in range(len(self.labels)):
            if (self.resolution is not None and self.match_resolution_to is not None) or self.scaling is not None:
                if self.resolution is not None and self.match_resolution_to is not None:
                    scale_factor = np.divide(self.labels[self.match_resolution_to].shape, self.labels[i].shape)
                else:
                    if isinstance(self.scaling, list) or isinstance(self.scaling, tuple):
                        scale_factor = self.scaling[i]
                    else:
                        scale_factor = self.scaling
                labels_it = torch.Tensor(self.labels[i]).unsqueeze(0).unsqueeze(0)
                self.labels[i] = F.interpolate(labels_it, scale_factor=scale_factor, mode='bilinear')[0, 0, ...].numpy()

        # select a subset of the labels if necessary
        for i in range(len(self.labels)):
            if isinstance(self.partial_labels, list) or isinstance(self.partial_labels, tuple):
                self.labels[i] = _select_labels(self.labels[i], frac=self.partial_labels[i])
            else:
                self.labels[i] = _select_labels(self.labels[i], frac=self.partial_labels)

        # pad data so that the dimensions are a multiple of the inputs shapes
        self.labels = [pad2multiple(l, input_shape, value=255) for l in self.labels]

        # pad data so that additional channels can be sampled
        self.labels = [pad_channels(l, in_channels=in_channels, orientations=self.orientations) for l in self.labels]

        # relabel classes of interest
        self.labels = [_map_cois(l, self.coi) for l in self.labels]

        # print label stats
        self.label_stats = _label_stats(self.labels, self.coi)

        # setup weight balancing if necessary
        if weight_balancing is not None:
            print_frm('Precomputing balancing weights...')
            self.weights = [_balance_weights(l, type=self.weight_balancing, label_stats=self.label_stats[i][:-1])
                            for i, l in enumerate(self.labels)]

    def _sample_xyw(self, i):

        # find dataset index
        r = 0
        szs = self.n_samples_dim.prod(axis=1)
        while szs[:r + 1].sum() <= i:
            r += 1

        # get spatial location
        j = i - szs[:r].sum()
        iz = j // (self.n_samples_dim[r, 1] * self.n_samples_dim[r, 2])
        iy = (j - iz * self.n_samples_dim[r, 1] * self.n_samples_dim[r, 2]) // self.n_samples_dim[r, 2]
        ix = j - iz * self.n_samples_dim[r, 1] * self.n_samples_dim[r, 2] - iy * self.n_samples_dim[r, 2]
        pz = self.input_shape[0] * iz
        py = self.input_shape[1] * iy
        px = self.input_shape[2] * ix

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data[r].shape, in_channels=self.in_channels)

        # get sample
        x, y, w = sample_synchronized([self.data[r], self.labels[r], self.weights[r]], input_shape, zyx=(pz, py, px))
        x = normalize(x, type=self.norm_type)
        y = y.astype(float)
        w = w.astype(float)

        # add channel axis if the data is 3D
        if self.input_shape[0] > 1:
            x, y, w = x[np.newaxis, ...], y[np.newaxis, ...], w[np.newaxis, ...]

        # select middle slice if multiple consecutive slices
        if self.in_channels > 1:
            c = self.in_channels // 2
            y = y[c:c + 1]
            w = w[c:c + 1]

        # augment sample
        if self.transform is not None:
            data = self.shared_transform(np.concatenate((x, y, w), axis=0))
            p = x.shape[0]
            q = y.shape[0]
            x = self.x_transform(data[:p])
            y = self.y_transform(data[p:p+q])
            w = w[p+q:]

        # transform to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()
        w = torch.from_numpy(w).float()

        # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
        if len(np.intersect1d(torch.unique(y).numpy(), np.arange(len(self.coi)))) == 0 and not self.warned:
            warnings.warn("No labeled pixels found! ")
            self.warned = True

        return r, x, y, w

    def _sample_xy(self, i):

        # find dataset index
        r = 0
        szs = self.n_samples_dim.prod(axis=1)
        while szs[:r + 1].sum() <= i:
            r += 1

        # get spatial location
        j = i - szs[:r].sum()
        iz = j // (self.n_samples_dim[r, 1] * self.n_samples_dim[r, 2])
        iy = (j - iz * self.n_samples_dim[r, 1] * self.n_samples_dim[r, 2]) // self.n_samples_dim[r, 2]
        ix = j - iz * self.n_samples_dim[r, 1] * self.n_samples_dim[r, 2] - iy * self.n_samples_dim[r, 2]
        pz = self.input_shape[0] * iz
        py = self.input_shape[1] * iy
        px = self.input_shape[2] * ix

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data[r].shape, in_channels=self.in_channels)

        # get sample
        x, y = sample_synchronized([self.data[r], self.labels[r]], input_shape, zyx=(pz, py, px))
        x = normalize(x, type=self.norm_type)
        y = y.astype(float)

        # add channel axis if the data is 3D
        if self.input_shape[0] > 1:
            x, y = x[np.newaxis, ...], y[np.newaxis, ...]

        # select middle slice if multiple consecutive slices
        if self.in_channels > 1:
            c = self.in_channels // 2
            y = y[c:c + 1]

        # augment sample
        if self.transform is not None:
            data = self.shared_transform(np.concatenate((x, y), axis=0))
            p = x.shape[0]
            x = self.x_transform(data[:p])
            y = self.y_transform(data[p:])

        # transform to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()

        # make sure we have at least one labeled pixel in the sample, otherwise processing is useless
        if len(np.intersect1d(torch.unique(y).numpy(), np.arange(len(self.coi)))) == 0 and not self.warned:
            warnings.warn("No labeled pixels found! ")
            self.warned = True

        return r, x, y

    def __getitem__(self, i):

        if self.weight_balancing is not None:
            r, x, y, w = self._sample_xyw(i)
            if self.return_domain:
                return r, x, y, w
            else:
                return x, y, w
        else:
            r, x, y = self._sample_xy(i)
            if self.return_domain:
                return r, x, y
            else:
                return x, y

    def get_original_labels(self):
        """
        Unmaps the classes of interest to the original labels

        :return: original class labels
        """

        return [_unmap_cois(labels, self.coi) for labels in self.labels]


class UnlabeledSlidingWindowDataset(SlidingWindowDataset):
    """
    Dataset for unlabeled volumes with a sliding window

    :param data: input data, possible formats
            - path to the dataset (string)
            - preloaded 3D volume (numpy array)
            - list of paths to multiple datasets (list of strings)
            - list of preloaded 3D volumes (list of numpy arrays)
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional scaling: 3-tuple used for rescaling the data, or a list of 3-tuples in case of multiple datasets
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional orientations: list of orientations for sampling
    :param optional batch_size: size of the sampling batch
    :param optional data_dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional transform: augmenter object
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1), or a list of
                                 2-tuples in case of multiple datasets
    :param optional range_dir: orientation of the slicing or a list of orientations in case of multiple datasets
    :param optional resolution: list of 3-tuples specifying the pixel resolution of the data
    :param optional match_resolution_to: match the resolution of all data to a specific dataset
    :param optional return_domain: return the domain id during iterating
    """

    def __init__(self, data, input_shape=None, scaling=None, type='tif3d', in_channels=1, orientations=(0,),
                 batch_size=1, data_dtype='uint8', norm_type='unit', transform=None, range_split=None, range_dir=None,
                 resolution=None, match_resolution_to=None, return_domain=False):
        super().__init__(data, input_shape, scaling=scaling, type=type, in_channels=in_channels,
                         orientations=orientations, batch_size=batch_size, dtype=data_dtype, norm_type=norm_type,
                         range_split=range_split, range_dir=range_dir, resolution=resolution,
                         match_resolution_to=match_resolution_to, return_domain=return_domain)

        self.transform = transform
        if transform is not None:
            self.shared_transform, self.x_transform, _ = split_segmentation_transforms(transform)
        self.weight_balancing = None

    def _sample_x(self, i):

        # find dataset index
        r = 0
        szs = self.n_samples_dim.prod(axis=1)
        while szs[:r + 1].sum() <= i:
            r += 1

        # get spatial location
        j = i - szs[:r].sum()
        iz = j // (self.n_samples_dim[r, 1] * self.n_samples_dim[r, 2])
        iy = (j - iz * self.n_samples_dim[r, 1] * self.n_samples_dim[r, 2]) // self.n_samples_dim[r, 2]
        ix = j - iz * self.n_samples_dim[r, 1] * self.n_samples_dim[r, 2] - iy * self.n_samples_dim[r, 2]
        pz = self.input_shape[0] * iz
        py = self.input_shape[1] * iy
        px = self.input_shape[2] * ix

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data[r].shape, in_channels=self.in_channels)

        # get sample
        x = sample_synchronized([self.data[r]], input_shape, zyx=(pz, py, px))[0]
        x = normalize(x, type=self.norm_type)

        # add channel axis if the data is 3D
        if self.input_shape[0] > 1:
            x = x[np.newaxis, ...]

        # augment sample
        if self.transform is not None:
            x = self.shared_transform(x)
            x = self.x_transform(x)

        # transform to tensors
        x = torch.from_numpy(x).float()

        return r, x

    def __getitem__(self, i):

        r, x = self._sample_x(i)
        if self.return_domain:
            return r, x
        else:
            return x
