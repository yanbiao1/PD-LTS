
import random
import torch
import numbers
import math



class AddRandomNoise(object):

    def __init__(self, std_range=[0.0, 0.10], noiseless_item_key='clean'):
        super(AddRandomNoise, self).__init__()

        self.std_range = std_range
        self.key = noiseless_item_key
    
    def __call__(self, data):
        noise_std = random.uniform(*self.std_range)
        data[self.key] = data['pos']
        data['pos'] = data['pos'] + torch.normal(0, noise_std, size=data['pos'].shape)
        return data

class AddNoise(object):

    def __init__(self, std=0.01, noiseless_item_key='clean'):
        super(AddNoise, self).__init__()

        self.std = std
        self.key = noiseless_item_key
    
    def __call__(self, data):
        data[self.key] = data['pos']
        data['pos'] = data['pos'] + torch.normal(0, self.std, size=data['pos'].shape)
        return data

class AddNoiseForEval(object):

    def __init__(self, stds=[0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15]):
        super(AddNoiseForEval, self).__init__()

        self.stds = stds
        self.keys = ['noisy_%.2f' % s for s in stds]
    
    def __call__(self, data):
        data['clean'] = data['pos']
        for noise_std in self.stds:
            data['noisy_%.2f' % noise_std] = data['pos'] + torch.normal(0, noise_std, size=data['pos'].shape)
        return data


class RandomScale(object):
    r"""Scales node positions by a randomly sampled factor :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix

    .. math::
        \begin{bmatrix}
            s & 0 & 0 \\
            0 & s & 0 \\
            0 & 0 & s \\
        \end{bmatrix}

    for three-dimensional positions.

    Args:
        scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
    """

    def __init__(self, scales, attr):
        super(RandomScale, self).__init__()

        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales
        self.attr = attr

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        for key in self.attr:
            data[key] = data[key] * scale
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.scales})'


class IdentityTransform(object):

    def __call__(self, data):
        return data


class LinearTransformation(object):
    r"""Transforms node positions with a square transformation matrix computed
    offline.

    Args:
        matrix (Tensor): tensor with shape :math:`[D, D]` where :math:`D`
            corresponds to the dimensionality of node positions.
    """

    def __init__(self, matrix, attr):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        self.matrix = matrix
        self.attr = attr

    def __call__(self, data):
        for key in self.attr:
            pos = data[key].view(-1, 1) if data[key].dim() == 1 else data[key]

            assert pos.size(-1) == self.matrix.size(-2), (
                'Node position matrix and transformation matrix have incompatible '
                'shape.')

            data[key] = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))

        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.matrix.tolist())


class RandomRotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.

    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degrees, attr, axis=0):
        super(RandomRotate, self).__init__()

        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        
        self.degrees = degrees
        self.axis = axis
        self.attr = attr
    
    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = torch.tensor([
                [1.0,  0.0, 0.0],
                [0.0,  cos, sin],
                [0.0, -sin, cos],
            ], dtype=torch.float32)
        elif self.axis == 1:
            matrix = torch.tensor([
                [cos, 0.0, -sin],
                [0.0, 1.0,  0.0],
                [sin, 0.0,  cos],
            ], dtype=torch.float32)
        else:
            matrix = torch.tensor([
                [ cos, sin, 0.0],
                [-sin, cos, 0.0],
                [ 0.0, 0.0, 1.0],
            ], dtype=torch.float32)
        
        return LinearTransformation(matrix, attr=self.attr)(data)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.degrees}, axis={self.axis})'
