# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
from mmcv.transforms import BaseTransform, KeyMapper
from mmengine.dataset import Compose
from packaging import version as pv
from scipy.stats import mode
from torch.nn.modules.utils import _pair

from mmaction.registry import TRANSFORMS
# from .loading import DecordDecode, DecordInit
# from .processing import _combine_quadruple

if pv.parse(scipy.__version__) < pv.parse('1.11.0'):
    get_mode = mode
else:
    from functools import partial
    get_mode = partial(mode, keepdims=True)


@TRANSFORMS.register_module()
class PoseCompact(BaseTransform):
    

    def __init__(self,
                 padding: float = 0.25,
                 threshold: int = 10,
                 hw_ratio: Optional[Union[float, Tuple[float]]] = None,
                 allow_imgpad: bool = True) -> None:

        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = _pair(hw_ratio)

        self.hw_ratio = hw_ratio

        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    def transform(self, results: Dict) -> Dict:
        """Convert the coordinates of keypoints to make it more compact.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']

        # Make NaN zero
        kp[np.isnan(kp)] = 0.
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return results

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)

        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape

        # the order is x, y, w, h (in [0, 1]), a tuple
        crop_quadruple = results.get('crop_quadruple', (0., 0., 1., 1.))
        new_crop_quadruple = (min_x / w, min_y / h, (max_x - min_x) / w,
                              (max_y - min_y) / h)
        crop_quadruple = _combine_quadruple(crop_quadruple, new_crop_quadruple)
        results['crop_quadruple'] = crop_quadruple
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}(padding={self.padding}, '
                    f'threshold={self.threshold}, '
                    f'hw_ratio={self.hw_ratio}, '
                    f'allow_imgpad={self.allow_imgpad})')
        return repr_str


@TRANSFORMS.register_module()
class PreNormalize2D(BaseTransform):
    """Normalize the range of keypoint values.

    Required Keys:

        - keypoint
        - img_shape (optional)

    Modified Keys:

        - keypoint

    Args:
        img_shape (tuple[int, int]): The resolution of the original video.
            Defaults to ``(1080, 1920)``.
    """

    def __init__(self, img_shape: Tuple[int, int] = (1080, 1920)) -> None:
        self.img_shape = img_shape

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PreNormalize2D`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        h, w = results.get('img_shape', self.img_shape)
        results['keypoint'][..., 0] = \
            (results['keypoint'][..., 0] - (w / 2)) / (w / 2)
        results['keypoint'][..., 1] = \
            (results['keypoint'][..., 1] - (h / 2)) / (h / 2)
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'img_shape={self.img_shape})')
        return repr_str


@TRANSFORMS.register_module()
class JointToBone(BaseTransform):
    """Convert the joint information to bone information.

    Required Keys:

        - keypoint

    Modified Keys:

        - keypoint

    Args:
        dataset (str): Define the type of dataset: 'nturgb+d', 'openpose',
            'coco'. Defaults to ``'nturgb+d'``.
        target (str): The target key for the bone information.
            Defaults to ``'keypoint'``.
    """

    def __init__(self,
                 dataset: str = 'nturgb+d',
                 target: str = 'keypoint') -> None:
        self.dataset = dataset
        self.target = target
        if self.dataset not in ['nturgb+d', 'openpose', 'coco']:
            raise ValueError(
                f'The dataset type {self.dataset} is not supported')
        if self.dataset == 'nturgb+d':
            self.pairs = [(0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4),
                          (6, 5), (7, 6), (8, 20), (9, 8), (10, 9), (11, 10),
                          (12, 0), (13, 12), (14, 13), (15, 14), (16, 0),
                          (17, 16), (18, 17), (19, 18), (21, 22), (20, 20),
                          (22, 7), (23, 24), (24, 11)]
        elif self.dataset == 'openpose':
            self.pairs = ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1),
                          (6, 5), (7, 6), (8, 2), (9, 8), (10, 9), (11, 5),
                          (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17,
                                                                           15))
        elif self.dataset == 'coco':
            self.pairs = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0),
                          (6, 0), (7, 5), (8, 6), (9, 7), (10, 8), (11, 0),
                          (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`JointToBone`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        keypoint = results['keypoint']
        M, T, V, C = keypoint.shape
        bone = np.zeros((M, T, V, C), dtype=np.float32)

        assert C in [2, 3]
        for v1, v2 in self.pairs:
            bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
            if C == 3 and self.dataset in ['openpose', 'coco']:
                score = (keypoint[..., v1, 2] + keypoint[..., v2, 2]) / 2
                bone[..., v1, 2] = score

        results[self.target] = bone
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'dataset={self.dataset}, '
                    f'target={self.target})')
        return repr_str


@TRANSFORMS.register_module()
class ToMotion(BaseTransform):
    """Convert the joint information or bone information to corresponding
    motion information.

    Required Keys:

        - keypoint

    Added Keys:

        - motion

    Args:
        dataset (str): Define the type of dataset: 'nturgb+d', 'openpose',
            'coco'. Defaults to ``'nturgb+d'``.
        source (str): The source key for the joint or bone information.
            Defaults to ``'keypoint'``.
        target (str): The target key for the motion information.
            Defaults to ``'motion'``.
    """

    def __init__(self,
                 dataset: str = 'nturgb+d',
                 source: str = 'keypoint',
                 target: str = 'motion') -> None:
        self.dataset = dataset
        self.source = source
        self.target = target

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`ToMotion`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        data = results[self.source]
        M, T, V, C = data.shape
        motion = np.zeros_like(data)

        assert C in [2, 3]
        motion[:, :T - 1] = np.diff(data, axis=1)
        if C == 3 and self.dataset in ['openpose', 'coco']:
            score = (data[:, :T - 1, :, 2] + data[:, 1:, :, 2]) / 2
            motion[:, :T - 1, :, 2] = score

        results[self.target] = motion

        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'dataset={self.dataset}, '
                    f'source={self.source}, '
                    f'target={self.target})')
        return repr_str


@TRANSFORMS.register_module()
class MergeSkeFeat(BaseTransform):
    """Merge multi-stream features.

    Args:
        feat_list (list[str]): The list of the keys of features.
            Defaults to ``['keypoint']``.
        target (str): The target key for the merged multi-stream information.
            Defaults to ``'keypoint'``.
        axis (int): The axis along which the features will be joined.
            Defaults to -1.
    """

    def __init__(self,
                 feat_list: List[str] = ['keypoint'],
                 target: str = 'keypoint',
                 axis: int = -1) -> None:
        self.feat_list = feat_list
        self.target = target
        self.axis = axis

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`MergeSkeFeat`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        feats = []
        for name in self.feat_list:
            feats.append(results.pop(name))
        feats = np.concatenate(feats, axis=self.axis)
        results[self.target] = feats
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'feat_list={self.feat_list}, '
                    f'target={self.target}, '
                    f'axis={self.axis})')
        return repr_str


@TRANSFORMS.register_module()
class GenSkeFeat(BaseTransform):
    """Unified interface for generating multi-stream skeleton features.

    Required Keys:

        - keypoint
        - keypoint_score (optional)

    Args:
        dataset (str): Define the type of dataset: 'nturgb+d', 'openpose',
            'coco'. Defaults to ``'nturgb+d'``.
        feats (list[str]): The list of the keys of features.
            Defaults to ``['j']``.
        axis (int): The axis along which the features will be joined.
            Defaults to -1.
    """

    def __init__(self,
                 dataset: str = 'nturgb+d',
                 feats: List[str] = ['j'],
                 axis: int = -1) -> None:
        self.dataset = dataset
        self.feats = feats
        self.axis = axis
        ops = []
        if 'b' in feats or 'bm' in feats:
            ops.append(JointToBone(dataset=dataset, target='b'))
        ops.append(KeyMapper(remapping={'keypoint': 'j'}))
        if 'jm' in feats:
            ops.append(ToMotion(dataset=dataset, source='j', target='jm'))
        if 'bm' in feats:
            ops.append(ToMotion(dataset=dataset, source='b', target='bm'))
        ops.append(MergeSkeFeat(feat_list=feats, axis=axis))
        self.ops = Compose(ops)

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`GenSkeFeat`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if 'keypoint_score' in results and 'keypoint' in results:
            assert self.dataset != 'nturgb+d'
            assert results['keypoint'].shape[
                -1] == 2, 'Only 2D keypoints have keypoint_score. '
            keypoint = results.pop('keypoint')
            keypoint_score = results.pop('keypoint_score')
            results['keypoint'] = np.concatenate(
                [keypoint, keypoint_score[..., None]], -1)
        return self.ops(results)

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'dataset={self.dataset}, '
                    f'feats={self.feats}, '
                    f'axis={self.axis})')
        return repr_str


@TRANSFORMS.register_module()
class UniformSampleFrames(BaseTransform):
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required Keys:

        - total_frames
        - start_index (optional)

    Added Keys:

        - frame_inds
        - frame_interval
        - num_clips
        - clip_len

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Defaults to 1.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        seed (int): The random seed used during test time. Defaults to 255.
    """

    def __init__(self,
                 clip_len: int,
                 num_clips: int = 1,
                 test_mode: bool = False,
                 seed: int = 255) -> None:
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.seed = seed

    def _get_train_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.

        Returns:
            np.ndarray: The sampled indices for training clips.
        """
        all_inds = []
        for clip_idx in range(self.num_clips):
            if num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int32)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds)

        return np.concatenate(all_inds)

    def _get_test_clips(self, num_frames: int, clip_len: int) -> np.ndarray:
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.

        Returns:
            np.ndarray: The sampled indices for testing clips.
        """

        np.random.seed(self.seed)
        all_inds = []
        for i in range(self.num_clips):
            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips \
                    else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds)

        return np.concatenate(all_inds)

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`UniformSampleFrames`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        num_frames = results['total_frames']

        if self.test_mode:
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results.get('start_index', 0)
        inds = inds + start_index

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(np.int64)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results['frame_inds'] = inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'test_mode={self.test_mode}, '
                    f'seed={self.seed})')
        return repr_str



@TRANSFORMS.register_module()
class PoseDecode(BaseTransform):
    """Load and decode pose with given indices.

    Required Keys:

        - keypoint
        - total_frames (optional)
        - frame_inds (optional)
        - offset (optional)
        - keypoint_score (optional)

    Modified Keys:

        - keypoint
        - keypoint_score (optional)
    """

    @staticmethod
    def _load_kp(kp: np.ndarray, frame_inds: np.ndarray) -> np.ndarray:
        """Load keypoints according to sampled indexes."""
        return kp[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_kpscore(kpscore: np.ndarray,
                      frame_inds: np.ndarray) -> np.ndarray:
        """Load keypoint scores according to sampled indexes."""
        return kpscore[:, frame_inds].astype(np.float32)

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`PoseDecode`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if 'total_frames' not in results:
            results['total_frames'] = results['keypoint'].shape[1]

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            results['keypoint_score'] = self._load_kpscore(
                results['keypoint_score'], frame_inds)

        results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)

        return results

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


