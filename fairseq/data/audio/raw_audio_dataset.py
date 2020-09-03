# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import logging
import numpy as np
import sys

import torch
import torch.nn.functional as F

from .. import FairseqDataset
from fairseq.data.data_utils import compute_mask_indices

logger = logging.getLogger(__name__)


class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        # debug-tpu
        pad=True,
        # pad=False,
        normalize=False,
        args=None
    ):
        super().__init__()
        self.args = args
        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    def collater(self, samples):
        # print(f"DEBUG_MESSAGE: data collater called: len(samples) = {len(samples)}")
        # raise RuntimeError("get trace")
        # from fairseq import pdb; pdb.set_trace()
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]
        # debug-tpu
        padded_sizes = [s["padded_size"] for s in samples]

        if self.pad:
            # debug-tpu
            target_size = min(max(padded_sizes), self.max_sample_size)
            # target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        # debug-tpu
        collated_sources = sources[0].new(len(sources), target_size)
        # collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        
        # debug-tpu
        # if False:
        if True:
            # replace with static mask and re-test
            # it's also possible that TPU hates different number of "TRUE" in masking
            # also wps without masking is not a good metric, because the way it's computed.
            # compute mask indices here at data collater instead of model forward path
            # DEBUG_MESSAGE: masking params B = 1, T = 781, C = 768, padding_mask = torch.Size([1, 781])
            # DEBUG_MESSAGE: masking params mask_indices.shape = (1, 781)
            B, T, C = len(sources), 781, 768
            feature_padding_mask = None
            if padding_mask is not None:
                extra = padding_mask.size(1) % T
                if extra > 0:
                    feature_padding_mask = padding_mask[:, :-extra]
                feature_padding_mask = feature_padding_mask.view(feature_padding_mask.size(0), T, -1)
                feature_padding_mask = feature_padding_mask.all(-1)

            # compute mask
            mask_indices, left_mask, right_mask = compute_mask_indices(
                (B, T),
                feature_padding_mask,
                self.args.mask_prob,
                self.args.mask_length,
                self.args.mask_selection,
                self.args.mask_other,
                min_masks=2,
                no_overlap=self.args.no_mask_overlap,
                min_space=self.args.mask_min_space,
            )

            # compute channel mask
            mask_channel_indices, _, _ = compute_mask_indices(
                (B, C),
                None,
                self.args.mask_channel_prob,
                self.args.mask_channel_length,
                self.args.mask_channel_selection,
                self.args.mask_channel_other,
                no_overlap=self.args.no_mask_channel_overlap,
                min_space=self.args.mask_channel_min_space,
            )

            # print(f"DEBUG_MESSAGE: data collater called: collated_sources.size() = {collated_sources.size()}, B = {B}, T = {T}, padding_mask.size() = {padding_mask.size()}, mask_indices.shape = {mask_indices.shape}")
            input["mask_indices"] = mask_indices
            input["left_mask"] = left_mask
            input["right_mask"] = right_mask
            input["mask_channel_indices"] = mask_channel_indices
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]


class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
        # debug-tpu
        batch_shapes=None,
        num_batch_buckets=0,
        args=None,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
            args=args,
        )
        # debug-tpu
        self.num_batch_buckets = num_batch_buckets
        self.batch_shapes = eval(batch_shapes) if batch_shapes is not None else batch_shapes
        self.padded_sizes = None
        if batch_shapes:
            self.padded_sizes = []
            self._fixed_lengths = np.array([l for (batch,l) in self.batch_shapes])
       
        self.fnames = []

        skipped = 0
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.sizes.append(sz)
                # debug-tpu
                if self.batch_shapes:
                    self.padded_sizes.append(self._fixed_lengths[np.abs(self._fixed_lengths-sz).argmin()])


        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    # debug-tpu
    def get_batch_shapes(self):
        return self.batch_shapes

    def __getitem__(self, index):
        import soundfile as sf

        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = sf.read(fname)

        s = wav.shape[0]
        #wav_pad = np.zeros((250000,))
        #if s <= 250000: 
        #    wav_pad[:s] = wav
        #else:
        #    wav_pad = wav[:250000]

        # debug-tpu
        if self.padded_sizes:
            psz = self.padded_sizes[index]
        else:
            psz = self.sizes[index] 

        if self.batch_shapes is not None and wav.shape > psz: 
            wav = wav[:psz]
        feats = torch.from_numpy(wav).float()
        #feats = torch.from_numpy(wav_pad).float()
        feats = self.postprocess(feats, curr_sample_rate)

        return {
                "id": index,
                # debug-tpu
                "padded_size": psz,
                "source": feats
                }
