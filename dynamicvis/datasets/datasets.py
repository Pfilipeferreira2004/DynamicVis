import itertools
import json
import math
import os
import warnings
from typing import List, Union
import mmengine
import numpy as np
from braceexpand import braceexpand
from mmengine.dataset import Compose

from mmdet.registry import DATASETS as DET_DATASETS
from mmpretrain.datasets import BaseDataset as MMPretrainBaseDataset
from mmdet.datasets import BaseDetDataset
from .category_map import CATEGORIES
import webdataset as wds



@DET_DATASETS.register_module()
class PretrainFmowWebDataset(wds.DataPipeline):
	def __init__(
			self,
			shards_path_or_url: Union[str, List[str]],
			data_name: str = "Fmow",
			pipeline: List[dict] = None,
			per_gpu_batch_size: int = 1,
			num_workers: int = 0,
			shuffle_buffer_size: int = 1000,
			test_mode: bool = False,
	):
		if not isinstance(shards_path_or_url, str):
			shards_path_or_url = [list(braceexpand(urls)) for urls in shards_path_or_url]
			# flatten list using itertools
			shards_path_or_url = list(itertools.chain.from_iterable(shards_path_or_url))
		self.shards_path_or_url = shards_path_or_url
		self.test_mode = test_mode

		self.metainfo = {'classes': CATEGORIES[data_name]}
		self.cat2label = {cat: i for i, cat in enumerate(self.metainfo['classes'])}
		self.transform_pipeline = Compose(pipeline)

		# Create train dataset and loader
		pipeline = [
			wds.ResampledShards(shards_path_or_url),
			# wds.SimpleShardList(shards_path_or_url),  # if use ResampledShards, it will shuffle the shards, and split by node and worker
			# wds.shuffle(100),
			# wds.split_by_node,
			# wds.split_by_worker,
			wds.tarfile_to_samples(),
			wds.shuffle(shuffle_buffer_size) if not test_mode else None,
			wds.map(self.transform),
		]
		super().__init__(*pipeline)

		num_gpu = mmengine.dist.get_world_size()

		global_batch_size = per_gpu_batch_size * num_gpu
		num_batches = math.ceil(self.real_len() / global_batch_size)
		num_workers = max(1, num_workers)
		self.num_worker_batches = math.ceil(num_batches / num_workers)  # per dataloader worker
		self.num_batches = self.num_worker_batches * num_workers
		self.num_samples = self.num_batches * global_batch_size

		self.with_length(self.num_samples // num_gpu)  # In Dataloader, if a iterable dataset, the length is defined as the length of the iterable dataset divided by bs
		self.with_epoch(self.num_worker_batches)  # multiple of per_gpu_batch_size as we loop the batch in external dataloader

	def transform(self, sample):
		'''
		'jpg': img_bytes,
		'json': gt_data
		'''
		sample_key = sample["__key__"]
		img_bytes = sample['jpg.jpg']
		gt_data = json.loads(sample['jpg.json'])
		gt_bboxes = gt_data['gt_bboxes']
		gt_bboxes_labels = [self.cat2label[x] for x in gt_data['gt_bboxes_labels']]
		data_info = dict(
			img_path=sample_key,
			img_bytes=img_bytes,
			gt_bboxes=np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4)),
			gt_bboxes_labels=np.array(gt_bboxes_labels, dtype=np.int64)
		)
		results = self.transform_pipeline(data_info)
		return results

	def real_len(self):
		meta_file = os.path.dirname(self.shards_path_or_url)+'/meta.json'
		if not os.path.exists(meta_file):
			warnings.warn(f"meta file {meta_file} not found")
			num_samples = 10000
		else:
			num_samples = mmengine.load(meta_file)['num_samples']
		if mmengine.dist.is_main_process():
			print(f"real_len: {num_samples}")
		return num_samples