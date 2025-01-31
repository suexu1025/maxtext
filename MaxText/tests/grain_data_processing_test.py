"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """


import os
import sys
import jax
from jax.sharding import Mesh
from jax.experimental import mesh_utils

import unittest

import pyconfig
from input_pipeline import _grain_data_processing
from multihost_dataloading import get_next_batch_sharded

if os.path.exists("../../gcsfuse"):
    os.system("fusermount -u ../../gcsfuse")
    os.rmdir("../../gcsfuse")
# os.mkdir("../gcsfuse")
exit_code = os.system("bash ../setup_gcsfuse.sh DATASET_GCS_BUCKET=maxtext-dataset MOUNT_PATH=../../gcsfuse")
if exit_code != 0:
    raise ValueError(f"Running setup_gcsfuse.sh failed with exit code: {exit_code}")

class GrainDataProcessingTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        pyconfig.initialize([sys.argv[0], 'configs/base.yml'],
                            per_device_batch_size=1,
                            run_name='test',
                            mesh_axes = ['data'],
                            logical_axis_rules = [['batch', 'data']],
                            data_sharding = ['data'],
                            base_output_directory = "gs://max-experiments/",
                            dataset_path = "../../gcsfuse",
                            assets_path = "../assets",
                            enable_checkpointing=False,
                            dataset_type="c4-array_record",
                            dataset_name='array-record/c4/en/3.0.1',
                            eval_dataset_name='array-record/c4/en/3.0.1')
        self.config = pyconfig.config
        self.mesh_shape_1d = (len(jax.devices()),)
        self.mesh = Mesh(mesh_utils.create_device_mesh(self.mesh_shape_1d), self.config.mesh_axes)
        self.train_ds, self.eval_ds = self._get_datasets()
        self.train_iter, self.eval_iter, self.predict_iter = self._get_preprocessed_datasets()
        # self.devices_array = create_device_mesh(self.config)
        # self.mesh = Mesh(self.devices_array, self.config.mesh_axes)


    def _get_datasets(self):
        print("Sharding dataset in ", jax.process_count(), " shards")
        train_ds, eval_ds = _grain_data_processing.get_datasets(
                config=self.config)
        return train_ds, eval_ds

    def _get_preprocessed_datasets(self):
        # mesh_shape_1d = (len(jax.devices()),)
        # mesh = Mesh(mesh_utils.create_device_mesh(mesh_shape_1d), self.config.mesh_axes)

        train_iter, eval_iter, test_iter, _ = _grain_data_processing.preprocess_dataset(
                self.config,
                self.mesh,
                self.train_ds, self.eval_ds,
                vocab_path=os.path.join(self.config.assets_path, self.config.vocab_relative_path))
        return train_iter, eval_iter, test_iter

    def test_train_ds(self):
        expected_shape = [jax.device_count(), self.config.max_target_length]
        # For training we pack multiple short examples in one example.
        # *_position and *_segmentation indicate the boundaries.
        batch = get_next_batch_sharded(self.train_iter, self.mesh)
        self.assertEqual({k: list(v.shape) for k, v in batch.items()}, {
            'inputs': expected_shape,
            'inputs_position': expected_shape,
            'inputs_segmentation': expected_shape,
            'targets': expected_shape,
            'targets_position': expected_shape,
            'targets_segmentation': expected_shape,
        })

    def test_eval_ds(self):
        expected_shape = [jax.device_count(), self.config.max_target_length]
        batch = get_next_batch_sharded(self.eval_iter, self.mesh)
        self.assertEqual({k: list(v.shape) for k, v in batch.items()}, {
            'inputs': expected_shape,
            'inputs_position': expected_shape,
            'inputs_segmentation': expected_shape,
            'targets': expected_shape,
            'targets_position': expected_shape,
            'targets_segmentation': expected_shape,
        })


    def test_predict_ds(self):
        expected_shape = [jax.device_count(), self.config.max_target_length]
        batch = get_next_batch_sharded(self.predict_iter, self.mesh)
        self.assertEqual({k: list(v.shape) for k, v in batch.items()}, {
            'inputs': expected_shape,
            'inputs_position': expected_shape,
            'inputs_segmentation': expected_shape,
            'targets': expected_shape,
            'targets_position': expected_shape,
            'targets_segmentation': expected_shape,
        })

    def test_batch_determinism(self):
        batch1 = next(self.train_iter)
        self.train_ds, _ = self._get_datasets()
        train_iter, _, _= self._get_preprocessed_datasets()
        batch2 = next(train_iter)
        self.assertTrue((batch1['inputs']==batch2['inputs']).all())
        self.assertTrue((batch1['targets']==batch2['targets']).all())
        self.assertTrue((batch1['inputs_segmentation']==batch2['inputs_segmentation']).all())
        self.assertTrue((batch1['targets_segmentation']==batch2['targets_segmentation']).all())
        self.assertTrue((batch1['inputs_position']==batch2['inputs_position']).all())
        self.assertTrue((batch1['targets_position']==batch2['targets_position']).all())

if __name__ == '__main__':
  unittest.main()
