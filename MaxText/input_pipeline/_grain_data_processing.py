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

"""Input pipeline using Grain."""

import os
import re
from typing import Optional

import ml_collections
import jax
import grain.python as grain

import tokenizer
from input_pipeline import _grain_operations
from input_pipeline import _grain_tokenizer

def get_datasets(
  config: ml_collections.ConfigDict
):
  """Load dataset from array_record files for using with grain"""
  data_dir = os.path.join(config.dataset_path, config.dataset_name)
  train_files = [data_dir + '/' + f for f in os.listdir(data_dir) if re.match(r'.*train.*', f)]
  train_ds = grain.ArrayRecordDataSource(train_files)
  if config.eval_dataset_name:
    eval_files = [data_dir + '/' + f for f in os.listdir(data_dir) if re.match(rf'.*{config.eval_split}.*', f)]
    eval_ds = grain.ArrayRecordDataSource(eval_files)
  else:
    eval_ds = train_ds

  return train_ds, eval_ds

def preprocess_dataset(config: ml_collections.ConfigDict,
                        global_mesh,
                        train_ds, eval_ds,
                        vocab_path: Optional[str] = None,
                        data_shuffle_seed = 0,
                        add_bos = True,
                        add_eos = True
                        ):
  """Use grain to pre-process the dataset and return iterators"""
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/lm1b_sentencepiece_model')

  # Load tokenizer
  sp_tokenizer = tokenizer.load_tokenizer(vocab_path=vocab_path,
                                          add_bos=add_bos,
                                          add_eos=add_eos)

  # Set global batch size.
  global_batch_size_to_load = config.global_batch_size_to_load

  if config.eval_per_device_batch_size > 0:
    eval_batch_size = config.eval_per_device_batch_size * global_mesh.size
  else:
    eval_batch_size = global_batch_size_to_load

  train_iter = preprocessing_pipeline(
      train_ds,
      vocab_path,
      add_bos,
      add_eos,
      config.grain_worker_count,
      global_batch_size_to_load,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      num_epochs=1,
      pack_examples=True,
      max_length=config.max_target_length,
      data_shuffle_seed=data_shuffle_seed,)

  eval_iter = preprocessing_pipeline(
      eval_ds,
      vocab_path,
      add_bos,
      add_eos,
      config.grain_worker_count,
      eval_batch_size,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      pack_examples=True,
      max_length=config.max_target_length,
      data_shuffle_seed=data_shuffle_seed,)

  predict_iter = preprocessing_pipeline(
      eval_ds,
      vocab_path,
      add_bos,
      add_eos,
      config.grain_worker_count,
      eval_batch_size,
      global_mesh,
      shuffle=config.enable_data_shuffling,
      pack_examples=True,
      max_length=config.max_target_length,
      data_shuffle_seed=data_shuffle_seed)

  return train_iter, eval_iter, predict_iter, sp_tokenizer

def preprocessing_pipeline(
  dataset,
  vocab_path,
  add_bos: bool,
  add_eos: bool,
  grain_worker_count: int,
  batch_size: int,
  global_mesh,
  shuffle: bool,
  num_epochs: Optional[int] = 1,
  pack_examples: bool = True,
  max_length: int = 512,
  shift: bool = True,
  drop_remainder: bool = True,
  data_shuffle_seed = 0,
):
  """Apply grain operations to preprocess the given dataset."""
  assert (
        batch_size % global_mesh.size == 0
  ), 'Batch size should be divisible number of global devices.'

  operations = []
  operations.append(_grain_operations.ParseFeatures())
  operations.append(_grain_operations.NormalizeFeatures())
  operations.append(_grain_tokenizer.TokenizeAndTrim(["inputs","targets"],
                                                      max_length, vocab_path,
                                                      add_bos, add_eos))

  # Pack and Batch examples.
  if pack_examples:
    operations.append(grain.experimental.PackAndBatchOperation(
                        batch_size=batch_size // jax.process_count(),
                        length_struct={'inputs':max_length,'targets':max_length}))
    operations.append(_grain_operations.ReformatPacking())
  else:
    operations.append(_grain_operations.PadToMaxLength(max_length))
    operations.append(grain.Batch(batch_size=batch_size // jax.process_count(), drop_remainder=drop_remainder))

  # Shift inputs for teacher-forced training
  if shift:
    operations.append(_grain_operations.ShiftData(axis=1))

  index_sampler = grain.IndexSampler(
    num_records=len(dataset),
    num_epochs = num_epochs,
    shard_options=grain.ShardOptions(
      shard_index = jax.process_index(), shard_count = jax.process_count(), drop_remainder = True
    ),
    shuffle = shuffle,
    seed = data_shuffle_seed
  )

  dataloader = grain.DataLoader(
      data_source = dataset,
      operations = operations,
      sampler = index_sampler,
      worker_count=grain_worker_count,
  )

  data_iter = iter(dataloader)

  return data_iter
