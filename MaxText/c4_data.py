"""Language Model configurations on the T5/C4 dataset."""

import functools
import math
from typing import Dict, List, Optional, Sequence

from absl import logging
import jax
from jax import numpy as jnp

import seqio
import t5.data
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics

from jax.sharding import PartitionSpec as P
import tensorflow as tf

GPT_SPM_PATH = (
    'gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model'
)
GPT_EOS_ID = 1
GPT_VOCABULARY = t5.data.SentencePieceVocabulary(GPT_SPM_PATH)
PASS_THROUGH_VOCABULARY = t5.data.PassThroughVocabulary(size=50257)

C4_GPT_TRAIN_FEATURES_LM = {
    'targets': t5.data.Feature(vocabulary=GPT_VOCABULARY, add_eos=False)
}

C4_GPT_EVAL_FEATURES_LM = {
    'targets': t5.data.Feature(
        vocabulary=PASS_THROUGH_VOCABULARY, add_eos=False
    )
}
C4_TRAIN_DATADIR = 'gs://test-example-123/datasets/'
C4_EVAL_DATADIR = 'gs://test-example-123/datasets/'

class TaskRegistry(t5.data.TaskRegistry):
  """Task registry with extra tracking."""

  TASK_NAMES = []

  @classmethod
  def add_versioned_tfds_task(cls,
                              name: str,
                              *,
                              versions: List[str],
                              pinned_version: Optional[str] = None,
                              tfds_name: str,
                              tfds_data_dir: Optional[str] = None,
                              **kwargs) -> List[seqio.Task]:
    tasks = []
    for version in versions:
      tasks.append(
          cls.add(
              f'{name}_{version}',
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    if pinned_version is not None:
      tasks.append(
          cls.add(
              name,
              seqio.Task,
              source=seqio.TfdsDataSource(
                  tfds_name=f'{tfds_name}:{pinned_version}',
                  tfds_data_dir=tfds_data_dir,
              ),
              **kwargs,
          ))
    return tasks

# C4 corpus for language model pretraining
TaskRegistry.add_versioned_tfds_task(
    'c4_lm_v301_gpt',
    versions=['3.0.4'],
    pinned_version='3.0.4',
    tfds_name='c4/en',
    tfds_data_dir=C4_TRAIN_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                'inputs': None,
                'targets': 'text',
            },
        ),
        seqio.preprocessors.tokenize,
        functools.partial(
            t5_preprocessors.reduce_concat_tokens,
            batch_size=4096,
        ),
        t5_preprocessors.split_tokens_to_targets_length,
    ],
    output_features=C4_GPT_TRAIN_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=10000,
)

TaskRegistry.add_versioned_tfds_task(
    'c4_lm_v301_gpt_eval_tokenized',
    versions=['3.0.5'],
    pinned_version='3.0.5',
    tfds_name='c4/en',
    tfds_data_dir=C4_EVAL_DATADIR,
    preprocessors=[
        functools.partial(
            t5_preprocessors.rekey,
            key_map={
                'inputs': None,
                'targets': 'ids',
            },
        ),
        seqio.preprocessors.tokenize,
    ],
    output_features=C4_GPT_EVAL_FEATURES_LM,
    metric_fns=[],
    shuffle_buffer_size=None,
)

TRAIN_MIXTURE = 'c4_lm_v301_gpt'
EVAL_MIXTURE = 'c4_lm_v301_gpt_eval_tokenized'

def _dataset_common(is_training:bool, percore_batch_size: int, TRAINING_SEED=8731, MAX_SEQ_LEN=2048, OUTPUTS_LENGTH=256, TRAINING_NUM_BATCHES_TO_SKIP=False, shard_info=None) -> tf.data.Dataset:

    num_local_devices = jax.local_device_count()
    global_batch_size = int(
        percore_batch_size * num_local_devices * jax.process_count() + 1e-6
    )
    if percore_batch_size >= 1:
      assert global_batch_size % num_local_devices == 0
      batch_size_per_process = int(
          math.ceil(percore_batch_size) * num_local_devices + 1e-6
      )
      num_infeed_hosts = global_batch_size // batch_size_per_process
    else:
      if jax.process_count() > 1:
        assert global_batch_size % num_local_devices == 0
        batch_size_per_process = num_local_devices
        num_infeed_hosts = global_batch_size // batch_size_per_process
      else:
        batch_size_per_process = int(
            percore_batch_size * num_local_devices + 1e-6
        )
        num_infeed_hosts = 1
    seed = None
    if is_training:
      seed = TRAINING_SEED
      # TODO(sgpyc): enable sync of seeds across hosts, currently the
      # following failed because of "sync_global_devices name mismatch"
      # seed = jnp.int32(multihost_utils.broadcast_one_to_all(seed))
      logging.info('Train input seed: %s',
                   'None' if seed is None else seed)
      
    dataset = seqio.get_dataset(
      mixture_or_task_name=TRAIN_MIXTURE if is_training else EVAL_MIXTURE,
      task_feature_lengths={
      'inputs': MAX_SEQ_LEN,
      'targets': OUTPUTS_LENGTH},         
      dataset_split='train2' if is_training else 'validation_tokenized_5662seqs',
      use_cached=False,
      num_epochs=None,
      feature_converter=seqio.seqio.DecoderFeatureConverter(
            pack=True if is_training else False,
            bos_id=0,
        ),
      shuffle=True if is_training else False,
      batch_size=batch_size_per_process,
      shard_info = shard_info,
      seed = seed,
    )
    return dataset

def get_c4_datasets(
  percore_batch_size
):
  train_ds = _dataset_common(is_training=True, percore_batch_size=percore_batch_size, TRAINING_SEED=1234)
  # shard the dataset as soon as it is loaded
  train_ds = train_ds.shard(num_shards = jax.process_count(), index = jax.process_index())

  eval_ds = _dataset_common(is_training=False, percore_batch_size=percore_batch_size)
  eval_ds = eval_ds.shard(num_shards = jax.process_count(), index = jax.process_index())

  return train_ds, eval_ds

# TODO / FIXME: should return the tokenizer!!
# TODO: use the config for input sizing
def make_c4_v304_train_iterator_and_tokenizer(config):
  """ Make train iterator and tokenizer for c4 dataset"""
  train_ds, eval_ds = get_c4_datasets(
    percore_batch_size=1
  )

  def filter_keys(record):
    """
    Maps data records from the  LMFeatureConverter format to the format
    expected by MaxText train.py.
    """
    data = {
      'inputs': record['decoder_input_tokens'],
      'targets': record['decoder_target_tokens'],
      # UNSUPPORTED 'decoder_loss_weights': record['decoder_loss_weights']
    }
    if 'decoder_segment_ids' in record:
      data['inputs_segmentation'] = record['decoder_segment_ids']
    if 'decoder_positions' in record:
      data['inputs_position']: record['decoder_positions']

    return data
    
  train_ds = train_ds.map(filter_keys,num_parallel_calls=tf.data.AUTOTUNE)
  eval_ds = eval_ds.map(filter_keys,num_parallel_calls=tf.data.AUTOTUNE)
  
  return train_ds, eval_ds
    