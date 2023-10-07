# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true TPU_MEGACORE=MEGACORE_DENSE"
# python3 MaxText/train.py MaxText/configs/base.yml run_name=1xv3-32 dcn_data_parallelism=1 ici_data_parallelism=2 ici_tensor_parallelism=16 steps=10 enable_profiler=true remat_policy=full base_emb_dim=4096 base_num_heads=64 head_dim=64 vocab_size=50272 base_num_decoder_layers=16 per_device_batch_size=0.5 file_pattern_for_train_data="gs://yejingxin-us-central2/external/lg/dummy-data/train/*.tfrecords" file_pattern_for_eval_data="gs://yejingxin-us-central2/external/lg/dummy-data/valid/*.tfrecords"

#converting in v3-8
# python3 MaxText/train.py MaxText/configs/base.yml run_name=1xv3-32 dcn_data_parallelism=1 ici_data_parallelism=2 ici_tensor_parallelism=4 steps=10 enable_profiler=true remat_policy=full base_emb_dim=4096 base_num_heads=64 head_dim=64 vocab_size=50272 base_num_decoder_layers=16 per_device_batch_size=0.5 file_pattern_for_train_data="gs://yejingxin-us-central2/external/lg/dummy-data/train/LG_GPT_KO2_8_11_100000.tfrecords" file_pattern_for_eval_data="gs://yejingxin-us-central2/external/lg/dummy-data/valid/LG_GPT_KO2_8_2_100000.tfrecords" enable_profiler=true base_output_directory="gs://mazumdera-test-bucket/maxtext/lg/10052023/1" dataset_type="lg" max_predict_length=512
python3 MaxText/train.py MaxText/configs/base.yml run_name=1xv3-32 save_period=5 ici_data_parallelism=2 ici_tensor_parallelism=4 steps=20 enable_profiler=true remat_policy=full base_emb_dim=4096 base_num_heads=64 head_dim=64 vocab_size=50272 base_num_decoder_layers=16 per_device_batch_size=0.5 file_pattern_for_train_data="gs://yejingxin-us-central2/external/lg/dummy-data/train/*.tfrecords" file_pattern_for_eval_data="gs://yejingxin-us-central2/external/lg/dummy-data/valid/*tfrecords" enable_profiler=true base_output_directory="gs://mazumdera-test-bucket/maxtext/lg/10052023/4" dataset_type="lg" max_predict_length=512
# TFLOP/s 165, 22B
