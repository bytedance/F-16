#!/bin/bash

# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT

bash run.sh \
    --training_data /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/preprocess_dataset/ytb0-62kCapHumanDimQa.json \
    --model_base /mnt/bn/tiktok-mm-4/aiic/public/model/OV-Qwen2-7B-AM9 \
    --model /mnt/bn/tiktok-mm-4/aiic/public/model/OV-Qwen2-7B-AM9 \
    --save_model_name debug \
    --epochs 1 --save_steps 1000 --lr 2e-5 --max_time 110 --fps 16 --add_time_token \
    --mm_pooling_position after --multi_frame_projector --multi_frame_num 16
