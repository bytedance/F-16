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

from transformers import AutoConfig
from llava.model import F16ForCausalLM
from transformers import AutoConfig, AutoTokenizer
import torch
import os
import json
from collections import OrderedDict
from peft import LoraConfig, get_peft_model
from dataclasses import make_dataclass

def load_qwen_lora_model(model_path, model_base=None, lora_enable=False, load_full=False, lora_r=128, lora_alpha=256, lora_dropout=0.05, model_max_length=32768, new_model_args=None):
    model_ckpt_path = model_path
    model_path = os.path.dirname(model_ckpt_path)

    lora_ckpt = os.path.join(model_path, "all_parameters.bin")

    with open(os.path.join(model_path, 'config.json'), 'r') as fp:
        config = json.load(fp)

    if model_base is None:
        model_base = config["_name_or_path"]
        while os.path.exists(os.path.join(model_base, "all_parameters.bin")):
            with open(os.path.join(model_base, 'config.json'), 'r') as fp:
                config = json.load(fp)
            model_base = config["_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(model_base, model_max_length=model_max_length, padding_side="right")
    cfg_pretrained = AutoConfig.from_pretrained(model_base)
    if "model_args" in config:
        model_args = config["model_args"]

        model_args["lora_r"] = lora_r
        model_args["lora_alpha"] = lora_alpha
        model_args["lora_dropout"] = lora_dropout

        TempData = make_dataclass('TempData', model_args)
        model_args = TempData(**model_args)

        overwrite_config = {"model_args": vars(model_args), "add_time_token": model_args.add_time_token, "multi_frame_projector": model_args.multi_frame_projector, "multi_frame_num": model_args.multi_frame_num}
    else:
        model_args = new_model_args
        overwrite_config = {"model_args": vars(model_args), "add_time_token": model_args.add_time_token, "multi_frame_projector": model_args.multi_frame_projector, "multi_frame_num": model_args.multi_frame_num}

    for k, v in overwrite_config.items():
        setattr(cfg_pretrained, k, v)

    model = F16ForCausalLM.from_pretrained(model_base, config=cfg_pretrained, cache_dir=None, attn_implementation="flash_attention_2", torch_dtype=(torch.bfloat16))
    model.get_model().initialize_vision_modules(model_args=model_args)
    model = model.to(torch.bfloat16)

    if load_full and lora_enable:
        ckpt = torch.load(lora_ckpt, map_location='cpu')
        if model_ckpt_path != lora_ckpt:
            ckpt_2 = torch.load(model_ckpt_path, map_location='cpu')
            for k in ckpt_2.keys():
                ckpt[k] = ckpt_2[k]
        
        new_ckpt = OrderedDict()
        for k in ckpt.keys():
            if 'vision_tower' not in k:
                new_ckpt[k[len('module.'):]] = ckpt[k]
            else:
                new_ckpt[k[len('module.'):]] = ckpt[k]

        kk = model.load_state_dict(new_ckpt, strict=False)
        print("Load full: ", len(kk.unexpected_keys), len(kk.missing_keys))

    if lora_enable:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        print("Lora Config: ", model_args.lora_r, model_args.lora_alpha, model_args.lora_dropout)
        model.to(torch.bfloat16)

        v_flag = False
        if hasattr(model.model, "vision_tower"):
            vision_tower = model.model.vision_tower
            del model.model.vision_tower
            v_flag = True
        model = get_peft_model(model, lora_config)
        if v_flag:
            model.model.model.vision_tower = vision_tower

    else:
        model.to(torch.bfloat16)
    
    if load_full and lora_enable:
        pass
    else:
        ckpt = torch.load(lora_ckpt, map_location='cpu')

        if model_ckpt_path != lora_ckpt:
            ckpt_2 = torch.load(model_ckpt_path, map_location='cpu')
            for k in ckpt_2.keys():
                ckpt[k] = ckpt_2[k]
        
        new_ckpt = OrderedDict()
        for k in ckpt.keys():
            if 'vision_tower' not in k:
                new_ckpt[k[len('module.'):]] = ckpt[k]
            else:
                new_ckpt[k[len('module.'):]] = ckpt[k]

        kk = model.load_state_dict(new_ckpt, strict=False)

        print(len(kk.unexpected_keys), len(kk.missing_keys))
        print(len([it for it in kk.missing_keys if 'lora' in it or 'final_linear' in it]))

    return model, tokenizer

