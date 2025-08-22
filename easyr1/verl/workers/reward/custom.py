# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


from collections import defaultdict
from typing import Callable, Dict, List, Tuple, TypedDict

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, reason_with_in_limit_compute_score
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: float
    accuracy: float


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: RewardConfig):
        self.config = config
        self.tokenizer = tokenizer
        if config.score_function == "math":
            self.compute_score: Callable[[str, str], RewardScore] = math_compute_score
        elif config.score_function == "r1v":
            self.compute_score: Callable[[str, str], RewardScore] = r1v_compute_score
        elif config.score_function == "reason_with_in_limit":
            self.reason_with_in_limit_compute_score: Callable[[str, str, int, int], RewardScore] = reason_with_in_limit_compute_score
        else:
            raise NotImplementedError(f"Unknown score function {config.score_function}.")

    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  
            # g 奖励函数所在处+dataitem数据读取用法
            # DataProtoItem
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            budget_and_tokens = data_item.batch["budget_and_tokens"]
            origin_response_length = data_item.batch["origin_response_length"]
            prompt_ids = data_item.batch['prompts']
            
            # print("*" * 100 + "budget_and_tokens.shape", budget_and_tokens.shape, "*" * 100)
            # print("*" * 100 + "origin_response_length.shape", origin_response_length.shape, "*"* 100)
            
            
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            valid_prompt_ids = prompt_ids

            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            raw_response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )
            # print(f"raw_response_str: {raw_response_str}")
            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=True
            )
            
            budget = data_item.non_tensor_batch["budget"]

            current_epoch = data_item.non_tensor_batch["current_epoch"]

            ground_truth = data_item.non_tensor_batch["ground_truth"]

            if self.config.score_function == "math" or self.config.score_function == "r1v":
                score = self.compute_score(response_str, ground_truth)
            elif self.config.score_function == "reason_with_in_limit":
                score = self.reason_with_in_limit_compute_score(response_str, ground_truth, current_length=origin_response_length, budget=budget_and_tokens, current_epoch=current_epoch, prompt_str=prompt_str, raw_response_str=raw_response_str)


            reward_tensor[i, valid_response_length - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics
