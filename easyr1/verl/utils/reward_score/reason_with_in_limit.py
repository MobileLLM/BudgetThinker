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

# Modified BY Xinrui Wu

import re
from typing import Dict
import json
import os
from mathruler.grader import extract_boxed_content, grade_answer
from .get_answer_utils import extract_answer

def limit_format_reward(predict_str: str) -> float:
    # 匹配 **Final Answer**\boxed{数学表达式}，前后可以有任何字符
    pattern = re.compile(r".*\*\*Final Answer\*\*\\boxed\{.*?\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def limit_acc_reward(predict_str: str, ground_truth: str) -> float:
    answer = extract_answer(predict_str)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def anwser_length_reward(current_length, budget) -> float:
    target_length = budget
    if budget >= 1500:
        if current_length >= budget:
            anwser_length_reward = max((1 - 16 * ((current_length - target_length) / target_length) * ((current_length - target_length) / target_length)), 0) # 1- 16*delta**2
        else:
            anwser_length_reward = max((1 - ((current_length - target_length) / target_length) * ((current_length - target_length) / target_length)), 0) # 1- delta**2
    else:
        if current_length >= budget:
            anwser_length_reward = max((1 - 16 * ((current_length - target_length) / target_length) * ((current_length - target_length) / target_length)), 0) # 1- 16*delta**2
        else:
            anwser_length_reward = max((1 - 16 * ((current_length - target_length) / target_length) * ((current_length - target_length) / target_length)), 0) # 1- 16*delta**2
    return anwser_length_reward


def reason_with_in_limit_compute_score(predict_str: str, ground_truth: str, current_length:int, budget:int, current_epoch:int, prompt_str:str, raw_response_str:str) -> Dict[str, float]:
    predict_str = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)
    format = limit_format_reward(predict_str)
    accuracy = limit_acc_reward(predict_str, ground_truth)

    anwser_length = anwser_length_reward(current_length, budget)

    length_weight = 0.15
    format_weight = 0.15
    accuracy_weight = 0.7


    if accuracy == 1 and current_length <= (budget):
        anwser_length = 1.0
    overall = accuracy_weight * accuracy + format_weight * format + length_weight * anwser_length
    
    return {
        "overall": overall,
        "format": format,
        "accuracy": accuracy,
        "anwser_length": anwser_length
    }
