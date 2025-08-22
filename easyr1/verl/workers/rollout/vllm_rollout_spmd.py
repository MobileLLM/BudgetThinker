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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
"""

import os
from contextlib import contextmanager
from typing import Any, List, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        vllm_init_kwargs = {}
        if config.limit_images > 0:
            vllm_init_kwargs = {"limit_mm_per_prompt": {"image": config.limit_images}}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            max_model_len=config.prompt_length + config.response_length,
            max_num_batched_tokens=config.max_num_batched_tokens,
            enable_sleep_mode=True,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            disable_log_stats=config.disable_log_stats,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **vllm_init_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {"max_tokens": config.response_length, "detokenize": False}
        
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.tokenizer = tokenizer
        self.stage = os.environ.get("stage", "1")
        print("#"*50 + f"stage = {self.stage}" + "#"*50)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto,) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # print(f"vllm_inputs's length = {len(vllm_inputs)}")
        vllm_inputs = vllm_inputs[:] # id形式的token
        
        budget_array = non_tensor_batch.pop('budget')
        budget = budget_array[0]
        budget_and_tokens = budget + (budget // 50)
        
        # # print(f"vllm_inputs[:10] = {vllm_inputs[:10]}") #g 同一组数据中的budget相同
        # budget_tokens = [i for i in range(151665, 152065)]
        # example = vllm_inputs[0]['prompt_token_ids']
        # for budget_token in budget_tokens:
        #     if budget_token in example:
        #         budget = (budget_token - 151664)*50
        #         budget_and_tokens = budget + (budget // 50)
        # users can customize different sampling_params at different run
        #ddd 不论1/2 stage推理
        with self.update_sampling_params(**prompts.meta_info):
            # cut_params = {"max_tokens": budget_and_tokens}
            
            max_tokens = (budget_and_tokens + (budget_and_tokens // 4)) if (budget_and_tokens + (budget_and_tokens // 4)) <= (self.config.response_length - 200) else (self.config.response_length - 200) #g 1.25倍budeet作为截断长度
            
            print(f"$$$$$$$$$$$$max_tokens = {max_tokens}$$$$$$$$$$$$$$$")
            #g 使用reason_with_in_limit对应reward机制的话，无需额外截断
            cut_params = {"max_tokens": max_tokens}

            #ddd 单独设置截断长度，更新sampling参数
            with self.update_sampling_params(**cut_params):
                print(f"self.sampling_params =  {self.sampling_params}")
                completions: List[RequestOutput] = self.inference_engine.generate(
                    prompts=vllm_inputs, sampling_params=self.sampling_params, use_tqdm=(self.rank == 0)
                )
                response_ids = [output.token_ids for completion in completions for output in completion.outputs]
                # print(f"response_ids = {response_ids}")
                # response_ids = VF.pad_2d_list_to_length(   #g 在第二阶段前，取消padding
                #     response_ids, self.pad_token_id, max_length=self.config.response_length
                # ).to(input_ids.device)
                # print(f"len of response_ids = {len(response_ids)}")
                
                #g 获取原始response长度
                origin_response_length = [len(output) for output in response_ids]
                
                #g 根据 budget_and_tokens 进行截断
                truncated_response_ids = []
                for tokens in response_ids:
                    if len(tokens) > (budget_and_tokens - 50):
                        # 超过 budget_and_tokens 的部分截断
                        truncated_tokens = tokens[:(budget_and_tokens - 50)]
                    else:
                        # 不足 budget_and_tokens 的保留原样
                        truncated_tokens = tokens
                    truncated_response_ids.append(truncated_tokens)
                response_ids = truncated_response_ids
                
                #! point2 : 注意缩进位置，要在val时要同步update_sampling_params，否则会变成5倍
                if self.sampling_params.n > 1:
                    batch_size = batch_size * self.sampling_params.n
                    input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                    attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                    position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                    if "multi_modal_inputs" in non_tensor_batch.keys():
                        non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                            non_tensor_batch["multi_modal_inputs"], self.sampling_params.n
                        )
            #ddd with块结束，sampling参数自动恢复


        #g =========2-stage inference============g#
        # import random
        # if os.environ['steady'] == "FULLv4":
        #     self.stage = "1" if random.random() < 0.5 else "2"
            
        # print(f"self.stage = {self.stage}\n" * 20)
        
        # if self.stage == "2":
        print(f"origin_stage = {self.stage}\n")
        stage = non_tensor_batch["stage"]
        non_tensor_batch.pop("stage") #g 使用后pop.否则形状对不上
        if (stage[0] == 0):
            current_stage = str(self.stage)
        else:
            current_stage = str(stage)
            
        import random
        random_2stage = random.random()
        if budget <=1000 and random_2stage < 0.2:
            current_stage = "2"

        
        print(f"current_stage = {current_stage}\n" * 10)

        if current_stage == "2":
            print("2 stage inference!!!!!!!!!!!!!")
            final_prompt_str = "\n</think>\n**Final Answer**\\boxed"
            final_prompt_token_ids = self.tokenizer.encode(final_prompt_str, add_special_tokens=False)

            # 更新 vllm_inputs，添加 final_prompt_str
            vllm_inputs = []
            for i in range(len(response_ids)):
                # 将 response_ids 转换为字符串
                prompt_str = self.tokenizer.decode(input_ids[i], skip_special_tokens=False)
                response_str = self.tokenizer.decode(response_ids[i], skip_special_tokens=False)
                # 在末尾添加 final_prompt_str
                updated_response_str = prompt_str + response_str + final_prompt_str
                # print(f"updated_response_str = {updated_response_str}")
                # 将更新后的字符串重新编码为 token_ids
                updated_response_ids = self.tokenizer.encode(updated_response_str, add_special_tokens=False)
                vllm_inputs.append({"prompt_token_ids": updated_response_ids})
                
            # print(f"len of vllm_inputs = {len(vllm_inputs)}")
            answer_max_length = 40
            default_sampling_params = SamplingParams(n=1, max_tokens=answer_max_length, temperature=1.0)
            completions_final: List[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=default_sampling_params, use_tqdm=(self.rank == 0)
            )
            final_response_ids = [output.token_ids for completion in completions_final for output in completion.outputs]
            # print(f"final_response_ids = {final_response_ids}")
            
            #g 拼接一阶段和二阶段的输出
            full_response_ids = []
            for i in range(len(response_ids)):
                # print(f"response_ids[i] = {response_ids[i]}")
                combined_response = response_ids[i] + tuple(final_prompt_token_ids) + final_response_ids[i]
                full_response_ids.append(combined_response)
            
            #g 在完成第二阶段推理后进行 padding
            padding_max_length = self.config.response_length + answer_max_length + 10 #g +10是为了统一所有的max_length
            full_response_ids = VF.pad_2d_list_to_length(
                full_response_ids, self.pad_token_id, max_length=padding_max_length
            ).to(input_ids.device)

            # print(f"final_response_ids.shape = {final_response_ids.shape}")
            response_ids = full_response_ids
        #g =========2-stage inference============g#
            
        #g 不进入二阶段，padding
        else:
            response_ids = VF.pad_2d_list_to_length(
                    response_ids, self.pad_token_id, max_length=self.config.response_length
                ).to(input_ids.device)

        print(f"response_ids.shape: {response_ids.shape}")
        print(f"input_ids.shape: {input_ids.shape}")
        
        #g ===results====
        '''
        8 gpus
        validate => response_ids与input_ids的shape分别为torch.Size([630, 6800])和torch.Size([630, 1024])
        '''
        #g ===results====
        
        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "budget_and_tokens": [budget_and_tokens] * len(origin_response_length),  #g 将budget_and_tokens作为tensor传入返回字典，方便reward函数调用
                "origin_response_length": origin_response_length,
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
