# BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens

## Table of Contents

- [About](#About)
- [Install](#Install)
- [Preparation](#preparation)
- [Training](#training)
- [Evaluation](#evaluation)


## About
This repository contains the code implementation for the paper : BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens.

Our training data can be downloaded from the following links:
SFT-Data：
RL-Data：

The trained model (based on DeepSeek-R1-Distill-Qwen-1.5B) can be obtained from the following link:


## Install

### Clone This Repo

### SFT-Stage：LLaMA-Factory

```
git clone git@github.com:hiyouga/LLaMA-Factory.git
```

After cloning the repository, follow the instructions in the [Installation Guide](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/installation.html) to configure the necessary dependencies.

### Modify Environments' Code

You need to modify a piece of code in the transformers library within the environment corresponding to the LLaMA-Factory project. Locate the source code of the transformers library in your environment and replace the loss/loss_utils.py file. For example, using my path:

```bash
/home/user/anaconda3/envs/llama-fac/lib/python3.11/site-packages/transformers/loss/loss_utils.py

↕️

to_replace/transformers/loss/loss_utils.py
```

> Note: The version of the transformers library corresponding to this code is 4.46.1.

The modified code will allow you to adjust the loss weights for special tokens during training by modifying environment variables. The specific instructions are as follows:

```bash
export special_token_loss=F # Set to F to disable loss calculation for special tokens (weight = 0)
export special_token_loss=T # Set to T to enable loss calculation for special tokens (default weight = 1)
export special_token_loss=Tn # Set the loss weight for special tokens, where n is a float representing the specified weight value
# For example: export special_token_loss=T10, which sets the loss weight for special tokens to 10
```

### RL-Stage：EasyR1

The modified project code is included in the `./easyr1` directory. For environment configuration, please refer to the [EasyR1](https://github.com/hiyouga/EasyR1) documentation.


### Eval-Stage: Qwen2.5-Math

The modified project code is included in the `./evaluation` directory. For environment configuration, please refer to the [Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) documentation.


### Modify Environments' Code

It is necessary to modify the code in the environments corresponding to the `./easyr1` and `./evaluation` directories. We need to modify the source code of vllm to support the insertion of special tokens during inference:

#### Method 1: Direct Replacement (Limited to vllm Version 0.7.3)
Locate the `worker/model_runner.py` file in the vllm library and replace it:

```bash
/home/user/anaconda3/envs/easyr1/lib/python3.11/site-packages/vllm/worker/model_runner.py
& 
/home/user/anaconda3/envs/QMath/lib/python3.11/site-packages/vllm/worker/model_runner.py

↕️

to_replace/vllm/worker/model_runner.py
```

> Note: The version of the vllm library corresponding to this code is 0.7.3.

#### Methods 2: Direct Modification

Focus on the execute_model function in the `...vllm/worker/model_runner.py` file. The original version is as follows:

```python

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        ... more code ...
        ... more code ...

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            return hidden_or_intermediate_states

        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )




        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            assert model_input.sampling_metadata is not None
            indices = model_input.sampling_metadata.selected_token_indices
            if model_input.is_prompt:
                hidden_states = hidden_or_intermediate_states.index_select(
                    0, indices)
            elif decode_meta.use_cuda_graph:
                hidden_states = hidden_or_intermediate_states[:len(indices)]
            else:
                hidden_states = hidden_or_intermediate_states

            output.hidden_states = hidden_states

        return [output]
```

Modify the code as follows:

```python

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[SamplerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in ModelRunner")

        ... more code ...
        ... more code ...

        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            return hidden_or_intermediate_states

        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        if not self.is_driver_worker:
            return []

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        #! >>>>>>>>>>> add remaining tokens to output <<<<<<<<<<<<
        import os
        if os.getenv("remaining", "remaing") == "remaing":
            special_tokens = [151665+i for i in range(400)]
            for seq_id in range(len(model_input.sampling_metadata.seq_groups)):
                prompt_token_ids = next(iter(model_input.sampling_metadata.seq_groups[seq_id].seq_data.values())).prompt_token_ids
                output_token_ids_till_now = next(iter(model_input.sampling_metadata.seq_groups[seq_id].seq_data.values())).output_token_ids
                # reversely iterate outputtoken_ids_till_now, which is a tuple, to find the last special token
                last_special_token_idx, last_special_token = None, None
                for idx in range(len(output_token_ids_till_now)-1, -1, -1):
                    token_id = output_token_ids_till_now[idx]
                    if token_id in special_tokens:
                        last_special_token_idx = idx
                        last_special_token = token_id
                        break
                if last_special_token == 151665:  # has reached the last special token of <remaining 50>
                    continue
                if last_special_token_idx is not None:
                    distance_to_last_special_token = len(output_token_ids_till_now) - last_special_token_idx - 1
                    if distance_to_last_special_token == 50:
                        output.outputs[seq_id].samples[0].output_token = last_special_token - 1
                        former_key = list(output.outputs[seq_id].samples[0].logprobs.keys())[0]
                        output.outputs[seq_id].samples[0].logprobs[last_special_token - 1] = list(output.outputs[seq_id].samples[0].logprobs.values())[0]
                        # delete former key-value pair
                        
                        #g
                        # print(f"former_key = {former_key}")
                        # print(f"last_special_token - 1 = {last_special_token - 1}")
                        if former_key == last_special_token -1:
                            print("&"*50 + f"former_key == last_special_token -1 == {former_key}" + "!"*50)
                        else:
                            del output.outputs[seq_id].samples[0].logprobs[former_key]
                        #g
                        
                        # del output.outputs[seq_id].samples[0].logprobs[former_key]
                else:  # there has not been any special token in the output
                    last_special_token = None
                    for prompt_token_id in prompt_token_ids:
                        if prompt_token_id in special_tokens:
                            last_special_token = prompt_token_id
                            break
                    if last_special_token is not None:
                        if len(output_token_ids_till_now) == 50:
                            output.outputs[seq_id].samples[0].output_token = last_special_token - 1
                            former_key = list(output.outputs[seq_id].samples[0].logprobs.keys())[0]
                            output.outputs[seq_id].samples[0].logprobs[last_special_token - 1] = list(output.outputs[seq_id].samples[0].logprobs.values())[0]
                            #g
                            # print(f"former_key = {former_key}")
                            # print(f"last_special_token - 1 = {last_special_token - 1}")
                            if former_key == last_special_token -1:
                                print("#"*50 + f"former_key == last_special_token -1 == {former_key}" + "!"*50)
                            else:
                                del output.outputs[seq_id].samples[0].logprobs[former_key]
                            #g
                            # del output.outputs[seq_id].samples[0].logprobs[former_key]

        elif "ratio" in os.getenv("remaining", "remaing"):
            N = int(os.getenv("remaining", "remaing").replace("ratio", ""))
            assert os.getenv("budget") is not None
            budget = int(os.environ["budget"])
            delta = budget // N + 1

            special_tokens = [151665+i for i in range(N-1)]
            for seq_id in range(len(model_input.sampling_metadata.seq_groups)):
                prompt_token_ids = next(iter(model_input.sampling_metadata.seq_groups[seq_id].seq_data.values())).prompt_token_ids
                output_token_ids_till_now = next(iter(model_input.sampling_metadata.seq_groups[seq_id].seq_data.values())).output_token_ids
                # reversely iterate outputtoken_ids_till_now, which is a tuple, to find the last special token
                last_special_token_idx, last_special_token = None, None
                for idx in range(len(output_token_ids_till_now)-1, -1, -1):
                    token_id = output_token_ids_till_now[idx]
                    if token_id in special_tokens:
                        last_special_token_idx = idx
                        last_special_token = token_id
                        break
                if last_special_token == 151665:  # has reached the last special token of <remaining 50>
                    continue
                if last_special_token_idx is not None:
                    distance_to_last_special_token = len(output_token_ids_till_now) - last_special_token_idx - 1
                    if distance_to_last_special_token == delta:
                        output.outputs[seq_id].samples[0].output_token = last_special_token - 1
                        former_key = list(output.outputs[seq_id].samples[0].logprobs.keys())[0]
                        output.outputs[seq_id].samples[0].logprobs[last_special_token - 1] = list(output.outputs[seq_id].samples[0].logprobs.values())[0]
                        # delete former key-value pair
                        
                        #g
                        # print(f"former_key = {former_key}")
                        # print(f"last_special_token - 1 = {last_special_token - 1}")
                        if former_key == last_special_token -1:
                            print("&"*50 + f"former_key == last_special_token -1 == {former_key}" + "!"*50)
                        else:
                            del output.outputs[seq_id].samples[0].logprobs[former_key]
                        #g
                        
                        # del output.outputs[seq_id].samples[0].logprobs[former_key]
                else:  # there has not been any special token in the output
                    last_special_token = 151671 + 1 #g 手动设置成7/8 + 1的token，否则全是从6/8开始输出。
                    if last_special_token is not None:
                        if len(output_token_ids_till_now) == delta:
                            output.outputs[seq_id].samples[0].output_token = last_special_token - 1
                            former_key = list(output.outputs[seq_id].samples[0].logprobs.keys())[0]
                            output.outputs[seq_id].samples[0].logprobs[last_special_token - 1] = list(output.outputs[seq_id].samples[0].logprobs.values())[0]
                            #g
                            # print(f"former_key = {former_key}")
                            # print(f"last_special_token - 1 = {last_special_token - 1}")
                            if former_key == last_special_token -1:
                                print("#"*50 + f"former_key == last_special_token -1 == {former_key}" + "!"*50)
                            else:
                                del output.outputs[seq_id].samples[0].logprobs[former_key]
                            #g
                            # del output.outputs[seq_id].samples[0].logprobs[former_key]
            

        elif os.getenv("remaining", "remaing") == "remaining250":
            special_tokens = [151665+i for i in range(40)]
            for seq_id in range(len(model_input.sampling_metadata.seq_groups)):
                prompt_token_ids = next(iter(model_input.sampling_metadata.seq_groups[seq_id].seq_data.values())).prompt_token_ids
                output_token_ids_till_now = next(iter(model_input.sampling_metadata.seq_groups[seq_id].seq_data.values())).output_token_ids
                # reversely iterate outputtoken_ids_till_now, which is a tuple, to find the last special token
                last_special_token_idx, last_special_token = None, None
                for idx in range(len(output_token_ids_till_now)-1, -1, -1):
                    token_id = output_token_ids_till_now[idx]
                    if token_id in special_tokens:
                        last_special_token_idx = idx
                        last_special_token = token_id
                        break
                if last_special_token == 151665:  # has reached the last special token of <remaining 50>
                    continue
                if last_special_token_idx is not None:
                    distance_to_last_special_token = len(output_token_ids_till_now) - last_special_token_idx - 1
                    if distance_to_last_special_token == 250:
                        output.outputs[seq_id].samples[0].output_token = last_special_token - 1
                        former_key = list(output.outputs[seq_id].samples[0].logprobs.keys())[0]
                        output.outputs[seq_id].samples[0].logprobs[last_special_token - 1] = list(output.outputs[seq_id].samples[0].logprobs.values())[0]
                        # delete former key-value pair
                        
                        #g
                        # print(f"former_key = {former_key}")
                        # print(f"last_special_token - 1 = {last_special_token - 1}")
                        if former_key == last_special_token -1:
                            print("&"*50 + f"former_key == last_special_token -1 == {former_key}" + "!"*50)
                        else:
                            del output.outputs[seq_id].samples[0].logprobs[former_key]
                        #g
                        
                        # del output.outputs[seq_id].samples[0].logprobs[former_key]
                else:  # there has not been any special token in the output
                    last_special_token = None
                    for prompt_token_id in prompt_token_ids:
                        if prompt_token_id in special_tokens:
                            last_special_token = prompt_token_id
                            break
                    if last_special_token is not None:
                        if len(output_token_ids_till_now) == 250:
                            output.outputs[seq_id].samples[0].output_token = last_special_token - 1
                            former_key = list(output.outputs[seq_id].samples[0].logprobs.keys())[0]
                            output.outputs[seq_id].samples[0].logprobs[last_special_token - 1] = list(output.outputs[seq_id].samples[0].logprobs.values())[0]
                            #g
                            # print(f"former_key = {former_key}")
                            # print(f"last_special_token - 1 = {last_special_token - 1}")
                            if former_key == last_special_token -1:
                                print("#"*50 + f"former_key == last_special_token -1 == {former_key}" + "!"*50)
                            else:
                                del output.outputs[seq_id].samples[0].logprobs[former_key]
                            #g
                            # del output.outputs[seq_id].samples[0].logprobs[former_key]
        
        else:
            pass
        #! >>>>>>>>>>> add remaining tokens to output <<<<<<<<<<<<


        if self.return_hidden_states:
            # we only need to pass hidden states of most recent token
            assert model_input.sampling_metadata is not None
            indices = model_input.sampling_metadata.selected_token_indices
            if model_input.is_prompt:
                hidden_states = hidden_or_intermediate_states.index_select(
                    0, indices)
            elif decode_meta.use_cuda_graph:
                hidden_states = hidden_or_intermediate_states[:len(indices)]
            else:
                hidden_states = hidden_or_intermediate_states

            output.hidden_states = hidden_states

        return [output]
```

## Preparation

### Model Preparation

```bash
cd ./Preparation
```

Modify the ori_model_path and new_model_path variables in `Preparation/add_special_tokens.py` to embed special tokens into the new model.

```python
    ori_model_path = '/path/to/your/ori/model'
    new_model_path = '/path/to/your/new/model'
```

### Data Preparation

Our training data can be downloaded from the following links:

SFT-Data：

RL-Data：

After downloading the SFT-Data, register it in the `dataset_info.json` file of LLaMA-Factory with the registration name `8ratio_SFT_below10000`.

## Training

### SFT Training
```
cd ./LLaMA-Factory
```

Use deepseed to accelerate the training process.
For detailed scripts, refer to `LLaMA-Factory/examples/deepseed_train.sh`.


### RL Training
```
cd ./easyr1
```

After configuring the `model_path` parameter in the `easyr1/examples/8ratio_v1.sh` and `easyr1/examples/8ratio_v1.yaml` files, you can run the following command:


```bash
bash /mnt/lyc/wuxinrui/BudgetThinker/easyr1/examples/8ratio_v1.sh
```

#### Parameter Introduction

The script involves three environment variables: stage, steady, and remaining.
- stage: 1/2, representing the use of 1/2 stage inference during training.

    Stage 1 represents normal output of the chain of thought.

    Stage 2 represents manually interrupting the output when the chain of thought reaches the budget, and 
manually inserting `</think>\n**Final Answer**` as the ending prompt at the current position, followed by another output.

- steady: Represents the name of the current training session. For example, with "8ratio_v1", it is best to modify all occurrences of this string in both the .sh and .yaml files. This will affect the output location of checkpoints, the output location of logs, and the budget settings under the current training configuration. For more details, refer to easyr1/verl/utils/dataset.py.

- remaining: The vllm inference mode. Setting it to 8ratio uses the default method (splitting the chain of thought into 8 parts). If set to default, vllm will perform normal inference without adding any special tokens.



## Evaluation

First, modify the `MODEL_NAME_OR_PATH` parameter in the `evaluation/remaining_eval/Eval.sh` script, and then run the following command:


```bash
cd ./evaluation

bash evaluation/remaining_eval/Eval.sh
```

### Parameter Introduction

The following parameters/environment variables need to be set in the script:

- remaining/stage: Same as described above.

- tip: The template for the prompt before the question. If using the 8ratio inference mode, the tip must also be set to 8ratio. Additionally, tip can be set to prompt_v1 or prompt_v2, which are two different natural language prompts.

- MODEL_NAME_OR_PATH: The path to the model. It is recommended to use a recognizable model name as the second-to-last folder name in the path, as the code will read this name as the current evaluation model and store logs in the corresponding folder. For example: /path1/path2/Model_Name/models