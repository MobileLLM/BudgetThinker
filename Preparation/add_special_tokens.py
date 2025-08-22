from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import json
# model = AutoModelForCausalLM.from_pretrained("/data/sunyi/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/6602cadec947dbb53e64f3d8d6425320b2197247")
# tokenizer = AutoTokenizer.from_pretrained("/data/sunyi/hf_cache/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/6602cadec947dbb53e64f3d8d6425320b2197247")




def gen_special_tokens_json():
    special_tokens_list = {}
    for i in range(7):
        special_tokens_list[f"{i}"] = f"\n<remaining>{i+1}/8</remaining>\n"
    print(special_tokens_list)
    
    with open('./special_tokens.json', 'w') as f:
        json.dump(special_tokens_list, f)
    print('special_tokens.json has been generated.')

if __name__ == "__main__":
    
    ori_model_path = '/path/to/your/ori/model'
    new_model_path = '/path/to/your/new/model'
    
    model = AutoModelForCausalLM.from_pretrained(ori_model_path)
    tokenizer = AutoTokenizer.from_pretrained(ori_model_path)
    print(model.get_input_embeddings())
    print(model.lm_head)
    print(len(tokenizer))

    gen_special_tokens_json()
    with open('./special_tokens.json') as f:
        special_tokens = json.load(f)
        
    bins_tokens = [
        special_tokens[f"{i}"] for i in range(7)
    ]

    tokenizer.add_special_tokens({'additional_special_tokens': bins_tokens})
    model.resize_token_embeddings(len(tokenizer))
    print('Vocab size after adding special tokens:', len(tokenizer))



    tokenizer.save_pretrained(new_model_path)
    model.save_pretrained(new_model_path)
    model = AutoModelForCausalLM.from_pretrained(new_model_path)
    tokenizer = AutoTokenizer.from_pretrained(new_model_path)
    print(model.get_input_embeddings())
    print(model.lm_head)
    print(len(tokenizer))
