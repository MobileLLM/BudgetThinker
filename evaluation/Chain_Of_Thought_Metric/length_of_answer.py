import os
import re
from transformers import AutoTokenizer
import json
import matplotlib.pyplot as plt

# 父文件夹路径
parent_folder = "/mnt/lyc/wuxinrui/Qwen2.5-Math/evaluation"
pattern = re.compile(r"MODEL-.*-TIP-.*-STAGE-add-DATA-.*")

entry_list = []
setting_names = []
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B", trust_remote_code=True)  # 使用合适的tokenizer
tokenizer = AutoTokenizer.from_pretrained("/mnt/lyc/wuxinrui/LLaMA-Factory/TCMv4_8ratio/1_5B_TCMv4_8ratio_models/models", trust_remote_code=True)

def calculate_token_length(text):
    """计算文本的token长度"""
    tokens = tokenizer(text)['input_ids']
    return len(tokens)

if __name__ == "__main__":
    for folder in os.listdir(parent_folder):
        # if pattern.match(folder):
        if folder == "MODEL-DS_QW_1_5B-TIP-prompt_v2-STAGE-2-DATA-math500":
            entry_list.append(os.path.join(parent_folder, folder))
            setting_names.append(folder)

    for entry, setting_name in zip(entry_list, setting_names):
        token_length_metrics = []
        final_answer_metrics = []
        
        for sub_entry in os.listdir(entry):
            if not os.path.isdir(os.path.join(entry, sub_entry)):
                continue
                
            for root, dirs, files in os.walk(os.path.join(entry, sub_entry)):
                for file in files:
                    if "metrics" in file:
                        continue
                        
                    cot_answer_path = os.path.join(root, file)
                    
                    with open(cot_answer_path, "r") as f:
                        token_length_data = {}
                        final_answer_data = {}
                        budget_length = int(sub_entry)
                        token_length_data['budget_length'] = budget_length
                        final_answer_data['budget_length'] = budget_length
                        total_tokens = 0
                        answer_count = 0
                        single_final_answer_count = 0
                        multiple_final_answer_count = 0
                        total_delta_length = 0  # 新增：总长度差
                        valid_delta_count = 0   # 新增：有效计数（长度小于budget的）
                        
                        for line in f:
                            data = json.loads(line)
                            answer_text = data['code'][0]
                            token_length = calculate_token_length(answer_text)
                            total_tokens += token_length
                            answer_count += 1
                            
                            # 计算delta_length（新增部分）
                            if token_length < budget_length:
                                delta = budget_length - token_length
                                total_delta_length += delta
                                valid_delta_count += 1
                            
                            # 判断 **Final Answer** 的数量
                            first_match = answer_text.find("</think>")
                            if first_match != -1:
                                modified_text = answer_text[:first_match] + answer_text[first_match + len("</think>"):]
                                second_match = modified_text.find("</think>")
                                
                                if second_match == -1:
                                    single_final_answer_count += 1
                                else:
                                    multiple_final_answer_count += 1
                        
                        avg_token_length = total_tokens / answer_count if answer_count > 0 else 0
                        token_length_data['avg_token_length'] = avg_token_length
                        token_length_data['total_tokens'] = total_tokens
                        
                        # 新增：计算平均delta_length
                        avg_delta_length = total_delta_length / valid_delta_count if valid_delta_count > 0 else 0
                        token_length_data['avg_delta_length'] = avg_delta_length
                        token_length_data['avg_delta_length/BUDGET'] = avg_delta_length / token_length_data['budget_length']
                        token_length_data['valid_delta_count'] = valid_delta_count
                        
                        token_length_metrics.append(token_length_data)
                        
                        final_answer_data['single_final_answer_count'] = single_final_answer_count
                        final_answer_data['multiple_final_answer_count'] = multiple_final_answer_count
                        final_answer_data['total_answer_count'] = answer_count
                        final_answer_data['multiple_final_answer_ratio'] = multiple_final_answer_count / answer_count if answer_count > 0 else 0
                        final_answer_metrics.append(final_answer_data)
        
        # 按budget_length排序
        token_length_metrics.sort(key=lambda x: x['budget_length'])
        final_answer_metrics.sort(key=lambda x: x['budget_length'])
        
        # 绘制图表 - 平均token长度
        fig, ax = plt.subplots(figsize=(10, 6))
        budget_lengths = [data['budget_length'] for data in token_length_metrics]
        avg_lengths = [data['avg_token_length'] for data in token_length_metrics]
        
        length_csv = os.path.join(entry, "token_length_metrics.csv")
        with open(length_csv, "w") as f:
            f.write("budget_length,avg_token_length,total_tokens,avg_delta_length,valid_delta_count, avg_delta_length/BUDGET]\n")
            for data in token_length_metrics:
                f.write(f"{data['budget_length']},{data['avg_token_length']},{data['total_tokens']},{data['avg_delta_length']},{data['valid_delta_count']}, {data['avg_delta_length/BUDGET']}\n")
        
        ax.plot(budget_lengths, avg_lengths, marker='o', color='blue', linewidth=2)
        ax.set_title(f"Average Token Length by Budget Length - {setting_name}")
        ax.set_xlabel('Budget Length')
        ax.set_ylabel('Average Token Length')
        ax.grid(True)
        
        plt.tight_layout()
        pic_name = os.path.join(entry, "token_length_metrics.png")
        plt.savefig(pic_name, dpi=300)
        print(f"Saved figure as {pic_name}")
        plt.close()
        
        # 新增：绘制delta_length图表
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_deltas = [data['avg_delta_length'] for data in token_length_metrics]
        
        ax.plot(budget_lengths, avg_deltas, marker='o', color='green', linewidth=2)
        ax.set_title(f"Average Delta Length by Budget Length - {setting_name}")
        ax.set_xlabel('Budget Length')
        ax.set_ylabel('Average Delta Length (budget - actual)')
        ax.grid(True)
        
        plt.tight_layout()
        delta_pic_name = os.path.join(entry, "delta_length_metrics.png")
        plt.savefig(delta_pic_name, dpi=300)
        print(f"Saved delta figure as {delta_pic_name}")
        plt.close()
        
        # 保存 **Final Answer** 的统计结果
        final_answer_csv = os.path.join(entry, "final_answer_metrics.csv")
        with open(final_answer_csv, "w") as f:
            f.write("budget_length,single_final_answer_count,multiple_final_answer_count,total_answer_count,multiple_final_answer_ratio\n")
            for data in final_answer_metrics:
                f.write(f"{data['budget_length']},{data['single_final_answer_count']},{data['multiple_final_answer_count']},{data['total_answer_count']},{data['multiple_final_answer_ratio']}\n")