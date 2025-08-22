import os 
import json


def get_answer_prompt():
    answer_prompt_data = []
    with open("/mnt/lyc/wuxinrui/Qwen2.5-Math/evaluation/data/math500/test.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            answer = data["answer"]
            problem = data["problem"]
            
            answer_prompt = f"The answer to this question is {answer}. Based on the answer and the constraints of the thought chain length, you should deduce the most logical reasoning process. Note: During the thought process, you should pretend not to have seen the answer, but you must rationally infer the correct answer mentioned earlier based on the content of the thought chain."

            data['problem'] = problem+answer_prompt
            answer_prompt_data.append(data)

    with open("/mnt/lyc/wuxinrui/Qwen2.5-Math/evaluation/data/math500_answer_prompt/test.jsonl", "w") as f:
        for data in answer_prompt_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    get_answer_prompt()
    