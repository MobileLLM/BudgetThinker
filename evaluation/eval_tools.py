import os
def apply_RL_prompt(chunk, args, budget):
    if args.prompt_type == "deepseek3" and os.environ['tip'] == "withoutremaining":
        return withoutremaining_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and os.environ['tip'] == "8ratio":
        return _8ratio_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and "prompt_v1" in os.environ['tip']:
        return prompt_v1_prompt(chunk, budget)
    elif args.prompt_type == "deepseek3" and "prompt_v2" in os.environ['tip']:
        return prompt_v2_prompt(chunk, budget)
    else:
        return chunk

def withoutremaining_prompt(chunk, budget):
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)'
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)\n<remaining>{budget}</remaining>\n'
        add_prompt = f"\n(Complete thinking within {budget} tokens or fewer.)"
        # add_prompt = f'\n<remaining>{budget}</remaining>\n'

        add_response = f""
        # head += f"\n<remaining>{budget}</remaining>\n"
        chunk[i] = head + add_prompt + find_strings + add_response + tail
        # print(f"chunk[i] = {chunk[i]}")
    return chunk


def _8ratio_prompt(chunk, budget):
    os.environ['budget'] = str(budget)
    print(f"budget = {budget}")
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)'
        add_prompt = f"\n(Complete thinking within {budget} tokens or fewer, 7 special tokens ( \n<remaining>7/8</remaining>\n , \n<remaining>6/8</remaining>\n , \n<remaining>5/8</remaining>\n , \n<remaining>4/8</remaining>\n , \n<remaining>3/8</remaining>\n , \n<remaining>2/8</remaining>\n , \n<remaining>1/8</remaining>\n ) will split the thinking process into 8 parts.)"
        
        add_response = f""

        chunk[i] = head + add_prompt + find_strings + add_response + tail
        
    return chunk


def prompt_v1_prompt(chunk, budget):
    os.environ['budget'] = str(budget)
    print(f"budget = {budget}")
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)'
        add_prompt = f"\n(Complete thinking within {budget} tokens or fewer, please output the remaining number of tokens every 200 tokens to facilitate control of the remaining length of the thinking process, here is a template: 'now remaining tokens: xxx', xxx is the real remaining number of tokens.)"
        add_response = f""

        chunk[i] = head + add_prompt + find_strings + add_response + tail
        
    return chunk


def prompt_v2_prompt(chunk, budget):
    os.environ['budget'] = str(budget)
    print(f"budget = {budget}")
    find_strings = "<｜Assistant｜>"
    for i in range(len(chunk)):
        head = chunk[i].split(find_strings)[0]
        tail = chunk[i].split(find_strings)[1]
        # add_prompt = f'\n(Complete thinking within {budget} tokens or fewer.)'
        add_prompt = f"\n(Complete thinking within {budget} tokens or fewer)"
        add_response = f""

        chunk[i] = head + add_prompt + find_strings + add_response + tail
        
    return chunk


# def solve_final_answer(chunk):
#     k = 0
#     for i in range(len(chunk)):
#         if "**Final Answer**\\boxed" in chunk[i][:-10] and "<｜end▁of▁sentence｜>" not in chunk[i]:
#             chunk[i] += "<｜end▁of▁sentence｜>"
#             k += 1
#     print(f"###added {k} final answer!")
#     return chunk

# import re

def is_balanced(s: str) -> bool:
    """验证大括号是否成对且正确嵌套"""
    stack = 0
    for char in s:
        if char == "{":
            stack += 1
        elif char == "}":
            stack -= 1
            if stack < 0:
                return False
    return stack == 0

def solve_final_answer(chunk: list) -> list:
    
    """处理包含嵌套大括号的答案匹配"""
    
    end_chunk = []
    open_chunk = []
    
    k = 0
    pattern = "**Final Answer**\\boxed{"
    
    for i in range(len(chunk)):
        line = chunk[i]
        if not pattern in line:
            open_chunk.append(chunk[i])
            continue
        start_idx = line.find('**Final Answer**\\boxed{')
        if start_idx == -1:
            open_chunk.append(chunk[i])
            continue
        stack = 1
        end_idx = start_idx + len('**Final Answer**\\boxed{')
        while end_idx < len(line) and stack > 0:
            if line[end_idx] == "{":
                stack += 1
            elif line[end_idx] == "}":
                stack -= 1
            end_idx += 1
        
        # 验证闭合状态
        if stack == 0 and is_balanced(line[start_idx:end_idx]):
            
            chunk[i] += "<｜end▁of▁sentence｜>"
            k += 1
            end_chunk.append(chunk[i])
        else:
            open_chunk.append(chunk[i])

    print(f"### Find {k} anwsers have final answer!")
    return chunk, end_chunk, open_chunk