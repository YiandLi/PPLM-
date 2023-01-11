import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

import random


def select_top_k(predictions, k=10):
    """
    概率最大的前k个token中进行随机采样
    """
    predicted_index = random.choice(
        predictions[0, -1, :].sort(descending=True)[1][:10]).item()
    return torch.tensor(predicted_index)


input_text = "You can use the Google Pay app for fast"
inputs = torch.tensor([tokenizer.encode(input_text)])
print(len(inputs[0]))
print(inputs)

generated = []
past_key_values = None
total_text = input_text
for i in range(100):
    # context 为每次迭代输入模型中的input_ids张量, 是一个字符或者序列
    # past_key_values 是历史 key/val 缓存
    output = model(inputs, past_key_values=past_key_values)
    past_key_values = output.past_key_values  # 这里 past_key_values 已经更新过了
    
    # 此时获取GPT2模型计算的输出结果hidden_states张量中第二维度最后一个元素的argmax值, 得出的argmax值即为此次GPT2模型迭代
    # 计算生成的下一个token. 注意, 此时若是第一次迭代, 输出结果hidden_states张量的形状为(batch_size, sel_len, n_state);
    # 此时若是第二次及之后的迭代, 输出结果hidden_states张量的形状为(batch_size, 1, n_state), all_head_size=n_state=nx=768.
    
    # 最大概率采样
    token = torch.argmax(output.logits[..., -1, :])
    
    # 随机采样
    # token = select_top_k(output.logits)  # 维度为 1
    
    # 将本次迭代生成的token的张量变为三维张量【batch_size, 1】, 以作为下一次GPT2模型迭代计算的上下文context.
    inputs = token.unsqueeze(0).unsqueeze(0)
    total_text += tokenizer.decode(token)
    
    if '<|endoftext|>' in total_text:
        total_text = ". ".join(total_text.split(".")[:-1])
        # 如果出现文本结束标志，就结束文本生成
        break

print(total_text)
