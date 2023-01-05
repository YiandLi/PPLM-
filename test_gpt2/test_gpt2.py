import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_text = "check if it can"
inputs = torch.tensor([tokenizer.encode(input_text)])
print(len(inputs[0]))
print(inputs)

generated = []
past_key_values = None
for i in range(10):
    # context 为每次迭代输入模型中的input_ids张量, 是一个字符或者序列
    # past_key_values 是历史 key/val 缓存
    output = model(inputs, past_key_values=past_key_values)
    past_key_values = output.past_key_values  # 这里 past_key_values 已经更新过了
    
    # 此时获取GPT2模型计算的输出结果hidden_states张量中第二维度最后一个元素的argmax值, 得出的argmax值即为此次GPT2模型迭代
    # 计算生成的下一个token. 注意, 此时若是第一次迭代, 输出结果hidden_states张量的形状为(batch_size, sel_len, n_state);
    # 此时若是第二次及之后的迭代, 输出结果hidden_states张量的形状为(batch_size, 1, n_state), all_head_size=n_state=nx=768.
    token = torch.argmax(output.logits[..., -1, :])
    
    # 将本次迭代生成的token的张量变为二维张量, 以作为下一次GPT2模型迭代计算的上下文context.
    context = token.unsqueeze(0)
    
    # 将本次迭代计算生成的token的序列索引变为列表存入generated
    generated += [token.tolist()]

# 将generated中所有的token的索引转化为token字符.
sequence = tokenizer.decode(generated)
sequence = sequence.split(".")[:-1]
print(sequence)
