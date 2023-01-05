# PPLM_code_reading
源码阅读，GPT2，PPLM


论文为 [Uber – ICLR 2020 : Plug and Play Language Models: A Simple Approach to Controlled Text Generation](https://arxiv.org/pdf/1912.02164.pdf)

---



# PPLM
[Uber -- ICLR 2020 : Plug and Play Language Models: A Simple Approach to Controlled Text Generation](https://arxiv.org/pdf/1912.02164.pdf)



从标题也可以看出来，是引入了一个**可插拔的组件**对语言模型进行操作，从而修改生成文本。该方法可以直接用于文本生成，不需要微调。


首先假设文本风格/属性为变量 $a$ ，模型生成的文本为 $x$ ，则「模型基于某个属性$a$ ，生成文本 $x$ 」可以表示为：
$$
p(x|a) \propto p(a|x)p(x)
$$
$p(x)$ 为文本的总体分布，即一个形式为 $p(X)=\prod_{i=1}^{n} p\left(x_{i} \mid x_{0}, \cdots, x_{i-1}\right)$ 的 **unconditional LM** ，比如 GPT2。
$p(x|a)$ 为我们想得到的 **conditional LM**，即给定文本特征 $a$ ，该模型可以生成具有该特征的文本。
$p(a|x)$  为额外的一个「**特征分类器/attribute model**」，输入文本，输出对应的风格。这个模型和语言模型相比，要小很多。

模型在生成文本时，每个时间步上都会进行三个步骤：
1. 前向过程：当模型生成一个 token 时，通过已经生成的部分文本 $x$ 得到 $p(a|x)$ ，和其对语言模型的梯度。
2. 反向过程：使用该梯度去更新语言模型的 **隐状态 (internal latent representations of the LM)**，让 $p(a|x)$ 变大。
3. 重采样：重新生成这一个 token

简而言之就是在生成的每个时间步，将历史隐向量往更接近目标风格的方向拉，从而使得新生成的token更符合目标的风格。
每个时间步都会让模型/隐向量进行微小地更新（lr比较小），并且用两个「梯度/目标函数」指导模型，使其生成具有先验文本特征的，流畅的文本，分别对应  $p(a|x)$ 和 $p(x)$ 。

该方法有三个优点：
1. 属性控制强度可控；
2. 多个属性可随意结合；
3. 生成模型、属性模型可选。


## p(a|x)
![](https://blogapi.uber.com/wp-content/uploads/2022/08/image5-1-1.jpg)

文本生成时，每个时间步都有一个隐状态 $H_t$ 和已生成序列 $x_t$ ，模型就是通过这两者生成输出 $o_t$ ，并且更新隐状态得到 $H_{t+1}$ ：
$$
o_{t+1}, H_{t+1}=\operatorname{LM}\left(x_{t}, H_{t}\right) \\
x_{t+1} \sim p_{t+1}=\operatorname{Softmax}\left(W o_{t+1}\right)
$$

而 $p(a|x)$ 在这的作用是，判断当前生成 $x_{t+1}$ 是否接近属性 $a$ 的需求。根据反馈，去修改之前的历史 $H_{t}$ 。
每一个时间步，我们通过 $p(a|x)$ ，即一个判别模型，得到一个梯度，用于更新 $H_{t}$ ：
$$
\tilde{H}_{t}=H_{t}+\Delta H_{t} \\
\widetilde{o}_{t+1}, H_{t+1}=\operatorname{LM}\left(x_{t}, \widetilde{H}_{t}\right)
$$
如果把 $H_{t}$ 看作历史知识，那就是先用历史知识计算 $p(a|x)$ 的损失，然后反过来更新 $H_{t}$ ，得到 $\tilde{H}_{t}$ ，最后用这个新的隐变量生成 token。

<br>

这里就牵扯到如何更新 $H$ 了。

**首先 $H$ 是什么 ？**
论文中因为用的是 GPT2 语言模型，所以模型基本组件是 Transformer，所以历史隐状态 $H_t$ 为 **K-V pair 序列**：
$$
H_{t .}=\left[\left(K_{t}^{(1)}, V_{t}^{(1)}\right), \cdots,\left(K_{t}^{(l)}, V_{t}^{(l)}\right)\right]
$$
是一个序列形式， $(K_{t}^{(i)}, V_{t}^{(i)})$ 表示 第 $i$ 层的所有时间步 $[0,t]$ /所有token 的**K-V pair**。
同时也只有这个变量，控制了下一个 token 的概率分布。
为了提高计算效率，可以选择仅修改最近过去的某个窗口内的隐变量，如红色虚线区域所示。


而如果要用其他模型，也可以根据模型不同改变具体 H 实现。

**怎么求 $\Delta H_{t}$ ？**
$\Delta H_{t}$ 为这一轮「文本隐状态 $H$ 」的改变量，则改变后「文本隐状态」变为 $\Delta H_{t}+H_{t}$ ，那么我们将判别模型 $p(a|x)$ 重写为 $p(a|H)$ （其实有点不严谨，但是也还好， $H_t$ 可以代表 $x_t$ ）。
然后 $\Delta H_{t}$ 梯度更新的公式就是：
$$
\Delta H_{t} \leftarrow \Delta H_{t}+\alpha \frac{\nabla_{\Delta H_{t}} \log p\left(a \mid H_{t}+\Delta H_{t}\right)}{\left\|\nabla_{\Delta H_{t}} \log p\left(a \mid H_{t}+\Delta H_{t}\right)\right\|^{\gamma}}
$$
$\alpha$ 为学习率， $\gamma$ 为标准化的缩放系数， $\Delta H_{t}$ 初始化为0。
这个更新步骤可以重复m次，通常取 [3,10] 。


同时，作者还提到可以通过 $p(a|x)$ 分数的大小，对结果进行 Re-ranking。


## p(x)
如果不使用 $p(x)$ 这个目标的话，模型容易退化，比如生成 “great great great great great”。所以我们期望PLM保持其原先的分布。
有两种方法：
1. 最小化「更新后模型预测分布」和「原始模型预测分布」之间的KL散度，该项超参数设置为 0.01 效果不错。
2. 融合修改前后的单词的输出概率（post-norm fusion），即
$$x_{t+1} \sim \frac{1}{\beta}\left(\tilde{p}_{t+1}^{\gamma_{g m}} p_{t+1}^{1-\gamma_{g m}}\right)$$
其中 ${p}_{t+1}$ 和 $\tilde{p}_{t+1}$ 是修改前后的词库概率分布。
$\gamma_{g m} \rightarrow 1$ 表示收敛于更新后的分布，$\gamma_{g m} \rightarrow 0$ 表示收敛于无条件的语言模型分布。实验发现 $\gamma_{g m} \in [0.8, 0.95]$ 效果不错。


## Attribute models
作者同样给了两种 **判别模型 $p(a|x)$** 的定义

### Bag-of-words

没有额外参数
针对每个主题先总结一批有代表性的词 $\{w_1,w_2...,w_n\}$ ，之后具体实现时只用在每个时间步上对输出概率分布取出对应词袋中词的对数似然分数：
$$
\log p(a \mid x)=\log \left(\sum_{i}^{k} p_{t+1}\left[w_{i}\right]\right)
$$
，然后反向传播就行。方法虽然简单却意外有效。

### Simple discriminators

- trained on a dataset labeled with the desired attributes.
- predicts the target label from the mean of the embedded representation extracted from the original LM model

训练一个分类器，输入是整个序列最后一层所有token的hidden state的avg pooling，输出就是类别数量（详见代码）。

## Coding
仅仅介绍 Simple discriminators ，分两个step：
1. 根据标注文本训练一个 discriminators
2. 使用训练好的 discriminators 做梯度指导

`discriminator`就很简单，就是一个线性分类器。
```
class ClassificationHead(nn.Module):
    def __init__(self, class_size, embed_size):
        super().__init__()
        self.class_size = class_size
        self.embed_size = embed_size  # gpt2model.transformer.config.hidden_size
        self.mlp = nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        logits = self.mlp(hidden_state)
        return logits
```

对每个sequence 来说，需要经过 `avg_representation`方法，得到 `discriminator` 的输入：
```
def avg_representation(self, x): # x 为 input_ids

    # ne(not equal)函数得到 mask矩阵 -> [batch_size, max_seq_len, hidden_state]
    mask = x.ne(0).unsqueeze(2).repeat(1, 1, self.embed_size).float().to(self.device).detach()

    # gpt2 编码，得到最后一层的隐状态 -> [batch_size, max_seq_len, hidden_state]
    hidden = self.encoder.transformer(x)["last_hidden_state"]

    # 使用 mask 矩阵进行 mask
    masked_hidden = hidden * mask

    # avg mean: 每句所有 token 的 avg  -> batch_size, hidden_state
    avg_hidden = torch.sum(masked_hidden, dim=1) / (torch.sum(mask, dim=1).detach() + EPSILON)
    return avg_hidden
```

<br>

接下来看主模型 PPLM 每一个时间步如何修改隐状态，
`length`为我们想让模型生成文本的长度，这里有一个`for i in trange(length, ascii=True)`，即每个时间步都进行修改：
1. 更新变量
2. 调用`generate_text_pplm()`方法得到根据 discriminator 更新后的隐状态`pert_past`
3. 根据新的隐状态`pert_past`和last token重新得到模型输出`pert_logits`
4. 根据输出，得到字典上的概率分布 `pert_probs`，这里有个小 trick
5. 融合修改前后的单词的输出概率（post-norm fusion）
6. 得到 token `last`，更新到 `output_so_far` 中


```
for i in trange(length, ascii=True):
    # 第一次输入，没有提供给 past
    if past is None and output_so_far is not None:  # 没有提供 past 序列，即 past_key_values
        last = output_so_far[:, -1:]  # 最后一个 token
        if output_so_far.shape[1] > 1:
            past = model(output_so_far[:, :-1])["past_key_values"]  # 最后一个token之前的序列的
            # layer_num * 2【key, value】 * ( batch_size, num_head, sql_len, head_features )

    lm_output = model(output_so_far)  # 原始序列喂给模型
    unpert_logits, unpert_past, unpert_all_hidden = (
        lm_output["logits"],
        lm_output["past_key_values"],
        lm_output["hidden_states"],
    )
    unpert_last_hidden = unpert_all_hidden[-1]  # 原始序列，对应输出的最后一个 token 的 hidden state

    # 修改隐藏状态，即论文的 h
    pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    model,
                    past,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    classifier=classifier,
                    class_label=class_label,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    ...
                )

    lm_output = model(last, past_key_values=pert_past)  # 整个序列全放入，预测下一个
    pert_logits, past = (
        lm_output["logits"],
        lm_output["past_key_values"],
    )
    pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST  # 最后一个 token 的 logit

    # Trick
    for token_idx in set(output_so_far[0].tolist()):
        if pert_logits[0, token_idx] < 0:  # 和前面的 token 相比，如果 相似度小于 0
            pert_logits[0, token_idx] *= repetition_penalty  # 乘法
        else:
            pert_logits[0, token_idx] /= repetition_penalty  # 不希望预测出来重复的 token

    pert_probs = nn.functional.softmax(pert_logits, dim=-1)

    # 得到概率分布
    # 融合修改前后的单词的输出概率（post-norm fusion）
    if perturb:
        unpert_probs = nn.functional.softmax(unpert_logits[:, -1, :], dim=-1)

        pert_probs = (pert_probs ** gm_scale) * (unpert_probs ** (1 - gm_scale))  # + SMALL_CONST
        pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)  # + SMALL_CONST

    else:
        pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
        pert_probs = nn.functional.softmax(pert_logits, dim=-1)

    # 根据概率分布得到下一个token ： sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)
        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

    # update context/output_so_far appending the new token
    output_so_far = last if output_so_far is None else torch.cat((output_so_far, last), dim=1)
```
重要的变量就是：
- `past`: 最后一个token之前序列的 past_key_values
- `last`: 最后一个 token
- `unpert_past`: 全序列的 past_key_values
- `unpert_logits`: 全序列的 logits(vocab_size)

然后使用`perturb_past()`方法修改隐状态得到`pert_past`，使用`pert_past`和`last`得到输出 token。过程详见 **section:perturb_past**。
这里有个小 trick，用来放缩相似度，希望预测出来重复的 token：得到 `pert_logits` 之后，如果这个token 和序列中已经出现的 token 相似（相似度 > 0），则让分数减少，反之分数变大。

然后就是融合修改前后的生成token的输出概率，修改前的为 `unpert_probs`，修改后的为`pert_probs`，这里使用超参数 `gm_scale`。

然后 top-k-sample，见下面的 **section：top-k-sample**，得到token，更新 `output_so_far`。

## perturb_past
```
def perturb_past(
        past,  # seq_len-1  的 ["past_key_values"]
        model,
        last,  # 最后一个 token id
        unpert_past=None,  # 全序列的 [past_key_values]
        unpert_logits=None,  # 全序列的 logits(vocab_size)
        accumulated_hidden=None,  # 最后一层的所有 hidden state 求和
        classifier=None,
        class_label=None,
        num_iterations=3, # 更新多少次
        horizon_length=1,
        window_length=0,  # 只改变前面多少个 token 的 hidden state
        ...
):
    # 自定义：用于将 past 的 tuple 结构转换为 list
    def listit(t):
        return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

    # 全 0 复制 past，这里论文提到过，初始化时为 0
    grad_accumulator = [(np.zeros(torch.vstack(p).shape).astype("float32")) for p in listit(past)]

    # 根据 Window size 设置 Window mask
    if curr_length > window_length and window_length > 0:
        ...

    # 隐状态的迭代数次
    for i in range(num_iterations):

        # TODO：得到 perturbed_past，即前面的 past 状态，也就是我们的目标更新量/grad
        # 将 grad_accumulator 设置为梯度
        curr_perturbation = [torch.from_numpy(p_).requires_grad_(True).to(device=device) for p_ in grad_accumulator]

        # 对应公式，确实是相加
        perturbed_past = list(map(add, past, curr_perturbation))

        # TODO：使用更新过的 perturbed_past 得到输出
        lm_output = model(last, past_key_values=perturbed_past)

        # 得到每个token的点积相似度 和  hidden  state
        all_logits, all_hidden = lm_output["logits"], lm_output["hidden_states"]

        # 求 Sum -> [batch_size, hidden_size]
        # accumulated_hidden 也是一个缓存量，就不用每次都重新求和然后avg了，只要加最后一个重新 avg 就行
        hidden = all_hidden[-1] # 最后一层的hidden state,
        # 这里没看懂为什么要加 accumulated_hidden
        new_accumulated_hidden = accumulated_hidden + torch.sum(hidden, dim=1).detach()

        # 这里是为了 KL 散度用的，求最后一个token 的点积相似度 -> [batch_size, 1, vocab_size]
        logits = all_logits[:, -1, :]
        probs = nn.functional.softmax(logits, dim=-1)

        # TODO: 求 loss （ 这里 loss_type == 2  表示使用 classifier
        if loss_type == 2:

            prediction = classifier(new_accumulated_hidden / (curr_length + 1))  # avg
            label = torch.tensor(prediction.shape[0] * [class_label], device=device, dtype=torch.long)
            loss = ce_loss(prediction, label)

        # 梯度下降
        loss.backward()

        # TODO: Within window mask
        #   calculate within-window gradient norms
        grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST) for index, p_ in enumerate(curr_perturbation)
            ]
        # normalize gradients
        grad = [
            -stepsize * (p_.grad * window_mask / grad_norms[index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # TODO: Re-initialize
        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [torch.from_numpy(p_).requires_grad_(True).to(device=device) for p_ in grad_accumulator]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter
```

这里只介绍最base的版本，此外源码还加了
1. window mask，论文中提到了
2. KL 散度约束，论文中提到了
3. horizon_length，反正都是求序列所有token的avg pooling，这里允许模型连续预测个 token，论文中讲的是一个 token。

更详细的可以参考上面的 github 代码文件。



### top-k-sample
```
def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins, torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -BIG_CONST, logits)
```
