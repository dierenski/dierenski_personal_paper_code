# dierenski_personal_paper_code
论文复现代码集

## findpo-reproduction/
**FinDPO** (Iacovides et al., ICAIF'25, arXiv:2507.18417) 复现：用 DPO + (Q)LoRA 对齐
Llama-3-8B-Instruct 做金融情绪分析，含分类 F1 基准、logit-to-score + 温度标定、以及与数据集无关的
多空组合回测。默认配置面向 16GB 消费级 GPU（4-bit QLoRA）。详见 [findpo-reproduction/](findpo-reproduction/)。
