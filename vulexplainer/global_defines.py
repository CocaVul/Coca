import random
import torch
random.seed(2)

vul_types = [
    "buffer overflow",
    "incorrect calculation",
    "memory leak",
    "path traversal"
]

cur_vul_type_idx = 2 # 目前处理的漏洞类型，值为0-3
vul_type = vul_types[cur_vul_type_idx]
cur_dir = "D:/Program Files/PythonProjects/VulDetectors-master"
sparsity_value = 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
num_classes = 2


# 传入的每个sample应为Tuple[int, List[Data], torch.LongTensor]
def get_dataloader(positive_samples, negative_samples, batch_size: int):
    all_samples = positive_samples + negative_samples
    random.shuffle(all_samples)

    num_batch = len(all_samples) // batch_size
    if len(all_samples) % batch_size != 0:
        num_batch += 1

    for i in range(num_batch):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(all_samples))

        yield all_samples[start_idx: end_idx]