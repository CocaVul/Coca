import json
import os
from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from detectors.Reveal.configurations import data_args, model_args

src_file = os.path.join(data_args.dataset_dir, 'pretrainCorpus.json')
pretrain_word2vec_model_path = model_args.pretrain_word2vec_model
vector_size = model_args.vector_size
window_size = 10


class PrintStatus(CallbackAny2Vec):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.batch = 0

    def on_epoch_begin(self, model):
        self.epoch += 1
        print(f"epoch {self.epoch} start")

    def on_epoch_end(self, model):
        self.batch = 0
        print(f"epoch {self.epoch} end")

    def on_batch_begin(self, model):
        self.batch += 1
        print(f"epoch {self.epoch} - batch {self.batch} start")


class Sentences:
    def __init__(self):
        self.datas: list = json.load(open(src_file, 'r', encoding='utf-8'))
        print(len(self.datas))

    def __iter__(self):
        for data in self.datas:
            contents = data.split(' ')
            yield contents


if __name__ == '__main__':
    sentences = Sentences()
    model = Word2Vec(size=vector_size, window=window_size, hs=1, min_count=1, workers=4)
    model.build_vocab(sentences)
    model.train(sentences, epochs=20, total_examples=model.corpus_count)
    model.save(pretrain_word2vec_model_path)
    # import numpy as np
    # arr = np.array([np.ones(100), np.zeros(100)])
    # arr = np.concatenate([np.ones(100), np.zeros(20)])
    # print(arr)
    # import torch
    # v = torch.FloatTensor(arr)
    #
    # data = torch.stack([torch.ones(size=(100,)), torch.ones(size=(100,))])
    # print(data.size())
    # 399 batch
