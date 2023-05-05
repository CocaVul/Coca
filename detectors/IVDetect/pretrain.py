import json
import os
from typing import List
from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from detectors.IVDetect.configurations import data_args, model_args

# IVDetect pretrain

from CppCodeAnalyzer.extraTools.vuldetect.ivdetect import lexical_parse


src_file = os.path.join(data_args.dataset_dir, 'pretrainCorpus.json')
pretrain_word2vec_model_path = model_args.pretrain_word2vec_model
vector_size = model_args.feature_representation_size
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
        self.datas: List[List[str]] = [lexical_parse(data) for data in json.load(open(src_file, 'r', encoding='utf-8'))]
        print(len(self.datas))

    def __iter__(self):
        for data in self.datas:
            yield data

if __name__ == '__main__':
    sentences = Sentences()
    model = Word2Vec(size=vector_size, window=window_size, hs=1, min_count=1, workers=4)
    model.build_vocab(sentences)
    model.train(sentences, epochs=20, total_examples=model.corpus_count)
    model.save(pretrain_word2vec_model_path)