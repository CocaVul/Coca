from gensim.models import Word2Vec
from typing import List, Dict
import sys
import os
import json
import shutil
import numpy as np
from tqdm import tqdm, trange
from sklearn import metrics
from time import time

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Batch, Data

from detectors.DeepWuKong.model import DeepWuKongModel
from detectors.DeepWuKong.configurations import data_args, model_args, train_args
from global_defines import get_dataloader

class TrainUtil(object):
    def __init__(self, w2v_model: Word2Vec, gnnNets: DeepWuKongModel):
        self.w2v_model: Word2Vec = w2v_model
        self.gnnNets: DeepWuKongModel = gnnNets

        self.train_positive: List[Dict] = json.load(
            open(os.path.join(data_args.dataset_dir, "slice/train_vuls.json"),
                 'r', encoding='utf-8'))
        self.train_negative: List[Dict] = json.load(
            open(os.path.join(data_args.dataset_dir, "slice/train_nors.json"),
                 'r', encoding='utf-8'))
        self.val_positive: List[Dict] = json.load(
            open(os.path.join(data_args.dataset_dir, "slice/val_vuls.json"),
                 'r', encoding='utf-8'))
        self.val_negative: List[Dict] = json.load(
            open(os.path.join(data_args.dataset_dir, "slice/val_nors.json"),
                 'r', encoding='utf-8'))
        self.test_positive: List[Dict] = json.load(
            open(os.path.join(data_args.dataset_dir, "slice/test_vuls.json"),
                 'r', encoding='utf-8'))
        self.test_negative: List[Dict] = json.load(
            open(os.path.join(data_args.dataset_dir, "slice/test_nors.json"),
                 'r', encoding='utf-8'))

    def generate_initial_training_datas(self, data: Dict) -> Data:
        token_seqs = data["line-contents"]
        # token_seqs = [json.loads(node_info)["contents"][0][1] for node_info in data["line-nodes"]]
        n_vs = [np.array([self.w2v_model[word] if word in self.w2v_model.wv.vocab else
                        np.zeros(model_args.vector_size) for word in token_seq.split(" ")]).mean(axis=0) for token_seq in token_seqs]
        t_vs = [torch.FloatTensor(n_v).to(data_args.device) for n_v in n_vs]
        vector = torch.stack(t_vs)
        edges = [json.loads(edge) for edge in data["data-dependences"] + data["control-dependences"]]
        edge_index = torch.LongTensor(edges).to(data_args.device).t()

        return Data(x=vector, edge_index=edge_index, y=data["target"])

    def save_best(self, epoch, eval_f1, is_best):
        print('saving....')
        self.gnnNets.to('cpu')
        state = {
            'net': self.gnnNets.state_dict(),
            'epoch': epoch,
            'f1': eval_f1
        }
        pth_name = f"{model_args.model_dir}/{model_args.model_name}_{model_args.detector}_latest.pth"
        best_pth_name = f'{model_args.model_dir}/{model_args.model_name}_{model_args.detector}_best.pth'
        ckpt_path = os.path.join(model_args.model_dir, pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copy(ckpt_path, os.path.join(model_args.model_dir, best_pth_name))
        self.gnnNets.to(model_args.device)

    def evaluate(self, eval_dataloader, num_batch):
        '''
        :param eval_dataloader: 数据集
        :param gnnNets:分类模型
        :return:
        '''
        acc = []
        recall = []
        precision = []
        f1 = []
        self.gnnNets.eval()
        with torch.no_grad():
            import sys
            for batch_data in tqdm(eval_dataloader, total=num_batch, desc="evaluating process",
                                   file=sys.stdout):
                batch = Batch.from_data_list(batch_data)
                batch.to(data_args.device)
                logits = self.gnnNets(data=batch)

                ## record
                _, prediction = torch.max(logits, -1)
                acc.append(metrics.accuracy_score(batch.y.cpu().numpy(), prediction.cpu().numpy()))
                recall.append(metrics.recall_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0))
                precision.append(
                    metrics.precision_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0))
                f1.append(metrics.f1_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0))

            eval_state = {
                'acc': np.average(acc),
                'recall': np.average(recall),
                'precision': np.average(precision),
                'f1': np.average(f1)
            }

        return eval_state

    def test(self):
        # load model
        checkpoint = torch.load(os.path.join(model_args.model_dir, f'{model_args.model_name}_{model_args.detector}_best.pth'))
        self.gnnNets.load_state_dict(checkpoint['net'])

        # embedding data
        print("start embedding data=================")
        start_time = time()
        test_vul_datas = [self.generate_initial_training_datas(sample) for sample in
                            self.test_positive]
        test_nor_datas = [self.generate_initial_training_datas(sample) for sample in
                            self.test_negative]
        end_time = time()
        spent_time = end_time - start_time
        print(f"embedding spent time: {spent_time // 3600}h-{(spent_time % 3600) // 60}m-{(spent_time % 3600) % 60}s")

        labels = []
        predictions = []

        test_dataloader = get_dataloader(test_vul_datas, test_nor_datas, data_args.batch_size)
        self.gnnNets.eval()
        print('start testing model==================')
        test_start_time = time()
        num_batch = len(test_vul_datas + test_nor_datas) // data_args.batch_size
        with torch.no_grad():
            import sys
            for batch_data in tqdm(test_dataloader, total=num_batch, desc="testing process", file=sys.stdout):
                batch = Batch.from_data_list(batch_data)
                batch.to(model_args.device)
                probs = self.gnnNets(data=batch)

                # record
                _, prediction = torch.max(probs, -1)
                labels.extend(batch.y.cpu().tolist())
                predictions.extend(prediction.cpu().tolist())

        cm = metrics.confusion_matrix(labels, predictions)
        TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        FNR = FN / (TP + FN)
        FPR = FP / (FP + TN)

        test_end_time = time()
        test_spent_time = test_end_time - test_start_time
        print(
            f"training spent time: {test_spent_time // 3600}h-{(test_spent_time % 3600) // 60}m-{(test_spent_time % 3600) % 60}s")

        print(f"Acc: {metrics.accuracy_score(labels, predictions):.3f} |"
              f"Recall: {metrics.recall_score(labels, predictions, zero_division=0):.3f} |"
              f"Precision: {metrics.precision_score(labels, predictions, zero_division=0):.3f} |"
              f"F1: {metrics.f1_score(labels, predictions, zero_division=0):.3f} |"
              f"FNR: {FNR:.3f} |"
              f"FPR: {FPR:.3f} |"
              )

    def train(self):
        print("start embedding data=================")
        start_time = time()
        train_vul_features = [self.generate_initial_training_datas(sample) for sample in
                                 self.train_positive]
        print("==============================")
        train_nor_features = [self.generate_initial_training_datas(sample) for sample in
                                 self.train_negative]
        val_vul_features = [self.generate_initial_training_datas(sample) for sample in
                                 self.val_positive]
        val_nor_features = [self.generate_initial_training_datas(sample) for sample in
                                 self.val_negative]

        path = os.path.join(model_args.model_dir, f'{model_args.model_name}_{model_args.detector}_best.pth')
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.gnnNets.load_state_dict(checkpoint['net'])

        end_time = time()
        spent_time = end_time - start_time
        print(f"embedding spent time: {spent_time // 3600}h-{(spent_time % 3600) // 60}m-{(spent_time % 3600) % 60}s")

        print('start training model==================')
        rate = len(self.train_negative) / len(self.train_positive)
        weight_ce = torch.FloatTensor([1, rate]).to(model_args.device)
        criterion = nn.CrossEntropyLoss(weight=weight_ce)
        optimizer = Adam(self.gnnNets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay,
                         betas=(0.9, 0.999))
        # save path for model
        best_f1 = 0
        early_stop_count = 0

        train_start_time = time()
        num_batch = len(train_vul_features + train_nor_features) // data_args.batch_size
        num_batch_val = len(val_vul_features + val_nor_features) // data_args.batch_size
        for epoch in range(train_args.max_epochs):
            print(f"epoch: {epoch} start===================")
            epoch_start_time = time()
            acc = []
            recall = []
            precision = []
            f1 = []
            loss_list = []
            self.gnnNets.train()
            import sys
            train_dataloader = get_dataloader(train_vul_features, train_nor_features, data_args.batch_size)
            for batch_data in tqdm(train_dataloader, total=num_batch, desc="training process",
                                   file=sys.stdout):
                batch = Batch.from_data_list(batch_data)
                batch.to(data_args.device)
                logits = self.gnnNets(data=batch)
                loss = criterion(logits, batch.y)

                # optimization
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.gnnNets.parameters(), clip_value=2.0)
                optimizer.step()

                ## record
                _, prediction = torch.max(logits, -1)
                loss_list.append(loss.item())
                acc.append(metrics.accuracy_score(batch.y.cpu().numpy(), prediction.cpu().numpy()))
                recall.append(metrics.recall_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0))
                precision.append(
                    metrics.precision_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0))
                f1.append(metrics.f1_score(batch.y.cpu().numpy(), prediction.cpu().numpy(), zero_division=0))
            # report train msg
            print(f"Train Epoch:{epoch}  |Loss: {np.average(loss_list):.3f} | "
                  f"Acc: {np.average(acc):.3f} |"
                  f"Recall: {np.average(recall):.3f} |"
                  f"Precision: {np.average(precision):.3f} |"
                  f"F1: {np.average(f1):.3f}")

            # report eval msg
            batch_dataloader = get_dataloader(val_vul_features, val_nor_features,
                                                   batch_size=data_args.batch_size)
            eval_state = self.evaluate(batch_dataloader, num_batch_val)
            print(
                f"Eval Epoch: {epoch} | Acc: {eval_state['acc']:.3f} | Recall: {eval_state['recall']:.3f} "
                f"| Precision: {eval_state['precision']:.3f} | F1: {eval_state['f1']:.3f}")

            # only save the best model
            is_best = (eval_state['f1'] > best_f1)

            if eval_state['f1'] > best_f1:
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count > train_args.early_stopping:
                break

            if is_best:
                best_f1 = eval_state['f1']
                early_stop_count = 0
            if is_best or epoch % train_args.save_epoch == 0:
                self.save_best(epoch, eval_state['f1'], is_best)

            epoch_end_time = time()
            epoch_spent_time = epoch_end_time - epoch_start_time
            print(
                f"epoch spent time: {epoch_spent_time // 3600}h-{(epoch_spent_time % 3600) // 60}m-{(epoch_spent_time % 3600) % 60}s")

        train_end_time = time()
        train_spent_time = train_end_time - train_start_time
        print(f"training spent time: {train_spent_time // 3600}h-{(train_spent_time % 3600) // 60}m-{(train_spent_time % 3600) % 60}s")
        print(f"The best validation f1 is {best_f1}.")

    def choose_data(self):
        # load model
        checkpoint = torch.load(
            os.path.join(model_args.model_dir, f'{model_args.model_name}_{model_args.detector}_best.pth'))
        self.gnnNets.load_state_dict(checkpoint['net'])

        # embedding data
        print("start embedding data=================")
        start_time = time()
        test_vul_datas = [self.generate_initial_training_datas(sample) for sample in
                          self.test_positive]

        end_time = time()
        spent_time = end_time - start_time
        print(f"embedding spent time: {spent_time // 3600}h-{(spent_time % 3600) // 60}m-{(spent_time % 3600) % 60}s")

        self.gnnNets.eval()
        print('start testing model==================')

        exported_datas = []
        for idx in trange(len(test_vul_datas), desc="generating explain datas", file=sys.stdout):
            if len(self.test_positive[idx]["line-nodes"]) < 10:
                continue
            batch = Batch.from_data_list([test_vul_datas[idx]])
            batch.to(model_args.device)
            probs = self.gnnNets(data=batch)
            # record
            _, prediction = torch.max(probs, -1)
            if prediction.cpu() == 0:  # 过滤掉被错误分类的结点
                continue

            exported_datas.append(self.test_positive[idx])

        print(len(exported_datas))
        json.dump(exported_datas, open(os.path.join(data_args.dataset_dir, "slice/explain_deepwukong.json"), 'w'
                                       , encoding='utf-8'), indent=2)


if __name__ == '__main__':
    pretrain_model = Word2Vec.load(model_args.pretrain_word2vec_model)
    dwk_model: DeepWuKongModel = DeepWuKongModel()
    dwk_model.to(model_args.device)
    train_util: TrainUtil = TrainUtil(pretrain_model, dwk_model)
    train_util.choose_data()
    train_util.train()
    train_util.test()