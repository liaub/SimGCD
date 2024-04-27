import pytorch_lightning as pl
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import numpy as np
# class Evaluation_TrainDataset(Dataset):
#     def __init__(self, configs, tokenizer, trainsets, train_comms, nx_graph, domain_hop):
#         self.configs = configs
#         self.trainsets = trainsets
#         self.train_comms = train_comms
#         self.nx_graph = nx_graph
#         self.domain_hop = domain_hop
#         self.tokenizer = tokenizer
#
#     def __len__(self):
#         return len(self.trainsets)
#
#     def __getitem__(self, index):
#         input = self.trainsets[index]
#         iteration = 0
#         inner_neis = []
#         nstore = [input]
#         while True:
#             if iteration >= self.configs.domain_margin:
#                 break
#             store = []
#             for node in nstore:
#                 for FNs in list(self.nx_graph.neighbors(node)):  # find 1_th neighbors
#                     inner_neis.append(FNs)
#                     store.append(FNs)
#             nstore = store
#             iteration += 1
#
#
#         pos_comms = []
#         neg_comms = []
#         for comms in self.train_comms:
#             if input in comms:
#                 pos_comms = pos_comms + comms
#             else:
#                 neg_comms = neg_comms + comms
#
#         pos_comms = list(set(pos_comms))
#         neg_comms = list(set(neg_comms))
#         pos_comms.remove(input)
#         # 正样本的内外域划分
#         inner_pos = []
#         external_pos = []
#         for pc in pos_comms:
#             if pc in inner_neis:
#                 inner_pos.append(pc)
#             else:
#                 external_pos.append(pc)
#         # 负样本的内外域划分
#         inner_neg = []
#         external_neg = []
#         for nc in neg_comms:
#             if nc in inner_neis:
#                 inner_neg.append(nc)
#             else:
#                 external_neg.append(nc)
#
#         out = {
#             'input': input,
#             'inner_pos': inner_pos,
#             'external_pos': external_pos,
#             'inner_neg': inner_neg,
#             'external_neg': external_neg
#         }
#
#         return out
#
#     def collate_fn(self, data):
#         agg_data = dict()
#         agg_data['input'] = [dt['input'] for dt in data]
#         agg_data['inner_pos'] = [dt['inner_pos'] for dt in data]
#         agg_data['external_pos'] = [dt['external_pos'] for dt in data]
#         agg_data['inner_neg'] = [dt['inner_neg'] for dt in data]
#         agg_data['external_neg'] = [dt['external_neg'] for dt in data]
#         return agg_data

class Evaluation_TrainDataset(Dataset):
    def __init__(self, configs, tokenizer, trainsets, train_comms, nx_graph, domain_hop):
        self.configs = configs
        self.trainsets = trainsets
        self.train_comms = train_comms
        self.nx_graph = nx_graph
        self.domain_hop = domain_hop
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.trainsets)

    def __getitem__(self, index):
        input = self.trainsets[index]
        pos_comms = []
        neg_comms = []
        for comms in self.train_comms:
            if input in comms:
                pos_comms = pos_comms + comms
            else:
                neg_comms = neg_comms + comms

        pos_comms = list(set(pos_comms))
        neg_comms = list(set(neg_comms))
        pos_comms.remove(input)
        # new_pos_comms = []
        # for comm in pos_comms:
        #     path_length = nx.shortest_path_length(self.nx_graph, source=input, target=comm)
        #     if path_length <= self.domain_hop:
        #         new_pos_comms.append(comm)
        new_neg_comms = []
        no_connect = []
        for comm in neg_comms:
            connection = True
            try:
                path_length = nx.shortest_path_length(self.nx_graph, source=input, target=comm)
            except Exception as e:
                connection = False
            if connection:
                new_neg_comms.append(comm)
            else:
                no_connect.append(comm)

        if len(new_neg_comms) < len(pos_comms):
            num = len(pos_comms) - len(new_neg_comms)
            neg_idx = list(range(len(no_connect)))
            weights = np.ones(len(no_connect)) / len(no_connect)
            sampling_ids = np.random.choice(a=neg_idx, size=num, p=weights, replace=False)
            neg_data = np.array(no_connect)[sampling_ids]
            new_neg_comms = new_neg_comms + neg_data.tolist()
        out = {
            'input': input,
            'pos_comms': pos_comms,
            'neg_comms': new_neg_comms
        }

        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['input'] = [dt['input'] for dt in data]
        agg_data['pos_comms'] = [dt['pos_comms'] for dt in data]
        agg_data['neg_comms'] = [dt['neg_comms'] for dt in data]
        return agg_data

class Evaluation_TestDataset(Dataset):
    def __init__(self, configs, tokenizer, testsets, test_comms, nx_graph, domain_hop):  # mode: {tail, head}
        self.configs = configs
        self.testsets = testsets
        self.test_comms = test_comms
        self.nx_graph = nx_graph
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.testsets)

    def __getitem__(self, index):
        input = self.testsets[index]
        out = {
            'input': input
        }

        return out

    def collate_fn(self, data):
        agg_data = dict()
        agg_data['input'] = [dt['input'] for dt in data]
        return agg_data

class EvaluationDataModule(pl.LightningDataModule):
    def __init__(self, configs, trainsets, validsets, testsets, train_comms, test_comms, nx_graph, domain_hop, running_model='train_model'):
        super().__init__()
        self.configs = configs
        self.trainsets = trainsets
        self.validsets = validsets
        self.testsets = testsets
        self.train_comms = train_comms
        self.test_comms = test_comms
        self.nx_graph = nx_graph
        self.domain_hop = domain_hop
        # ent_name_list, rel_name_list .type: list
        self.running_model = running_model


        self.tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
        self.train_both = None
        self.valid_tail, self.valid_head = None, None
        self.test_tail, self.test_head = None, None

    def prepare_data(self):
        self.train_both = Evaluation_TrainDataset(self.configs, self.tokenizer, self.trainsets, self.train_comms, self.nx_graph, self.domain_hop)
        self.valid_both = Evaluation_TestDataset(self.configs, self.tokenizer, self.validsets, self.test_comms, self.nx_graph, self.domain_hop)
        self.test_both = Evaluation_TestDataset(self.configs, self.tokenizer, self.testsets, self.test_comms,self.nx_graph, self.domain_hop)


    def train_dataloader(self):

        train_loader = DataLoader(self.train_both,
                                  batch_size=self.configs.batch_size,
                                  shuffle=True,
                                  collate_fn=self.train_both.collate_fn,
                                  pin_memory=True,
                                  num_workers=self.configs.num_workers)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_both,
                                       batch_size=self.configs.val_batch_size,
                                       shuffle=False,
                                       collate_fn=self.valid_both.collate_fn,
                                       pin_memory=True,
                                       num_workers=self.configs.num_workers)

        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_both,
                                      batch_size=self.configs.val_batch_size,
                                      shuffle=True,
                                      collate_fn=self.test_both.collate_fn,
                                      pin_memory=True,
                                      num_workers=self.configs.num_workers)

        return test_loader
