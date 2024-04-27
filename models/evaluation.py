import os
import re
import sqlite3
import random
import pickle
import networkx as nx
import numpy as np
import torch
import time
from multiprocessing.pool import ThreadPool
import torch.nn as nn
from models.gnn import GNNEncoder
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import f1_score, accuracy_score, recall_score
from models.modified_model.modified_T5 import ModifiedT5ForConditionalGeneration
from transformers.optimization import Adafactor
from script.helper_funcs import prepare_locator_train_data, generate_ego_net
from script.metrics import eval_scores


class EvalutionFinetuner(pl.LightningModule):
    def __init__(self, configs, tokenizer, nx_graph, graph_data, statis_value, random_seeds, domain_hop, cuda, test_comms):
        super().__init__()
        self.save_hyperparameters()
        self.configs = configs
        self.tokenizer = tokenizer
        self.nx_graph = nx_graph
        self.graph_data = graph_data
        self.statis_value = statis_value
        self.domain_hop = domain_hop
        self.T5ForConditionalGeneration = ModifiedT5ForConditionalGeneration.from_pretrained(configs.pretrained_model)
        self.history = {'perf': ..., 'loss': []}
        self.num_node, input_dim = graph_data.x.size(0), graph_data.x.size(1)
        self.model_dim = self.T5ForConditionalGeneration.model_dim
        self.gnn_encoder = GNNEncoder(input_dim, self.model_dim, self.model_dim, configs.n_layers,
                                      gnn_type=configs.gnn_type)
        self.random_seeds = np.array(random_seeds)
        self.buckets = []
        self.test_comms = test_comms
        self.cuda = cuda


    def collate_fn(self, data, flag):
        agg_data = dict()
        agg_data['source_ids'] = self.batchify(data, 'source_ids', padding_value=0)
        agg_data['source_mask'] = self.batchify(data, 'source_mask', padding_value=0)
        if flag != 'test':
            agg_data['target_ids'] = self.batchify(data, 'target_ids', padding_value=0)
            agg_data['target_mask'] = self.batchify(data, 'target_mask', padding_value=0)
        return agg_data

    def batchify(self, output_dict, key, padding_value=None, return_list=False):
        tensor_out = [out[key] for out in output_dict]
        if return_list:
            return tensor_out
        if not isinstance(tensor_out[0], torch.LongTensor) and not isinstance(tensor_out[0], torch.FloatTensor):
            tensor_out = [torch.LongTensor(value) for value in tensor_out]
        if padding_value is None:
            tensor_out = torch.stack(tensor_out, dim=0)
        else:
            tensor_out = pad_sequence(tensor_out, batch_first=True, padding_value=padding_value)
        return tensor_out

    def sample_construction(self,  node, bucket, out, flag='pos'):
        # 正样本构建 <extra_id_2>=32097
        input_text = 'predict correlation: ('
        # for idx, entity in enumerate(bucket):
        #     if idx + 1 == len(bucket):
        #         input_text += str(entity) + ' <extra_id_2>) | '
        #     else:
        #         input_text += str(entity) + ' <extra_id_2>, '

        for idx, entity in enumerate(bucket):
            if idx + 1 == len(bucket):
                input_text += str(entity) + ') <extra_id_2> | '
            else:
                input_text += str(entity) + ', '

        input_text += "({ch}) <extra_id_2>".format(ch=str(node))
        if flag == 'pos':
            target_text = '<extra_id_0>yes<extra_id_1>'
        elif flag == 'neg':
            target_text = '<extra_id_0>no<extra_id_1>'
        else:
            target_text = ''
        tokenized_src = self.tokenizer(input_text, max_length=self.configs.src_max_length, truncation=True)
        source_ids = tokenized_src.input_ids
        source_mask = tokenized_src.attention_mask
        if flag == 'test':
            out.append({'source_ids': source_ids, 'source_mask': source_mask})
        else:
            tokenized_tgt = self.tokenizer(target_text, max_length=self.configs.train_tgt_max_length, truncation=True)
            target_ids = tokenized_tgt.input_ids
            target_mask = tokenized_tgt.attention_mask
            out.append({'source_ids': source_ids, 'source_mask': source_mask, 'target_ids': target_ids,
                        'target_mask': target_mask})
        return out


    def training_step(self, batched_data, batch_idx):
        # src_ids, src_mask: .shape: (batch_size, padded_seq_len)
        subg_max_size = 20
        num_hop = 2
        node_list = batched_data['input']
        pos_comms = batched_data['pos_comms']
        neg_comms = batched_data['neg_comms']
        out = []
        batch_data, batch_index = prepare_locator_train_data(node_list, self.graph_data, max_size=subg_max_size,
                                                num_hop=num_hop)
        batch_data = batch_data.to(self.cuda)
        _, node_emb = self.gnn_encoder(batch_data.x, batch_data.edge_index, batch_data.batch, batch_index)
        parameters = torch.Tensor([])
        # 采样过程
        for idx, node in enumerate(node_list):
            #TODO 正采样
            comm = np.array(pos_comms[idx])
            pos_seeds = []
            pos_index = []
            pos_sum = 0
            for jdx, com in enumerate(comm):
                seed = self.random_seeds[com]
                pos_seeds.append(seed)
                pos_index.append(jdx)
                pos_sum = pos_sum + seed
            # sampling_num = random.choice(list(range(1, len(comm)+1)))
            sampling_num = random.choice(list(range(1, self.statis_value)))
            if self.statis_value < len(comm):
                weights = pos_seeds / pos_sum
                sampling_ids = np.random.choice(a=pos_index, size=sampling_num, p=weights, replace=False)
                pos_data = comm[sampling_ids]
            else:
                pos_data = comm
                sampling_num = len(pos_data)
            # self.random_seeds[pos_data] = self.random_seeds[pos_data]-1

            #TODO 负采样
            comm = np.array(neg_comms[idx])
            neg_seeds = []
            neg_index = []
            neg_sum = 0
            for jdx, com in enumerate(comm):
                seed = self.random_seeds[com]
                neg_seeds.append(seed)
                neg_index.append(jdx)
                neg_sum = neg_sum + seed


            if sampling_num < len(comm):
                weights = neg_seeds / neg_sum
                sampling_ids = np.random.choice(a=neg_index, size=sampling_num, p=weights, replace=False)
                neg_data = comm[sampling_ids]
            else:
                neg_data = comm
            # weights = neg_seeds / neg_sum
            # sampling_ids = np.random.choice(a=neg_index, size=sampling_num, p=weights, replace=False)
            # neg_data = comm[sampling_ids]
            # self.random_seeds[neg_data] = self.random_seeds[neg_data] - 1

            #TODO 获取正负样本的表示
            batch_pos_data, batch_pos_index = prepare_locator_train_data(pos_data.tolist(), self.graph_data, max_size=subg_max_size,
                                                    num_hop=num_hop)
            batch_pos_data = batch_pos_data.to(self.cuda)
            _, pos_emb = self.gnn_encoder(batch_pos_data.x, batch_pos_data.edge_index, batch_pos_data.batch, batch_pos_index)
            pos_emb = torch.mean(pos_emb, dim=0).unsqueeze(0)

            batch_neg_data, batch_neg_index = prepare_locator_train_data(neg_data.tolist(), self.graph_data, max_size=subg_max_size,
                                                        num_hop=num_hop)
            batch_neg_data = batch_neg_data.to(self.cuda)
            _, neg_emb = self.gnn_encoder(batch_neg_data.x, batch_neg_data.edge_index, batch_neg_data.batch, batch_neg_index)
            neg_emb = torch.mean(neg_emb, dim=0).unsqueeze(0)

            #TODO 正负训练样本构建
            out = self.sample_construction(node, pos_data, out, 'pos')
            out = self.sample_construction(node, neg_data, out, 'neg')


            #TODO 构建实体属性
            if idx == 0:
                parameters = pos_emb
            else:
                parameters = torch.cat([parameters, pos_emb], dim=0)
            parameters = torch.cat([parameters, node_emb[idx].unsqueeze(0)], dim=0)
            parameters = torch.cat([parameters, neg_emb], dim=0)
            parameters = torch.cat([parameters, node_emb[idx].unsqueeze(0)], dim=0)


        agg_data = self.collate_fn(out, 'train')

        # target_ids, target_mask, labels: .shape: (batch_size, padded_seq_len)
        source_ids = agg_data['source_ids'].to(self.cuda)
        source_mask = agg_data['source_mask'].to(self.cuda)
        target_ids = agg_data['target_ids'].to(self.cuda)
        labels = target_ids.clone()
        labels[labels[:, :] == self.trainer.datamodule.tokenizer.pad_token_id] = -100

        # ent_rel .shape: (batch_size, 2)
        source_emb = self.T5ForConditionalGeneration.encoder.embed_tokens(source_ids)
        if self.configs.graph_embedding:
            batch_size, seq_len, model_dim = source_emb.shape
            source_emb = source_emb.view(-1, self.model_dim)  # (batch_size * seq_len, model_dim)
            source_ids = source_ids.view(-1)  # (batch_size * seq_len)
            source_ids = torch.where(source_ids == 32097)
            source_emb[source_ids] = parameters
            source_emb = source_emb.view(batch_size, -1, model_dim)
        # batch_size, seq_len, model_dim = inputs_emb.shape
        output = self.T5ForConditionalGeneration(inputs_embeds=source_emb, attention_mask=source_mask, labels=labels, output_hidden_states=True)
        if self.configs.train_style == 1:
            # 方案一：Use only generated loss
            loss = torch.mean(output.loss)
        elif self.configs.train_style == 2:
            # 方案二：Logarithmic contrast loss
            gloss = torch.mean(output.loss)
            logits = output.logits
            probs = F.softmax(logits, dim=-1)
            # 正样本的loss构建
            yes_token_id = self.tokenizer.encode("yes")[0]
            yes_score = probs[:, 1, yes_token_id]
            yprob = yes_score[torch.where(labels[:, 1] == 4273)]  # 4273->yes
            nprob = yes_score[torch.where(labels[:, 1] != 4273)]  # 150->no
            yloss = 0
            count = 0
            for npb in nprob:
                for ypb in yprob:
                    count += 1
                    yloss += torch.log(1 + torch.exp((npb - ypb) * self.configs.lamda))
            yloss = yloss / count
            # 负样本的loss构建
            no_token_id = self.tokenizer.encode("no")[0]
            no_score = probs[:, 1, no_token_id]
            yprob = no_score[torch.where(labels[:, 1] == 150)]  # 4273->yes
            nprob = no_score[torch.where(labels[:, 1] != 150)]  # 150->no
            nloss = 0
            count = 0
            for npb in nprob:
                for ypb in yprob:
                    count += 1
                    nloss += torch.log(1 + torch.exp((npb - ypb) * self.configs.lamda))
            nloss = nloss / count
            loss = yloss + nloss + gloss

            # loss = yloss + gloss
        else:
            # 方案三：Max contrast loss
            gloss = torch.mean(output.loss)
            logits = output.logits
            probs = F.softmax(logits, dim=-1)
            yes_token_id = self.tokenizer.encode("yes")[0]
            yes_score = probs[:, 1, yes_token_id]
            yprob = yes_score[torch.where(labels[:, 1] == 4273)]  # 4273->yes
            nprob = yes_score[torch.where(labels[:, 1] != 4273)]  # 150->no
            yloss = 0
            count = 0
            for npb in nprob:
                for ypb in yprob:
                    count += 1
                    margin = 1
                    yloss = torch.max(torch.tensor([0, npb - ypb + margin]))
            yloss = yloss / count

            loss = yloss + gloss
        self.history['loss'].append(loss.detach().item())
        self.log('val_loss', loss, on_step=True)
        return {'loss': loss}

    # def validation_step(self, batched_data, batch_idx):
    #     if self.current_epoch < self.configs.skip_n_val_epoch:
    #         return
    #     subg_max_size = 20
    #     num_hop = 2
    #     node_list = batched_data['input']
    #     # 用第一个实体新建一个bucket
    #     if batch_idx == 0:
    #         self.buckets.append([node_list[0]])
    #         node_list.pop(0)
    #
    #     batch_data, batch_index = prepare_locator_train_data(node_list, self.graph_data, max_size=subg_max_size,
    #                                             num_hop=num_hop)
    #     batch_data = batch_data.to(self.cuda)
    #     _, node_emb = self.gnn_encoder(batch_data.x, batch_data.edge_index, batch_data.batch, batch_index)
    #
    #     for idx, node in enumerate(node_list):
    #         out = []
    #         parameters = torch.Tensor([])
    #         reason_status = []
    #         for bix, bucket in enumerate(self.buckets):
    #             # 提前判断桶中的信息
    #             reason_count = 0
    #             for btn in bucket:
    #                 try:
    #                     path_length = nx.shortest_path_length(self.nx_graph, source=node, target=btn)
    #                 except Exception as e:
    #                     path_length = 100
    #
    #                 if path_length <= self.domain_hop:
    #                     reason_count = reason_count + 1
    #                 elif path_length == 100:
    #                     reason_count = reason_count -100
    #                 else:
    #                     reason_count = reason_count + 0
    #
    #             reason_radio = reason_count / len(bucket)
    #             if reason_radio <= 0.8:
    #                 reason_status.append(0)
    #             else:
    #                 reason_status.append(1)
    #
    #             if len(bucket) > self.statis_value:
    #                 sampling_num = random.choice(list(range(1, self.statis_value)))
    #                 weights = np.ones(len(bucket)) / len(bucket)
    #                 bucket = np.random.choice(a=bucket, size=sampling_num, p=weights, replace=False).tolist()
    #             bucket_data, bucket_index = prepare_locator_train_data(bucket, self.graph_data, max_size=subg_max_size,
    #                                                         num_hop=num_hop)
    #             bucket_data = bucket_data.to(self.cuda)
    #             _, bucket_emb = self.gnn_encoder(bucket_data.x, bucket_data.edge_index, bucket_data.batch, bucket_index)
    #             bucket_emb = torch.mean(bucket_emb, dim=0).unsqueeze(0)
    #             out = self.sample_construction(node, bucket, out, 'test')
    #             # TODO 构建实体属性
    #             if bix == 0:
    #                 parameters = bucket_emb
    #             else:
    #                 parameters = torch.cat([parameters, bucket_emb], dim=0)
    #             parameters = torch.cat([parameters, node_emb[idx].unsqueeze(0)], dim=0)
    #
    #         agg_data = self.collate_fn(out, 'test')
    #         source_ids = agg_data['source_ids'].to(self.cuda)
    #         source_mask = agg_data['source_mask'].to(self.cuda)
    #         results = self.decode(source_ids, source_mask, parameters)
    #         # results = ['no']
    #         if 'yes' in results:
    #             status = True
    #             for i, result in enumerate(results):
    #                 if 'yes' == result and reason_status[i] == 1:
    #                     self.buckets[i].append(node)
    #                     status = False
    #             if status:
    #                 self.buckets.append([node])
    #         else:
    #             # 创建一个新的bucket
    #             self.buckets.append([node])
    #
    #     return None

    def validation_step(self, batched_data, batch_idx):
        if self.current_epoch < self.configs.skip_n_val_epoch:
            return
        subg_max_size = 20
        num_hop = 2
        node_list = batched_data['input']
        # 用第一个实体新建一个bucket
        if batch_idx == 0:
            self.buckets.append([node_list[0]])
            node_list.pop(0)

        batch_data, batch_index = prepare_locator_train_data(node_list, self.graph_data, max_size=subg_max_size,
                                                num_hop=num_hop)
        batch_data = batch_data.to(self.cuda)
        _, node_emb = self.gnn_encoder(batch_data.x, batch_data.edge_index, batch_data.batch, batch_index)

        for idx, node in enumerate(node_list):
            out = []
            parameters = torch.Tensor([])
            reason_status = []
            for bix, bucket in enumerate(self.buckets):
                # 提前判断桶中的信息
                reason_count = 0
                if len(bucket) > self.statis_value:
                    sampling_num = random.choice(list(range(1, self.statis_value)))
                    weights = np.ones(len(bucket)) / len(bucket)
                    bucket = np.random.choice(a=bucket, size=sampling_num, p=weights, replace=False).tolist()
                #     for btn in bucket:
                #         try:
                #             path_length = nx.shortest_path_length(self.nx_graph, source=node, target=btn)
                #         except Exception as e:
                #             path_length = 100
                #
                #         if path_length <= self.domain_hop:
                #             reason_count = reason_count + 1
                #         elif path_length == 100:
                #             reason_count = reason_count - 100
                #         else:
                #             reason_count = reason_count + 0
                #
                #     reason_radio = reason_count / len(bucket)
                #     if reason_radio <= 0.8:
                #         reason_status.append(0)
                #     else:
                #         reason_status.append(1)
                # else:
                #     for btn in bucket:
                #         try:
                #             path_length = nx.shortest_path_length(self.nx_graph, source=node, target=btn)
                #         except Exception as e:
                #             path_length = 100
                #
                #         if path_length <= self.domain_hop:
                #             reason_count = reason_count + 1
                #         elif path_length == 100:
                #             reason_count = reason_count - 100
                #         else:
                #             reason_count = reason_count + 0
                #
                #     reason_radio = reason_count / len(bucket)
                #     if reason_radio <= 0.8:
                #         reason_status.append(0)
                #     else:
                #         reason_status.append(1)
                bucket_data, bucket_index = prepare_locator_train_data(bucket, self.graph_data, max_size=subg_max_size,
                                                            num_hop=num_hop)
                bucket_data = bucket_data.to(self.cuda)
                _, bucket_emb = self.gnn_encoder(bucket_data.x, bucket_data.edge_index, bucket_data.batch, bucket_index)
                bucket_emb = torch.mean(bucket_emb, dim=0).unsqueeze(0)
                out = self.sample_construction(node, bucket, out, 'test')
                # TODO 构建实体属性
                if bix == 0:
                    parameters = bucket_emb
                else:
                    parameters = torch.cat([parameters, bucket_emb], dim=0)
                parameters = torch.cat([parameters, node_emb[idx].unsqueeze(0)], dim=0)

            agg_data = self.collate_fn(out, 'test')
            source_ids = agg_data['source_ids'].to(self.cuda)
            source_mask = agg_data['source_mask'].to(self.cuda)
            results = self.decode(source_ids, source_mask, parameters)
            if 'yes' in results:
                status = True
                for i, result in enumerate(results):
                    if 'yes' == result:
                        # 检验是否有边，没有边则说明不是一个社区
                        for btn in self.buckets[i]:
                            try:
                                path_length = nx.shortest_path_length(self.nx_graph, source=node, target=btn)
                            except Exception as e:
                                path_length = 100
                                continue
                        if path_length <= self.domain_hop:
                            self.buckets[i].append(node)
                            status = False
                if status:
                    self.buckets.append([node])
            else:
                # 创建一个新的bucket
                self.buckets.append([node])

        return None

    def decode(self, input_ids, input_mask, parameters):
        def _extract(generated_text):
            compiler = re.compile(r'<extra_id_0>(.*)<extra_id_1>')
            extracted_text = []
            for text in generated_text:
                match = compiler.search(text)
                if match is None:
                    # text = text.strip().lstrip('<pad> <extra_id_0>')
                    extracted_text.append(text.strip())
                else:
                    extracted_text.append(match.group(1).strip())
            return extracted_text

        inputs_emb = self.T5ForConditionalGeneration.encoder.embed_tokens(input_ids)
        batch_size, seq_len, model_dim = inputs_emb.shape
        inputs_emb = inputs_emb.view(-1, self.model_dim)  # (batch_size * seq_len, model_dim)
        input_ids = input_ids.view(-1)  # (batch_size * seq_len)
        input_ids = torch.where(input_ids == 32097)
        inputs_emb[input_ids] = parameters
        inputs_emb = inputs_emb.view(batch_size, -1, model_dim)
        input_mask = input_mask

        outputs = self.T5ForConditionalGeneration.generate(inputs_embeds=inputs_emb,
                                                           attention_mask=input_mask,
                                                           max_length=self.configs.eval_tgt_max_length,
                                                           return_dict_in_generate=True,
                                                           output_scores=True,)

        # outputs = self.T5ForConditionalGeneration(inputs_embeds=inputs_emb, attention_mask=input_mask)

        raw_generated_text = self.trainer.datamodule.tokenizer.batch_decode(outputs['sequences'])
        generated_text = _extract(raw_generated_text)
        # if self.configs.running_model == "test_model":
        #     raw_label = self.trainer.datamodule.tokenizer.batch_decode(target_ids)
        #     tgt_label = _extract(raw_label)
        # else:
        #     scores = outputs['scores'][1]  # [seq_len, batch_size, num_labels]
        #     tgt_label = []
        #     for score in scores:
        #         probs = F.softmax(score, dim=-1)
        #         yes_token_id = self.tokenizer.encode("yes")[0]
        #         yes_score = probs[yes_token_id]
        #         tgt_label.append(yes_score.item())

        return generated_text


    def validation_epoch_end(self, outs):
        if self.current_epoch < self.configs.skip_n_val_epoch:
            return

        f1, jaccard, onmi = eval_scores(self.buckets, self.test_comms, tmp_print=True)
        print("f1:{}".format(f1))
        print("jaccard:{}".format(jaccard))
        print("onmi:{}".format(onmi))
        # 保存结果
        results ={"predict_comms":self.buckets, "test_comms":self.test_comms}
        filename = self.configs.model_path + 'result.pkl'
        with open(filename, 'wb') as fo:
            pickle.dump(results, fo)
            fo.close()
        print("结果已经保存:{}".format(filename))

    def test_step(self, batched_data, batch_idx):

        return self.validation_step(batched_data, batch_idx)


    def test_epoch_end(self, outs):
        self.validation_epoch_end(outs)

    def configure_optimizers(self):
        if self.configs.optim == 'Adafactor':
            optim = Adafactor(self.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=self.configs.lr)
        else:
            optim = torch.optim.Adam(self.parameters(), lr=self.configs.lr)
        return optim
