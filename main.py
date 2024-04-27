import os
import datetime
import argparse
from datetime import datetime
import pickle
import statistics
import numpy as np
import os
import warnings
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import T5Tokenizer
from transformers import T5Config
from models.evaluation import EvalutionFinetuner
from data import EvaluationDataModule
from callbacks import PrintingCallback
from script.helper_funcs import split_communities
from script.load_dataset import prepare_data
from script.helper_funcs import statistical_domain_hop


def main():
    ## read triples
    num_node, num_edge, num_community, graph_data, nx_graph, communities = prepare_data(configs.dataset)
    # train_comms, val_comms, test_comms = split_communities(communities, configs.train_radio)
    with open('./data/{}/trainsets.pkl'.format(configs.dataset), 'rb') as f:
        train_comms = pickle.load(f)
    with open('./data/{}/validsets.pkl'.format(configs.dataset), 'rb') as f:
        val_comms = pickle.load(f)
    with open('./data/{}/testsets.pkl'.format(configs.dataset), 'rb') as f:
        test_comms = pickle.load(f)[:100]
    trainsets = list({node for com in train_comms for node in com})
    testsets = list({node for com in test_comms for node in com})
    validsets = list({node for com in val_comms for node in com})
    comms_count = [len(comms) for comms in train_comms]

    # 统计域的范围值
    domain_hop = statistical_domain_hop(configs.domain_style, trainsets, train_comms, nx_graph)

    # 计算中位数
    statis_value = int(statistics.median(comms_count))
    print("Median:", statis_value)
    # 计算众数
    # statis_value = int(statistics.mode(comms_count))
    # print("Mode:", statis_value)

    ## construct name list
    original_ent_name_list = list({str(node) for com in train_comms + test_comms for node in com})
    tokenizer = T5Tokenizer.from_pretrained(configs.pretrained_model)
    ent_token_ids_in_trie = tokenizer(['<extra_id_0>' + ent_name + '<extra_id_1>' for ent_name in original_ent_name_list], max_length=configs.train_tgt_max_length, truncation=True).input_ids

    ent_name_list = tokenizer.batch_decode([tokens[1:-2] for tokens in ent_token_ids_in_trie])
    random_seeds = np.ones(max([int(ent)+1 for idx, ent in enumerate(ent_name_list)]),) * 1

    filename = 'lm-evaluate-{epoch:02d}-{val_loss:.4f}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=configs.save_dir,
        filename=filename,
        mode='min'
    )

    printing_callback = PrintingCallback()

    gpu = [int(configs.gpu)] if torch.cuda.is_available() else 0
    trainer_params = {
        'gpus': gpu,
        # 'limit_train_batches': 0.1,  # 限制训练模型的batch数
        'max_epochs': configs.epochs,  # 1000
        'checkpoint_callback': True,  # True
        'logger': False,  # TensorBoardLogger
        'num_sanity_val_steps': 0,  # 模型训练开始前，提前验证模型是否能跑起来
        'check_val_every_n_epoch': 3, # 每n个epoch验证模型
        'enable_progress_bar': True, # 使用进度条
        'callbacks': [
            printing_callback,
            checkpoint_callback
        ],
    }
    trainer = pl.Trainer(**trainer_params)

    kw_args = {
        'nx_graph': nx_graph,
        'graph_data': graph_data,
        'statis_value': statis_value,
        'random_seeds': random_seeds,
        'domain_hop': domain_hop,
        'cuda': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }


    if configs.model_path == '' and configs.running_model == 'train_model':

        datamodule = EvaluationDataModule(configs, trainsets, validsets, testsets, train_comms, test_comms, nx_graph, domain_hop, running_model='train_model')
        print('train_model datamodule construction done.', flush=True)
        kw_args['test_comms'] = None
        model = EvalutionFinetuner(configs, tokenizer, **kw_args)
        trainer.fit(model, datamodule)
        model_path = checkpoint_callback.best_model_path
        print('training best model path:', model_path, flush=True)

    else:
        model_path = configs.model_path
        model_name = configs.model_name
        datamodule = EvaluationDataModule(configs, trainsets, validsets, testsets, train_comms, test_comms, nx_graph, domain_hop, running_model='test_model')
        kw_args['test_comms'] = test_comms
        model = EvalutionFinetuner.load_from_checkpoint(model_path + model_name, strict=False, configs=configs, **kw_args)
        trainer.test(model, dataloaders=datamodule)


if __name__ == '__main__':

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset_path', type=str, default='./dat:q:a')
    parser.add_argument('-dataset', dest='dataset', default='lj', help='Dataset to use, amazon, dblp, lj')
    parser.add_argument('-model', default='T5Finetuner', help='Model Name')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('-seed', dest='seed', default=41504, type=int, help='Seed for randomization')
    parser.add_argument('-num_workers', type=int, default=4, help='Number of processes to construct batches')
    parser.add_argument('-save_dir', type=str, default='', help='')

    parser.add_argument('-pretrained_model', type=str, default='./models/t5-base', help='')
    parser.add_argument('-train_style', default=2, type=int, help='Training style->1:Use only generated loss; 2:Logarithmic contrast loss, 3:Max contrast loss')
    parser.add_argument('-batch_size', default=60, type=int, help='Batch size')
    parser.add_argument('-val_batch_size', default=2, type=int, help='Batch size')
    parser.add_argument('-src_max_length', default=512, type=int, help='')
    parser.add_argument('-train_tgt_max_length', default=512, type=int, help='')
    parser.add_argument('-eval_tgt_max_length', default=512, type=int, help='')
    parser.add_argument('-epoch', dest='epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('-lr', type=float, default=0.0005, help='Starting Learning Rate')
    parser.add_argument('-lamda', default=20, type=int, help='Boundary interval constant of positive and negative samples')
    parser.add_argument("--domain_style", type=int, help="Constraint pattern:1.Median;2.mode;3.mean;4.max", default=2)
    parser.add_argument("--graph_embedding", type=bool, help="if using graph embedding", default=True)
    parser.add_argument("--n_layers", type=int, help="number of gnn layers", default=2)
    parser.add_argument("--gnn_type", type=str, help="type of convolution", default="GCN")


    parser.add_argument('-model_path', dest='model_path', default='', help='The path for reloading models')
    # parser.add_argument('-model_path', dest='model_path', default='D:/学术研究/GCD/GCD_v1/checkpoint/amazon-train_model/', help='The path for reloading models')
    # parser.add_argument('-model_path', dest='model_path', default='/home/shu/products/liaub/GCD_v1/checkpoint/dblp_lj-train_model/', help='The path for reloading models')
    # parser.add_argument('-model_name', dest='model_name', default='lm-evaluate-epoch=251-val_loss=0.0000.ckpt', help='The path for reloading models')
    parser.add_argument('-optim', default='Adam', type=str, help='')
    parser.add_argument('-skip_n_val_epoch', default=1000, type=int, help='Using train process')
    # parser.add_argument('-skip_n_val_epoch', default=0, type=int, help='Using test process')
    parser.add_argument('-running_model', type=str, default='train_model', help='[train_model, test_model]')
    configs = parser.parse_args()
    configs.vocab_size = T5Config.from_pretrained(configs.pretrained_model).vocab_size
    configs.model_dim = T5Config.from_pretrained(configs.pretrained_model).d_model
    if configs.save_dir == '' and configs.running_model == 'train_model': # if train and valid, makedires else not makedires
        # configs.save_dir = os.path.join('./checkpoint', configs.dataset + '-train_model-' + str(datetime.now())) # use liunx
        configs.save_dir = os.path.join('./checkpoint', configs.dataset + '-train_model')  # use windows
        os.makedirs(configs.save_dir, exist_ok=True)

    # if configs.error_store == '' and configs.running_model == 'test_model': # if train and valid, makedires else not makedires
    #     # configs.save_dir = os.path.join('./checkpoint', configs.dataset + '-train_model-' + str(datetime.now())) # use liunx
    #     configs.error_store = os.path.join('./checkpoint', configs.dataset + '-error_store')  # use windows
    #     os.makedirs(configs.error_store, exist_ok=True)

    pl.seed_everything(configs.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_printoptions(profile='full')
    main()
