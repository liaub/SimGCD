import numpy as np
from torch_geometric.data import Data, Batch
import torch
from torch_geometric.utils import k_hop_subgraph, subgraph
import networkx as nx
import statistics
from collections import Counter

def split_communities(communities, train_radio):
    r"""Randomly split all communities into train set, validation set, and test set"""
    idxes = list(range(len(communities)))
    np.random.shuffle(idxes)
    n_train = int((max(idxes) + 1) * train_radio)

    train_comms = [communities[idx] for idx in idxes[:n_train]]
    test_comms = [communities[idx] for idx in idxes[n_train:]]

    return train_comms, test_comms,test_comms


def drop_nodes(graph_data, aug_ratio=0.15):
    r"""Contrastive model corruption: dropping nodes"""
    node_num, edge_num = graph_data.x.size(0), graph_data.edge_index.size(1)

    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)
    idx_drop = idx_perm[:drop_num]
    idx_non_drop = idx_perm[drop_num:]
    idx_non_drop.sort()

    idx_dict = {idx_non_drop[n]: n for n in list(range(idx_non_drop.shape[0]))}

    device = graph_data.edge_index.device

    edge_index = graph_data.edge_index.detach().cpu().numpy()

    # re-label edges
    edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if
                  (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]

    try:
        # create a corrupted subgraph
        new_edge_index = torch.tensor(edge_index).transpose_(0, 1).to(device)
        new_x = graph_data.x[idx_non_drop]
        new_graph_data = Data(x=new_x, edge_index=new_edge_index)
    except:
        new_graph_data = graph_data
    return new_graph_data


def prepare_locator_train_data(node_list, data: Data, max_size=25, num_hop=2):
    r"""Generate batch data for Community Locator training. For each node,
    extract its ego-net and generate a sub-ego-net"""
    # batch, subg_batch = [], []
    batch = []
    num_nodes = data.x.size(0)
    batch_num = []
    for node in node_list:
        node_set, _, _, _ = k_hop_subgraph(node_idx=node, num_hops=num_hop, edge_index=data.edge_index,
                                           num_nodes=num_nodes)

        if len(node_set) > max_size:
            node_set = node_set[torch.randperm(node_set.shape[0])][:max_size]
            node_set = torch.unique(torch.cat([torch.LongTensor([node]), torch.flatten(node_set)]))

        node_list = node_set.detach().cpu().numpy().tolist()
        seed_idx = node_list.index(node)

        if seed_idx != 0:
            node_list[seed_idx], node_list[0] = node_list[0], node_list[seed_idx]

        # Hint: important!!!
        #  We must ensure all the first node is the centric node
        assert node_list[0] == node
        batch_num.append(len(node_list))
        edge_index, _ = subgraph(node_list, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
        node_x = data.x[node_list]  # node features
        g_data = Data(x=node_x, edge_index=edge_index)
        batch.append(g_data)

        # # apply **Drop-Node** to obtain its subgraph
        # corrupt_data = drop_nodes(g_data)
        # subg_batch.append(corrupt_data)
        # print(g_data, corrupt_data)

    batch_index = [0]
    record_id = 0
    for idx, num in enumerate(batch_num):
        if idx + 1 < len(batch_num):
            batch_index.append(batch_index[-1]+num)
    batch = Batch().from_data_list(batch)
    # subg_batch = Batch().from_data_list(subg_batch)
    # assert batch.x.shape[0] == num
    # return batch, subg_batch
    # data_list = batch.to_data_list()
    # print(data_list[0].x)  # 输出 tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    # print(data_list[1].x)
    return batch, batch_index

def statistical_domain_hop(domain_style, nodes, comms, nx_graph):
    # 统计域的范围
    hops = []
    for node in nodes:
        pos_comms = []
        for comm in comms:
            if node in comm:
                pos_comms = pos_comms + comm
        pos_comms = list(set(pos_comms))
        pos_comms.remove(node)
        for target in pos_comms:
            path_length = nx.shortest_path_length(nx_graph, source=node, target=target)
            hops.append(path_length)

    element_counts = Counter(hops)

    # 输出每个元素出现的个数
    for element, count in element_counts.items():
        print(f"{element}: {count} 次")

    if domain_style == 1:
        # 计算中位数
        domain_value = int(statistics.median(hops))
        print("Median:", domain_value)
    elif domain_style == 2:
        # 计算众数
        domain_value = int(statistics.mode(hops))
        print("Mode:", domain_value)
    elif domain_style == 3:
        # 计算平均数
        domain_value = int(statistics.mean(hops))
        print("mean:", domain_value)
    else:
        # 计算最大数
        domain_value = max(hops)
        print("max:", domain_value)
    return domain_value

def generate_ego_net(graph, start_node, k=1, max_size=15, choice="subgraph"):
    """Generate **k** ego-net"""
    q = [start_node]
    visited = [start_node]

    iteration = 0
    while True:
        if iteration >= k:
            break
        length = len(q)
        if length == 0 or len(visited) >= max_size:
            break

        for i in range(length):
            # Queue pop
            u = q[0]
            q = q[1:]

            for v in list(graph.neighbors(u)):
                if v not in visited:
                    q.append(v)
                    visited.append(v)
                if len(visited) >= max_size:
                    break
            if len(visited) >= max_size:
                break
        iteration += 1
    visited = sorted(visited)
    # print(visited)
    return visited if choice == "neighbors" else graph.subgraph(visited)


def generate_outer_boundary(graph, com_nodes, max_size=20):
    """For a given graph and a community `com_nodes`, generate its outer-boundary"""
    outer_nodes = []
    for node in com_nodes:
        outer_nodes += list(graph.neighbors(node))
    outer_nodes = list(set(outer_nodes) - set(com_nodes))
    outer_nodes = sorted(outer_nodes)
    return outer_nodes if len(outer_nodes) <= max_size else outer_nodes[:max_size]
