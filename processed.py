import pickle
import numpy as np
from script.load_dataset import prepare_data
import networkx as nx
from script.helper_funcs import split_communities
import statistics
class Community_Discovery():
    def __init__(self):
        self.dataset = "lj"
        self.train_ratio = 0.1
    def construct_datasets(self):
        num_node, num_edge, num_community, graph_data, nx_graph, communities = prepare_data(self.dataset)
        train_comms, val_comms, test_comms = split_communities(communities, self.train_ratio)
        with open('./data/{}/trainsets.pkl'.format(self.dataset), 'wb') as fo:
            pickle.dump(train_comms, fo)
            fo.close()
        with open('./data/{}/validsets.pkl'.format(self.dataset), 'wb') as fo:
            pickle.dump(val_comms, fo)
            fo.close()
        with open('./data/{}/testsets.pkl'.format(self.dataset), 'wb') as fo:
            pickle.dump(test_comms, fo)
            fo.close()

    def statistics_features(self):
        self.dataset = 'dblp'
        num_node, num_edge, num_community, graph_data, nx_graph, communities = prepare_data(self.dataset)
        with open('//result.pkl', 'rb') as f:
            data = pickle.load(f)

        predict_comms = data['predict_comms']
        test_comms = data['test_comms']
        reports = []
        for pcm in predict_comms:
            counter = []
            for tcm in test_comms:
                intersection_list = list(set(pcm) & set(tcm))
                counter.append(len(intersection_list))
            max_index = np.argmax(counter)

            comms = test_comms[max_index]
            rate = counter[max_index] / len(comms)
            reports.append({"predict_comms": pcm, "test_comms": comms, "counter": counter[max_index], "rate": rate})
        sorted_a_desc = sorted(reports, key=lambda x: x["rate"], reverse=True)
        for sad in sorted_a_desc:
            if sad['rate'] <= 0.5:
                print(sad)
                hops = []
                for i in range(len(sad['test_comms']) - 1):
                    for j in range(1, len(sad['test_comms'])):
                        path_length = nx.shortest_path_length(nx_graph, source=sad['test_comms'][i],
                                                              target=sad['test_comms'][j])
                        hops.append(path_length)
                print("跳数：{}".format(hops))
                statis_value = int(statistics.median(hops))
                print("Median:", statis_value)
                # 计算众数
                statis_value = int(statistics.mode(hops))
                print("Mode:", statis_value)






if __name__ == '__main__':
    community_discovery = Community_Discovery()
    # community_discovery.construct_datasets()
    community_discovery.statistics_features()









