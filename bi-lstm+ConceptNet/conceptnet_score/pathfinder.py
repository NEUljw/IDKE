import networkx as nx

cpnet = None            # 属性有rel
cpnet_simple = None     # 属性有weight


def load_cpnet():
    global cpnet, cpnet_simple
    print("loading cpnet....")
    cpnet = nx.read_gpickle('cpnet.graph')
    print("Done.")

    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)
    # print(cpnet_simple.edges(data=True))


def get_edge(src_concept, tgt_concept):
    global cpnet
    rel_list = cpnet[src_concept][tgt_concept]
    return list(set([rel_list[item]["rel"] for item in rel_list]))


def get_edge_weight(src, tgt):
    global cpnet_simple
    edge = cpnet_simple[src][tgt]
    return edge['weight']


# source and target is text, 返回的是两个node之间所有路径得分的平均值
def find_paths(source, target):
    global cpnet, cpnet_simple
    s = source
    t = target

    # concept不在图中
    if s not in cpnet_simple.nodes() or t not in cpnet_simple.nodes():
        return 6

    all_path = []
    all_path_set = set()

    for max_len in range(1, 5):
        for p in nx.all_simple_paths(cpnet_simple, source=s, target=t, cutoff=max_len):
            path_str = "-".join([str(c) for c in p])
            if path_str not in all_path_set:
                all_path_set.add(path_str)
                all_path.append(p)
            if len(all_path) >= 20:  # top shortest 20 paths
                break
        if len(all_path) >= 20:  # top shortest 20 paths
            break

    # print(all_path)
    # print(all_path_set)

    if len(all_path) == 0:
        return 0

    all_path.sort(key=len, reverse=False)

    all_path_weight = 0
    for one_path in all_path:
        one_path_weight = 0
        for i in range(len(one_path)-1):
            one_edge_weight = get_edge_weight(one_path[i], one_path[i+1])
            one_path_weight += one_edge_weight
        one_path_weight_ave = one_path_weight/(len(one_path)-1)
        all_path_weight += one_path_weight_ave
    all_path_ave = all_path_weight/len(all_path)
    return all_path_ave
    # pf_res = []
    # for p in all_path:
    #     rl = []
    #     for src in range(len(p) - 1):
    #         src_concept = p[src]
    #         tgt_concept = p[src + 1]
    #         rel_list = get_edge(src_concept, tgt_concept)
    #         rl.append(rel_list)
    #     pf_res.append({"path": p, "rel": rl})
    # return pf_res


def cal_qa_score(q_concept, a_concept):
    # load_cpnet()
    score_sum = 0
    for i in q_concept:
        for j in a_concept:
            one_pair_score = find_paths(i, j)
            score_sum += one_pair_score
            # print(one_pair_score)
    score_ave = score_sum/(len(q_concept)*len(a_concept))
    score_ave = round(score_ave, 4)
    return score_ave


'''if __name__ == "__main__":
    load_cpnet()
    q_concept, a_concept = [], []
    score_sum = 0
    for i in q_concept:
        for j in a_concept:
            one_pair_score = find_paths(i, j)
            score_sum += one_pair_score
    score_ave = score_sum/(len(q_concept)*len(a_concept))
    print('score ave:', score_ave)
    # a = find_paths("family name", "name")'''
