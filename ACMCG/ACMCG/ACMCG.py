import networkx as nx
import time
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from networkx import shortest_path
from matplotlib import pyplot as plt
from .subtree_pe import WSLC_Metric
import copy
import numpy as np
import random
np.random.seed(0)

def compute_mean_std(data):
    n_samples = data.shape[0]
    max_pairs = n_samples * (n_samples - 1) // 2
    sample_pairs = min(10000, max_pairs)
    indices = []
    if sample_pairs > 0:
        if max_pairs <= sample_pairs:
            indices = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]
        else:
            k_values = random.sample(range(max_pairs), sample_pairs)
            indices = []
            for k in k_values:
                i = 0
                k_temp = k
                while True:
                    remaining = n_samples - i - 1
                    if k_temp < remaining:
                        j = i + 1 + k_temp
                        break
                    k_temp -= remaining
                    i += 1
                indices.append((i, j))

    mean_distance = 0.0
    m2 = 0.0
    count = 0

    for i, j in indices:
        dist = np.linalg.norm(data[i] - data[j])
        count += 1
        delta = dist - mean_distance
        mean_distance += delta / count
        m2 += delta * (dist - mean_distance)
    if count < 2:
        return mean_distance, 0.0
    variance = m2 / (count - 1)
    std_distance = np.sqrt(variance)
    return mean_distance, std_distance



def auto_judgement(pairwise, data, mean, std, constraint_graph):
    node1, node2 = pairwise
    dist = np.linalg.norm(data[node1] - data[node2])

    if dist < mean-4.753*std:
        result = "like"
        weight = 0
    else:
        result = "unknown"

    if result == "like":
        constraint_graph.add_edge(node1, node2, weight=weight)

    return result

# ---------------------------
def draw_contraint_graph(constraint_G):
    elarge = [(u, v) for (u, v, d) in constraint_G.edges(data=True) if d["weight"] == 1]
    esmall = [(u, v) for (u, v, d) in constraint_G.edges(data=True) if d["weight"] == 0]
    pos = nx.drawing.nx_agraph.graphviz_layout(constraint_G, prog='neato')
    nx.draw_networkx_nodes(constraint_G, pos, node_size=10)
    nx.draw_networkx_edges(constraint_G, pos, edgelist=esmall, width=1, edge_color="r", style="dashed")
    nx.draw_networkx_edges(constraint_G, pos, edgelist=elarge, width=1, edge_color="b", style="dashed")
    nx.draw_networkx_labels(constraint_G, pos, font_size=10, font_family="sans-serif")
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def anomaly_detection(Graph):
    max_value_node = None
    max_value = float('-inf')
    for node, data in Graph.nodes(data=True):
        if 'uncertainty' in data and data['uncertainty'] > max_value:
            max_value = data['uncertainty']
            max_value_node = node
    uncertain_node = max_value_node
    try:
        anomaly = list(Graph.out_edges(uncertain_node))[0]
    except Exception:
        anomaly = None
    suspend = False
    if Graph.nodes[uncertain_node]["uncertainty"] == 0:
        suspend = True
    return anomaly, suspend


def human_judgement(anomaly, real_labels, constraint_graph):
    node1 = int(anomaly[0])
    node2 = int(anomaly[1])
    if real_labels[node1] == real_labels[node2]:
        result = "like"
        weight = 0
    else:
        result = "dislike"
        weight = 1
    constraint_graph.add_edge(anomaly[0], anomaly[1], weight=weight)
    return constraint_graph, result


def distance_cal(path, G):
    sum = 0
    for i in range(len(path) - 1):
        sum = sum + G[path[i]][path[i + 1]]["weight"]
    return sum


def constraint_judgement(G, pairwise):
    source = int(pairwise[0])
    target = int(pairwise[1])
    if (source not in list(G.nodes)) or (target not in list(G.nodes)) or not nx.has_path(G, source, target):
        result = "unknown"
    else:
        path = shortest_path(G, source=source, target=target, weight='weight', method='dijkstra')
        sum = distance_cal(path, G)
        if sum == 0:
            result = "like"
        elif sum == 1:
            result = "dislike"
        else:
            result = "unknown"
    return result


def judgement(anomaly, constraint_graph, real_labels, data, mean, std):
    pairwise = [int(anomaly[0]), int(anomaly[1])]
    result = constraint_judgement(constraint_graph, pairwise)
    if result == "unknown":
        result = auto_judgement(pairwise, data, mean, std, constraint_graph)
        if result == "unknown":
            constraint_graph, result = human_judgement(pairwise, real_labels, constraint_graph)
            judgement_type = "human"
        else:
            judgement_type = "auto"
    else:
        judgement_type = "constraint"
    return constraint_graph, result, judgement_type


def skeleton_reconstruction_like(skeleton, anomaly):
    skeleton.nodes[anomaly[0]]["uncertainty"] = 0
    return skeleton


def connections_cal(edge, representatives, data):
    connections = []
    for representative in representatives:
        euc_distance = np.linalg.norm(data[edge[0]] - data[representative])
        connections.append([edge[0], representative, euc_distance])
    connections = np.array(connections)
    sorted_indices = np.argsort(connections[:, 2])
    connections = connections[sorted_indices]
    return connections


def skeleton_reconstruction_dislike(skeleton, anomaly, representatives, data, real_labels, constraint_graph,
                                    threshold, epsilon, count):
    skeleton.remove_edge(anomaly[0], anomaly[1])
    connections = connections_cal(anomaly, representatives, data)
    find = False
    for connection in connections:
        constraint_graph, result, judgement_type = judgement(connection, constraint_graph, real_labels, data,
                                                             threshold, epsilon)
        if result == "like":
            find = True
            node1 = int(connection[0])
            node2 = int(connection[1])
            in_degree_node1 = skeleton.in_degree(node1)
            in_degree_node2 = skeleton.in_degree(node2)
            skeleton.nodes[node1]["uncertainty"] = 0
            skeleton.nodes[node2]["uncertainty"] = 0
            if in_degree_node1 > in_degree_node2:
                representatives.remove(node2)
                representatives.append(node1)
                skeleton.add_edge(node2, node1)
            else:
                skeleton.add_edge(node1, node2)
            break
    if not find:
        representatives.append(anomaly[0])
        skeleton.nodes[anomaly[0]]["uncertainty"] = 0
    return skeleton, representatives, constraint_graph, count


def uncertainty_propagation_like(skeleton, anomaly, alpha):
    result = dict(enumerate(nx.bfs_layers(skeleton.reverse(), [anomaly[0]])))
    count = 1
    while count < len(result):
        amptitude = 1 - alpha ** count
        nodes_layer = result[count]
        for node in nodes_layer:
            skeleton.nodes[node]["uncertainty"] = skeleton.nodes[node]["uncertainty"] * amptitude
        count += 1
    return skeleton


def uncertainty_propagation_dislike(skeleton, anomaly, beta):
    result = dict(enumerate(nx.bfs_layers(skeleton.reverse(), [anomaly[0]])))
    count = 1
    while count < len(result):
        amptitude = 1 + beta ** count
        nodes_layer = result[count]
        for node in nodes_layer:
            skeleton.nodes[node]["uncertainty"] = skeleton.nodes[node]["uncertainty"] * amptitude
        count += 1
    return skeleton


def skeleton_reconstruction(skeleton, anomaly, representatives, data, real_labels, constraint_graph, threshold,
                            epsilon, count, result):
    if result == "like":
        skeleton = skeleton_reconstruction_like(skeleton, anomaly)
    elif result == "dislike":
        skeleton, representatives, constraint_graph, count = skeleton_reconstruction_dislike(
            skeleton, anomaly, representatives, data, real_labels, constraint_graph, threshold, epsilon, count)
    return skeleton, representatives, constraint_graph, count


def iteration_once(skeleton, representatives, data, real_labels, constraint_graph, mean, std):
    count = 0
    anomaly, suspend = anomaly_detection(skeleton)
    if anomaly is not None:
        constraint_graph, result, judgement_type = judgement(anomaly, constraint_graph, real_labels, data, mean, std)
        if judgement_type == "human":
            count += 1
        skeleton, representatives, constraint_graph, count = skeleton_reconstruction(
            skeleton, anomaly, representatives, data, real_labels, constraint_graph, mean, std, count, result)
    return skeleton, representatives, constraint_graph, count, suspend


def nearest_neighbor_cal(feature_space):
    neighbors = NearestNeighbors(n_neighbors=2).fit(feature_space)
    distance_vals, nearest_neighbors = neighbors.kneighbors(feature_space, return_distance=True)
    distance_vals = distance_vals[:, 1]
    nearest_neighbors = nearest_neighbors.tolist()
    for i in range(len(nearest_neighbors)):
        nearest_neighbors[i].append(distance_vals[i])
    return nearest_neighbors


def sub_nodes_cal(sub_S):
    points = None
    for edge in sub_S.edges:
        if sub_S.has_edge(edge[1], edge[0]):
            point1 = edge[0]
            point2 = edge[1]
            points = [point1, point2]
            break
    return points


def representative_find_sitation_2(points, skeleton):
    sum1 = 0
    in_edges = list(skeleton.in_edges(points[0]))
    for edge in in_edges:
        sum1 += skeleton.nodes[edge[0]]["uncertainty"]
    sum2 = 0
    in_edges = list(skeleton.in_edges(points[1]))
    for edge in in_edges:
        sum2 += skeleton.nodes[edge[0]]["uncertainty"]
    index = np.argmax([sum1, sum2])
    representative = points[index]
    return index, representative


def clustering_loop(feature_space, dict_mapping, skeleton):
    representatives = []
    edges = nearest_neighbor_cal(feature_space)

    if not hasattr(skeleton, 'wslc_cache'):
        skeleton.wslc_cache = {}

    affected_nodes = set()
    for i in range(len(edges)):
        edges[i][0] = dict_mapping[edges[i][0]]
        edges[i][1] = dict_mapping[edges[i][1]]
        uncertainty = edges[i][2]
        skeleton.add_edge(edges[i][0], edges[i][1])
        skeleton.nodes[edges[i][0]]['uncertainty'] = uncertainty
        affected_nodes.add(edges[i][0])
        affected_nodes.add(edges[i][1])

    S = [skeleton.subgraph(c).copy() for c in nx.weakly_connected_components(skeleton)]
    for sub_S in S:
        points = sub_nodes_cal(sub_S)

        if points is None:
            continue

        need_compute = False
        for p in points:
            if p in affected_nodes or p not in skeleton.wslc_cache:
                need_compute = True
                break

        if need_compute:
            local_nodes = sorted(list(sub_S.nodes()))
            node_idx = {n: i for i, n in enumerate(local_nodes)}

            local_graph = nx.DiGraph()
            local_graph.add_nodes_from(range(len(local_nodes)))

            for u, v in sub_S.edges():
                local_graph.add_edge(node_idx[u], node_idx[v])

            N = local_graph.number_of_nodes()
            L = 1

            wslc_scores = WSLC_Metric(local_graph, N, L)

            for idx, node in enumerate(local_nodes):
                skeleton.wslc_cache[node] = wslc_scores[idx]

        score1 = skeleton.wslc_cache.get(points[0], 0)
        score2 = skeleton.wslc_cache.get(points[1], 0)

        index = np.argmax([score1, score2])
        representative = points[index]

        representatives.append(representative)
        if skeleton.has_edge(points[index], points[1 - index]):
            skeleton.remove_edge(points[index], points[1 - index])
            affected_nodes.add(points[index])
            affected_nodes.add(points[1 - index])

    dict_mapping = {}
    for i in range(len(representatives)):
        dict_mapping[i] = representatives[i]
    return representatives, skeleton, dict_mapping


def clustering(data):
    feature_space = copy.deepcopy(data)
    dict_mapping = {i: i for i in range(len(feature_space))}
    skeleton = nx.DiGraph()
    while True:
        representatives, skeleton, dict_mapping = clustering_loop(feature_space, dict_mapping, skeleton)
        feature_space = data[representatives]
        if len(representatives) == 1:
            break
    skeleton.nodes[representatives[0]]['uncertainty'] = 0
    return skeleton, representatives


def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.000001
    data = data + random_matrix
    return data


def clusters_to_predict_vec(clusters):
    tranversal_dict = {}
    predict_vec = []
    for i in range(len(clusters)):
        for j in clusters[i]:
            tranversal_dict[j] = i
    for i in range(len(tranversal_dict)):
        predict_vec.append(tranversal_dict[i])
    return predict_vec


def skeleton_process(Graph):
    clusters = []
    S = [Graph.subgraph(c) for c in nx.weakly_connected_components(Graph)]
    for s in S:
        clusters.append(list(s.nodes))
    predict_labels = clusters_to_predict_vec(clusters)
    return predict_labels


def get_predict_labels(Graph: nx.Graph):
    S = [Graph.subgraph(c) for c in nx.weakly_connected_components(Graph)]
    predict_labels = np.zeros(len(Graph.nodes), dtype=int)
    for i, s in enumerate(S[1:], 1):
        predict_labels[list(s.nodes)] = i
    return predict_labels


def analyze_relationships(G):
    from itertools import combinations
    true_count = 0
    false_count = 0
    true_relationships = []
    false_relationships = []
    relationships = {}
    for u, v in combinations(G.nodes, 2):
        try:
            path = nx.shortest_path(G, source=u, target=v)
            path_weight_sum = sum(G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
            if path_weight_sum == 0:
                relationships[(u, v)] = True
                true_count += 1
                true_relationships.append((u, v))
            elif path_weight_sum == 1:
                relationships[(u, v)] = False
                false_count += 1
                false_relationships.append((u, v))
            else:
                relationships[(u, v)] = None
        except nx.NetworkXNoPath:
            relationships[(u, v)] = None
    return true_count, false_count


def get_edges_conunt(G: nx.Graph):
    return len(G.edges)



def ACMCG(data, real_labels, title, q=1000, k=1):
    df = {"iter": [], "interaction": [], "ari": [], "nmi": [], "time": []}
    start_time = time.time()
    mean, std = compute_mean_std(data)
    skeleton, representatives = clustering(data)
    loop = len(data) + 10
    constraint_graph = nx.Graph()
    interaction = 0
    predict_labels = get_predict_labels(skeleton)
    ARI = adjusted_rand_score(real_labels, predict_labels)
    NMI = normalized_mutual_info_score(real_labels, predict_labels)
    df["iter"].append(0)
    df["interaction"].append(interaction)
    df["ari"].append(ARI)
    df["nmi"].append(NMI)
    df["time"].append(0)
    for i in range(loop):
        skeleton, representatives, constraint_graph, count, suspend = iteration_once(
            skeleton, representatives, data, real_labels, constraint_graph, mean, std)
        interaction += count
        if suspend:
            print("The algorithm is down")
            break
        predict_labels = get_predict_labels(skeleton)
        ARI = adjusted_rand_score(real_labels, predict_labels)
        NMI = normalized_mutual_info_score(real_labels, predict_labels)
        duration = time.time() - start_time
        df["iter"].append(i + 1)
        df["interaction"].append(interaction)
        df["ari"].append(ARI)
        df["nmi"].append(NMI)
        df["time"].append(duration)

    print(f'ARI:{df["ari"][-1]}')
    print(f'NMI:{df["nmi"][-1]}')
    return df