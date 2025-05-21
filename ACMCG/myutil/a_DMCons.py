
def draw_graph(G):
    pos = graphviz_layout(G, prog="twopi")
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, alpha=0.5, node_color="blue",
            with_labels=True, font_size=20, node_size=30)
    plt.axis("equal")
    plt.show()


def nearest_neighbor_cal(feature_space, k):
    k = k if k < feature_space.shape[0] else feature_space.shape[0]
    neighbors = NearestNeighbors(n_neighbors=k).fit(feature_space)
    distances, nearest_neighbors = neighbors.kneighbors(
        feature_space, return_distance=True)

    edges = []
    for i in range(nearest_neighbors.shape[0]):
        for j in range(1, nearest_neighbors.shape[1]):
            edges.append(
                [nearest_neighbors[i, 0], nearest_neighbors[i, j], distances[i, j]])
    print(edges[:5])
    return edges


def data_preprocess(data):
    size = np.shape(data)
    random_matrix = np.random.rand(size[0], size[1]) * 0.000001
    data = data+random_matrix
    return data



def representative_cal(sub_S: nx.Graph):
    degree_dict = dict(sub_S.degree())
    max_degree = max(degree_dict.values())
    nodes_with_max_degree = [
        node for node, degree in degree_dict.items() if degree == max_degree]
    representative = random.choice(nodes_with_max_degree)
    return representative


def clustering_loop(feature_space, dict_mapping, skeleton: nx.Graph, k):
    Graph = nx.Graph()
    representatives = []
    edges = nearest_neighbor_cal(feature_space, k)
    Graph.add_weighted_edges_from(edges)
    S = [Graph.subgraph(c).copy() for c in nx.connected_components(Graph)]
    for sub_S in S:
        representative = representative_cal(sub_S)
        representatives.append(representative)
    for i in range(len(edges)):
        edges[i][0] = dict_mapping[edges[i][0]]
        edges[i][1] = dict_mapping[edges[i][1]]
    for i in range(len(representatives)):
        representatives[i] = dict_mapping[representatives[i]]
    skeleton.add_weighted_edges_from(edges)
    dict_mapping = {}
    for i in range(len(representatives)):
        dict_mapping[i] = representatives[i]
    return representatives, skeleton, dict_mapping


def graph_initialization(data, k):
    feature_space = copy.deepcopy(data)
    dict_mapping = {}
    for i in range(len(feature_space)):
        dict_mapping[i] = i
    skeleton = nx.DiGraph()
    while (True):
        representatives, skeleton, dict_mapping = clustering_loop(
            feature_space, dict_mapping, skeleton, k)
        feature_space = data[representatives]
        if len(representatives) == 1:
            break
    representative = representatives[0]
    return skeleton, representative


if __name__ == '__main__':

    data, labels = generate_iris_data(path="../dataset/small/iris/iris.data")
    data = data_preprocess(data)
    skeleton, representative = graph_initialization(data)
