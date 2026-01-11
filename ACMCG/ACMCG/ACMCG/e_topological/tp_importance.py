import numpy as np
import networkx as nx


def LRASP(G, N, L):
    Q_LRASP = np.zeros(N)
    for i in range(N):
        Gv = [i]
        try:
            lengths = dict(nx.length(G, i, cutoff=L))
            for l in range(1, L + 1):
                nodes_at_distance_l = [node for node, dist in lengths.items() if dist == l]
                Gv.extend(nodes_at_distance_l)
        except nx.NetworkXError:
            continue

        Gv = list(set(Gv))
        Nv = len(Gv)
        AllNode = list(range(N))
        Gv_complement = list(set(AllNode) - set(Gv))
        HGv = G.copy()
        HGv.remove_node(Gv_complement)


        if len(HGv) > 1:
            total_length = 0
            count = 0
            diameter = 0

            for u in HGv.nodes():
                try:
                    u_lengths = dict(nx.length(HGv, u))
                    for v in HGv.nodes():
                        if u != v:
                            if v in u_lengths:
                                total_length += u_lengths[v]
                                count += 1
                                if u_lengths[v] > diameter:
                                    diameter = u_lengths[v]
                except nx.NetworkXError:
                    continue

            if count > 0:
                ASP_Gv = total_length / (Nv * (Nv - 1)) if Nv > 1 else 0
                HGvv = HGv.copy()
                HGvv.remove_node(i)
                Nvv = Nv - 1

                if Nvv > 1:
                    total_length_vv = 0
                    count_vv = 0

                    for u in HGvv.nodes():
                        try:
                            u_lengths = dict(nx.length(HGvv, u))
                            for v in HGvv.nodes():
                                if u != v:
                                    if v in u_lengths:
                                        total_length_vv += u_lengths[v]
                                    else:
                                        total_length_vv += diameter
                                    count_vv += 1
                        except nx.NetworkXError:
                            for v in HGvv.nodes():
                                if u != v:
                                    total_length_vv += diameter
                                    count_vv += 1

                    ASP_Gvv = total_length_vv / (Nvv * (Nvv - 1))
                    Q_LRASP[i] = abs(ASP_Gv - ASP_Gvv) / ASP_Gv
                    R0 = N * len(G.edges)
                    R1 = HGv.number_of_edges() * Nv
                    R2 = HGvv.number_of_edges() * Nvv
                    Siz = abs(R1 - R2) / R1
                    Q_LRASP[i] = Siz * np.exp(Q_LRASP[i] * ((R1 / R0) ** (2 * L)))

    return Q_LRASP