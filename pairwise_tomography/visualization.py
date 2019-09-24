""" 
Visualization of pairwise relations
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def make_graph(ent_dict, qubit_list):
    G = nx.Graph()
    G.add_nodes_from(qubit_list)
    for i in range(len(qubit_list)-1):
        qi = qubit_list[i]
        for j in range(i+1,len(qubit_list)):
            qj = qubit_list[j]
            if ent_dict[(qi,qj)] != 0.0:
                G.add_edge(qi,qj,weight=ent_dict[(i,j)])
    return G

def draw_entanglement_graph(ent_dict, qubit_list, layout="circular", scale_factor = 1., labels = {}, node_color = "#0A7290", **kwargs):
    G = make_graph(ent_dict,qubit_list)
    valid_layout = {"circular","spring"}
    if layout not in valid_layout:
            raise ValueError("Not a valid layout name (circular,spring).")
    
    if layout == "circular":
        pos = nx.circular_layout(G, dim=2, scale=1, center=None)
        print(pos)
        label_pos = [pos[i]*1.3 for i in pos]
    if layout == "spring":
        pos = nx.spring_layout(G)
    
    
    edgewidth = [d['weight']*10*scale_factor for (u,v,d) in G.edges(data=True)]
    nodesize = [(e[1]*200+10)*scale_factor for e in G.degree(weight='weight')]
    
    nx.draw_networkx_nodes(G, pos, node_size=nodesize, node_color=node_color, edgecolors="k", linewidths = 0.3, **kwargs)
    nx.draw_networkx_edges(G, pos, width=edgewidth, **kwargs)
    nx.draw_networkx_labels(G, label_pos, labels = labels, font_size = 10, **kwargs)
    
