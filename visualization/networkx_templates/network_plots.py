"""NetworkX 网络图模板 - 数学建模常用"""
import numpy as np
import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError:
    print("请安装 networkx: pip install networkx")
    raise

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def simple_graph(edges, node_labels=None, figsize=(10, 8), node_color='lightblue'):
    """简单无向图"""
    G = nx.Graph()
    G.add_edges_from(edges)
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_color,
            node_size=700, font_size=12, font_weight='bold',
            edge_color='gray', width=2)
    if node_labels:
        nx.draw_networkx_labels(G, pos, node_labels, font_size=10)
    ax.set_title('Network Graph', fontsize=14, fontweight='bold')
    return fig, ax, G


def directed_graph(edges, figsize=(10, 8), node_color='lightgreen'):
    """有向图"""
    G = nx.DiGraph()
    G.add_edges_from(edges)
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_color,
            node_size=700, font_size=12, font_weight='bold',
            edge_color='gray', width=2, arrows=True, arrowsize=20)
    ax.set_title('Directed Graph', fontsize=14, fontweight='bold')
    return fig, ax, G


def weighted_graph(edges_with_weights, figsize=(10, 8)):
    """带权重的图"""
    G = nx.Graph()
    for u, v, w in edges_with_weights:
        G.add_edge(u, v, weight=w)
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightyellow',
            node_size=700, font_size=12, font_weight='bold',
            edge_color='gray', width=[w/max(weights)*3 for w in weights])
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
    ax.set_title('Weighted Graph', fontsize=14, fontweight='bold')
    return fig, ax, G


def shortest_path_visualization(G, source, target, figsize=(10, 8)):
    """最短路径可视化"""
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    path = nx.shortest_path(G, source, target)
    path_edges = list(zip(path[:-1], path[1:]))
    
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightgray',
            node_size=700, font_size=12, edge_color='lightgray', width=2)
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='lightcoral',
                           node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red',
                           width=3, ax=ax)
    ax.set_title(f'Shortest Path: {source} -> {target}', fontsize=14, fontweight='bold')
    return fig, ax, path


def community_detection(G, figsize=(12, 10)):
    """社区检测可视化"""
    from networkx.algorithms import community
    communities = community.greedy_modularity_communities(G)
    
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    colors = plt.cm.Set3(np.linspace(0, 1, len(communities)))
    
    for i, comm in enumerate(communities):
        nx.draw_networkx_nodes(G, pos, nodelist=list(comm), node_color=[colors[i]],
                               node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    ax.set_title(f'Community Detection ({len(communities)} communities)', 
                 fontsize=14, fontweight='bold')
    return fig, ax, communities


def centrality_visualization(G, centrality_type='degree', figsize=(10, 8)):
    """中心性可视化"""
    if centrality_type == 'degree':
        centrality = nx.degree_centrality(G)
    elif centrality_type == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_type == 'closeness':
        centrality = nx.closeness_centrality(G)
    else:
        centrality = nx.eigenvector_centrality(G)
    
    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    node_sizes = [v * 3000 + 300 for v in centrality.values()]
    node_colors = list(centrality.values())
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                                   cmap=plt.cm.Reds, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    plt.colorbar(nodes, ax=ax, label=f'{centrality_type.title()} Centrality')
    ax.set_title(f'{centrality_type.title()} Centrality', fontsize=14, fontweight='bold')
    return fig, ax, centrality


if __name__ == '__main__':
    edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6), (3, 6)]
    fig, ax, G = simple_graph(edges)
    plt.savefig('networkx_demo.png', dpi=150)
    print("Demo saved to networkx_demo.png")
