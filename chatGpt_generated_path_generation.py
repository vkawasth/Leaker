import math, heapq
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# --- Input seed table (flattened row-major)
seed_flat = (1, 684, 5, 1, 20, 27)
seed = ((seed_flat[0],seed_flat[1],seed_flat[2]), (seed_flat[3],seed_flat[4],seed_flat[5]))

# --- Derive margins
row_sums = (seed[0][0]+seed[0][1]+seed[0][2], seed[1][0]+seed[1][1]+seed[1][2])
col_sums = (seed[0][0]+seed[1][0], seed[0][1]+seed[1][1], seed[0][2]+seed[1][2])

# --- Enumerate strictly-positive tables (entries >= 1)
tables = []
for a in range(row_sums[0]+1):
    for b in range(row_sums[0]-a+1):
        c = row_sums[0] - a - b
        d = col_sums[0] - a
        e = col_sums[1] - b
        f = col_sums[2] - c
        if d >= 1 and e >= 1 and f >= 1 and d + e + f == row_sums[1]:
            T = ((a,b,c),(d,e,f))
            tables.append(T)
if seed not in tables and all(x>=1 for x in seed_flat):
    tables.append(seed)
tables = sorted(tables, key=lambda T: (T[0][0],T[0][1],T[0][2],T[1][0],T[1][1],T[1][2]))

# --- Strict-positive adjacency (2x2 moves that keep all entries >= 1)
def neighbors_pos(T):
    neigh = []
    for j in range(3):
        for k in range(j+1,3):
            new = [[T[0][0],T[0][1],T[0][2]],[T[1][0],T[1][1],T[1][2]]]
            new[0][j] += 1
            new[0][k] -= 1
            new[1][j] -= 1
            new[1][k] += 1
            if all(x>=1 for row in new for x in row):
                neigh.append(((new[0][0],new[0][1],new[0][2]),(new[1][0],new[1][1],new[1][2])))
    uniq = []
    for u in neigh:
        if u not in uniq:
            uniq.append(u)
    return uniq

# --- Probability model (log-space) that biases toward the seed counts:
#    log pi(T) ∝ sum_{i,j} t_ij * log(seed_count_ij + eps)
eps = 1e-12
seed_weights = [[seed[0][0]+eps, seed[0][1]+eps, seed[0][2]+eps],
                [seed[1][0]+eps, seed[1][1]+eps, seed[1][2]+eps]]
log_w = [[math.log(seed_weights[i][j]) for j in range(3)] for i in range(2)]
def log_table_weight(T):
    s = 0.0
    for i in range(2):
        for j in range(3):
            s += T[i][j] * log_w[i][j]
    return s
log_weights = {T: log_table_weight(T) for T in tables}
max_log = max(log_weights.values())
Z = sum(math.exp(log_weights[T]-max_log) for T in tables)
logZ = max_log + math.log(Z)
pi = {T: math.exp(log_weights[T]-logZ) for T in tables}

# --- Build graph
G = nx.Graph()
for T in tables:
    G.add_node(T, pi=pi[T])
for T in tables:
    for U in neighbors_pos(T):
        if U in G and not G.has_edge(T,U):
            G.add_edge(T,U, capacity=min(pi[T], pi[U]))

# --- Shortest path that maximizes product of min-edge capacity along path
def best_path_maxminprob(G, pi, source, target):
    dist = {node: float('inf') for node in G.nodes}
    prev = {}
    dist[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        d,u = heapq.heappop(pq)
        if d>dist[u]: continue
        if u==target: break
        for v in G.neighbors(u):
            cap = min(pi[u], pi[v])
            if cap<=0: continue
            cost = -math.log(cap)
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd,v))
    if dist[target] == float('inf'):
        return None, float('inf')
    path = []
    cur = target
    while cur!=source:
        path.append(cur)
        cur = prev[cur]
    path.append(source)
    path.reverse()
    return path, dist[target]

reachable = set(nx.node_connected_component(G, seed))
paths = {}
for T in reachable:
    if T == seed: continue
    path, cost = best_path_maxminprob(G, pi, seed, T)
    if path:
        min_edge_cap = math.exp(-cost/(len(path)-1)) if len(path)>1 else pi[seed]
        paths[T] = {"path": path, "cost": cost, "min_edge_cap": min_edge_cap}

# --- top corridors
top_corridors = sorted(paths.items(), key=lambda kv: kv[1]['min_edge_cap'], reverse=True)[:3]

# --- Plotting
pos = nx.spring_layout(G, seed=42, k=0.7)
node_sizes = [max(30, 3000 * G.nodes[n]['pi']) for n in G.nodes()]
labels = {n: str(i) for i,n in enumerate(G.nodes())}

plt.figure(figsize=(10,8))
nx.draw_networkx_edges(G, pos, alpha=0.6)
nx.draw_networkx_nodes(G, pos, node_size=node_sizes)
nx.draw_networkx_labels(G, pos, labels, font_size=8)

# overlay top corridors (thicker edges)
for idx,(target,info) in enumerate(top_corridors):
    path = info['path']
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, alpha=0.9)

    # annotate flattened table strings slightly offset for readability
    for j,p in enumerate(path):
        x,y = pos[p]
        txt = str(tuple(p[0]+p[1]))
        plt.text(x, y-0.03 - 0.02*j, txt, fontsize=7, ha='center')

plt.title("Strict-positive 2x3 fiber graph with top corridors highlighted\nNodes labeled by index; node sizes ∝ π(T)")
plt.axis('off')
outpath = "strict_positive_corridors.png"
plt.savefig(outpath, bbox_inches='tight', dpi=200)
plt.show()

# Print a small summary table of top corridors
corr_summary = []
for idx,(target,info) in enumerate(top_corridors):
    corr_summary.append({
        "rank": idx+1,
        "target_flat": tuple(target[0]+target[1]),
        "path_length": len(info['path']),
        "min_edge_cap": info['min_edge_cap'],
        "path_flat": [tuple(p[0]+p[1]) for p in info['path']]
    })
print(pd.DataFrame(corr_summary).to_string(index=False))
print("\nSaved figure to:", outpath)
