import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from scipy.sparse import lil_matrix, csr_matrix 

# --- Configuration ---
N_TOTAL_NODES = 3500000
N_RUNS = 30
N_TIMESTEPS = 50
TOP_PERCENT = 0.01
# List of feature set sizes (K_final) to test
FEATURE_COUNT_CANDIDATES = [200, 500, 1000, 2000, 5000] 
LASSO_ALPHA = 0.01
TARGET_K_FOR_DETAIL = 1000 # The K value for which we will perform the detailed Lasso analysis
SAMPLE_SIZE = 10 # Number of Node IDs to print as a sample

# --- Global Mappings ---
GLOBAL_SELECTED_NODES = []
GLOBAL_IDX_TO_H_COL = {}
GLOBAL_IDX_TO_P_COL = {}


def simulate_raw_data(n_runs, n_timesteps, n_total_nodes, top_percent):
    """
    Simulates the raw filtered data, returning the data structure and a count of 
    how often each node index appears.
    """
    print("1. Simulating raw filtered data (30 runs x 50 time steps)...")
    raw_data = []
    node_frequency = defaultdict(int) 
    N_SAVED_NODES = int(n_total_nodes * top_percent)

    for run_id in range(n_runs):
        run_data = []
        # Target Outcome (Y) for the run
        y_target = 1.0 if run_id % 3 == 0 else 0.0 
        
        for t in range(n_timesteps):
            # Introduce a core set of "important" nodes that appear frequently
            base_idx = run_id * 1000
            important_indices = np.arange(base_idx, base_idx + N_SAVED_NODES)
            
            # Introduce a smaller set of random, less frequent nodes
            random_indices = np.random.randint(0, n_total_nodes, size=N_SAVED_NODES // 10)
            
            indices = np.unique(np.concatenate([important_indices, random_indices]))
            if len(indices) > N_SAVED_NODES:
                 indices = indices[:N_SAVED_NODES]

            # The values h_top and p_top are correlated with y_target for the important nodes
            h_top = np.random.rand(len(indices)) * (1 + 0.5 * y_target)
            p_top = np.random.rand(len(indices)) * (1 - 0.5 * y_target)
            
            time_step_data = []
            for h, p, i in zip(h_top, p_top, indices):
                time_step_data.append((h, p, i))
                node_frequency[i] += 1
            
            run_data.append({'time_step_data': time_step_data, 'y_target': y_target})
        
        raw_data.append(run_data)
    
    print(f"   -> Data Structure: {N_RUNS} runs, {N_TIMESTEPS} time steps each.")
    return raw_data, node_frequency

def filter_and_build_feature_set(node_frequency, target_k):
    """Filters the full I_master set down to the top K most frequently observed nodes."""
    
    # Sort nodes by how frequently they appeared (descending)
    sorted_nodes = sorted(node_frequency.items(), key=lambda item: item[1], reverse=True)
    
    # Select only the top K nodes for the final feature set
    final_master_indices = [idx for idx, count in sorted_nodes[:target_k]]
    K_final = len(final_master_indices)
    
    # Create the mapping for feature column definition
    # H columns are 0 to K-1, P columns are K to 2K-1
    idx_to_h_col = {idx: i for i, idx in enumerate(final_master_indices)}
    idx_to_p_col = {idx: i + K_final for i, idx in enumerate(final_master_indices)}
    
    return final_master_indices, K_final, idx_to_h_col, idx_to_p_col

def build_fixed_feature_matrix(raw_data, K_final, idx_to_h_col, idx_to_p_col):
    """Constructs the fixed feature matrix X using a SPARSE structure."""
    
    total_time_steps = N_RUNS * N_TIMESTEPS
    total_features = 2 * K_final
    
    # Initialize a Sparse Matrix (List of Lists format for efficient filling)
    X_sparse = lil_matrix((total_time_steps, total_features))
    Y = []
    
    row_index = 0
    for run in raw_data:
        for step in run:
            Y.append(step['y_target']) 
            
            for h_top, p_top, idx in step['time_step_data']:
                
                # Check if this node is one of the retained K_final nodes
                if idx in idx_to_h_col:
                    h_col = idx_to_h_col[idx]
                    p_col = idx_to_p_col[idx]
                    
                    X_sparse[row_index, h_col] = h_top
                    X_sparse[row_index, p_col] = p_top
                
            row_index += 1
            
    # Convert to Compressed Sparse Row format for faster computation
    X = csr_matrix(X_sparse)
    Y = np.array(Y)
    
    return X, Y

def train_and_evaluate_lasso(X, Y):
    """Trains a Lasso model and returns the test R-squared score and the trained model."""
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    lasso_model = Lasso(alpha=LASSO_ALPHA, max_iter=10000, tol=0.001)
    lasso_model.fit(X_train, Y_train)
    
    r_squared = lasso_model.score(X_test, Y_test)
    
    # Return both the score and the model instance
    return r_squared, lasso_model

def get_predicted_nodes_with_coefficients(lasso_model, K_final_actual, idx_to_h_col, idx_to_p_col):
    """
    Returns a list of predicted nodes with their associated H and P coefficients.
    """
    coefficients = lasso_model.coef_
    K_nodes = K_final_actual
    
    # Invert the column-to-index maps
    h_col_to_idx = {v: k for k, v in idx_to_h_col.items()}
    p_col_to_idx = {v: k for k, v in idx_to_p_col.items()}
    
    node_data = defaultdict(lambda: {'H_coeff': 0.0, 'P_coeff': 0.0})
    
    # 1. Check H features (first K columns)
    for col_idx in range(K_nodes):
        coeff = coefficients[col_idx]
        if abs(coeff) > 1e-6:
            node_idx = h_col_to_idx[col_idx]
            node_data[node_idx]['H_coeff'] = coeff
            
    # 2. Check P features (last K columns)
    for col_idx in range(K_nodes, 2 * K_nodes):
        coeff = coefficients[col_idx]
        if abs(coeff) > 1e-6:
            node_idx = p_col_to_idx[col_idx]
            node_data[node_idx]['P_coeff'] = coeff
            
    # Format for output
    predicted_nodes_list = []
    for idx, data in node_data.items():
        predicted_nodes_list.append({
            'Node_ID': idx,
            'H_coeff': data['H_coeff'],
            'P_coeff': data['P_coeff']
        })
            
    # Sort by Node ID for stable output
    return sorted(predicted_nodes_list, key=lambda x: x['Node_ID'])


def compare_predicted_vs_selected_nodes(lasso_model, K_final_actual, selected_nodes_list, idx_to_h_col, idx_to_p_col):
    """Compares the initial K selected nodes vs the final Lasso-predicted nodes and prints IDs and coefficients."""
    
    predicted_node_details = get_predicted_nodes_with_coefficients(lasso_model, K_final_actual, idx_to_h_col, idx_to_p_col)
    
    selected_set = set(selected_nodes_list)
    predicted_set = {d['Node_ID'] for d in predicted_node_details}
    
    # Intersection: nodes that were selected by frequency AND used by Lasso
    active_in_selected_count = len(selected_set.intersection(predicted_set))
    
    print(f"\n--- 6. Lasso Prediction Analysis (K={K_final_actual}) ---")
    
    print("\nConceptual Mapping Confirmation:")
    print(f"   -> The 'Selected Nodes' list below is the direct equivalent of your 'sims.top_priority.idx' for K={K_final_actual}.")
    print("   -> The 'Predictive Nodes' are the subset used by the final Lasso model (non-zero coefficients).")
    
    # A. Selected Nodes (Frequency Filter / sims.top_priority.idx)
    print(f"\nA. Candidate Nodes (sims.top_priority.idx): {len(selected_nodes_list)} nodes")
    print(f"   Sample Node IDs: {selected_nodes_list[:SAMPLE_SIZE]}")

    # B. Detailed Predictive Nodes (Lasso Non-Zero Coeff)
    print(f"\nB. Predictive Nodes (Lasso Selected): {len(predicted_node_details)} nodes")
    
    pruning_ratio = (1 - active_in_selected_count / len(selected_set)) * 100
    
    print(f"   Lasso Pruning Percentage (on candidate set): {pruning_ratio:.2f}%")
    print("-" * 50)
    print(f"| {'Node ID':<10} | {'H Coeff':<15} | {'P Coeff':<15} |")
    print("-" * 50)
    
    # Print sample of the predicted nodes with coefficients
    for detail in predicted_node_details[:SAMPLE_SIZE]:
        h_coeff_str = f"{detail['H_coeff']:+.4f}"
        p_coeff_str = f"{detail['P_coeff']:+.4f}"
        print(f"| {detail['Node_ID']:<10} | {h_coeff_str:<15} | {p_coeff_str:<15} |")

    print("-" * 50)
    
def analyze_overlap(selected_node_sets):
    """
    Analyzes the overlap between the feature sets selected at different K thresholds.
    (Kept for completeness, as requested in the initial workflow)
    """
    print("\n--- 5. Feature Set Overlap Analysis (Stability Check) ---")
    
    sets = {k: set(v) for k, v in selected_node_sets.items()}
    k_base = min(sets.keys())
    base_set = sets[k_base]
    base_size = len(base_set)
    
    print(f"Base Feature Set (K={k_base}) size: {base_size} nodes.")
    print("---------------------------------------------------------")
    print("| Comparison K | Overlap Count | Overlap Percentage |")
    print("---------------------------------------------------------")

    for k_compare, compare_set in sets.items():
        if k_compare > k_base:
            overlap_count = len(base_set.intersection(compare_set))
            overlap_percent = (overlap_count / base_size) * 100
            print(f"| {k_compare:<12} | {overlap_count:<13} | {overlap_percent:.2f}%          |")
    
    print("---------------------------------------------------------")
    print("\nInterpretation: High overlap confirms the stability of the frequency-based ranking.")


def main():
    """Main execution function to optimize feature set size and analyze stability."""
    global GLOBAL_SELECTED_NODES, GLOBAL_IDX_TO_H_COL, GLOBAL_IDX_TO_P_COL

    # 1. Simulate Raw Data and get frequency counts (this is done only once)
    raw_data, node_frequency = simulate_raw_data(N_RUNS, N_TIMESTEPS, N_TOTAL_NODES, TOP_PERCENT)
    
    full_unique_size = len(node_frequency)
    print(f"\n2. Full unique set (I_master) size: {full_unique_size}")
    
    selected_node_sets = {}
    
    print("\n3. Starting Feature Set Optimization...")
    print("--------------------------------------------------")
    print(f"| Feature Count (K) | Total Features (2K) | R-squared |")
    print("--------------------------------------------------")

    last_lasso_model = None
    last_K_final = None

    for k_final in FEATURE_COUNT_CANDIDATES:
        
        # 3a. Filter the Master Index Set (I_master) to the target K
        master_index_list, K_final_actual, idx_to_h_col, idx_to_p_col = filter_and_build_feature_set(
            node_frequency, k_final
        )
        selected_node_sets[K_final_actual] = master_index_list
        
        # 3b. Build Fixed, SPARSE Feature Matrix X
        X, Y = build_fixed_feature_matrix(raw_data, K_final_actual, idx_to_h_col, idx_to_p_col)
        
        # 3c. Train Lasso and Evaluate
        r_squared, lasso_model = train_and_evaluate_lasso(X, Y)
        
        print(f"| {K_final_actual:<17} | {2 * K_final_actual:<19} | {r_squared:.4f}    |")

        if k_final == TARGET_K_FOR_DETAIL:
             # Store results for the detailed comparison
             last_lasso_model = lasso_model
             last_K_final = K_final_actual
             GLOBAL_SELECTED_NODES = master_index_list
             GLOBAL_IDX_TO_H_COL = idx_to_h_col
             GLOBAL_IDX_TO_P_COL = idx_to_p_col

    print("--------------------------------------------------")
    print("\n4. Optimization Complete.")
    
    # 5. Run Overlap Analysis (Feature Set Stability)
    analyze_overlap(selected_node_sets)
    
    # 6. Run Prediction Analysis (Lasso Pruning)
    if last_lasso_model is not None:
        compare_predicted_vs_selected_nodes(
            last_lasso_model, 
            last_K_final, 
            GLOBAL_SELECTED_NODES, 
            GLOBAL_IDX_TO_H_COL, 
            GLOBAL_IDX_TO_P_COL
        )
    else:
        print(f"\nSkipping detailed prediction analysis: TARGET_K_FOR_DETAIL ({TARGET_K_FOR_DETAIL}) was not in the candidates list.")

if __name__ == '__main__':
    main()