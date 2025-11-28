import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# --- Configuration ---
N_TOTAL_NODES = 3500000  # The total population of nodes (3.5 million)
N_RUNS = 30              # Number of simulation runs
N_TIMESTEPS = 50         # Time steps per run
TOP_PERCENT = 0.01       # The 1% filter initially applied (35,000 nodes saved per time step)

def simulate_raw_data(n_runs, n_timesteps, n_total_nodes, top_percent):
    """
    Simulates the raw data structure: a list of runs, where each run is a list of time steps,
    and each time step contains the top 1% (h_top, p_top, idx) tuples.
    
    The indices (idx) are intentionally varied across time steps and runs
    to simulate the dynamic nature of the original entropy-based filtering.
    """
    print("1. Simulating raw filtered data (30 runs x 50 time steps)...")
    raw_data = []
    
    # Calculate the number of saved nodes per time step
    N_SAVED_NODES = int(n_total_nodes * top_percent)

    for run_id in range(n_runs):
        run_data = []
        # Simulate a simple, binary target variable (Y) for the run outcome
        # e.g., 1.0 if the perturbation was successful, 0.0 otherwise.
        y_target = 1.0 if run_id % 3 == 0 else 0.0 
        
        for t in range(n_timesteps):
            # Generate indices that are 'important' for this time step.
            # We shift the base index for each run to ensure different sets are considered
            base_idx = run_id * 10000
            
            # This logic ensures some 'core' nodes are always saved, plus some random noise
            important_indices = np.arange(base_idx, base_idx + N_SAVED_NODES)
            random_indices = np.random.randint(0, n_total_nodes, size=N_SAVED_NODES // 10)
            
            # Combine, take unique indices, and trim to the required size
            indices = np.unique(np.concatenate([important_indices, random_indices]))
            if len(indices) > N_SAVED_NODES:
                 indices = indices[:N_SAVED_NODES]

            # Simulate the h_top and p_top values
            h_top = np.random.rand(len(indices)) * (1 + 0.5 * y_target) # Higher values in successful runs
            p_top = np.random.rand(len(indices)) * (1 - 0.5 * y_target) # Lower p_top in successful runs
            
            time_step_data = []
            for h, p, i in zip(h_top, p_top, indices):
                time_step_data.append((h, p, i))
            
            run_data.append({'time_step_data': time_step_data, 'y_target': y_target})
        
        raw_data.append(run_data)
    
    print(f"   -> Data Structure: {N_RUNS} runs, {N_TIMESTEPS} time steps each.")
    print(f"   -> Approx. {N_SAVED_NODES} nodes saved per time step.")
    return raw_data

def build_master_index_set(raw_data):
    """
    Implements Step 2: Gathers the union of all unique node indices (idx) across all runs.
    """
    print("\n2. Building the Master Index Set (I_master)...")
    all_unique_indices = set()
    
    for run in raw_data:
        for step in run:
            for _, _, idx in step['time_step_data']:
                all_unique_indices.add(idx)
                
    master_index_list = sorted(list(all_unique_indices))
    K = len(master_index_list)
    
    print(f"   -> Total unique 'important' nodes observed across all 30 runs: K = {K}")
    
    # Create the mapping for feature column definition
    # The first K columns are h_top, the next K columns are p_top
    idx_to_h_col = {idx: i for i, idx in enumerate(master_index_list)}
    idx_to_p_col = {idx: i + K for i, idx in enumerate(master_index_list)}
    
    return master_index_list, K, idx_to_h_col, idx_to_p_col

def build_fixed_feature_matrix(raw_data, K, idx_to_h_col, idx_to_p_col):
    """
    Implements Step 4: Creates the final, stable feature matrix X and target vector Y.
    """
    print("\n3. Constructing the Fixed Feature Matrix X...")
    
    total_time_steps = N_RUNS * N_TIMESTEPS
    total_features = 2 * K
    
    # Initialize the fixed feature matrix with zeros (Crucial for Strategy 2)
    X = np.zeros((total_time_steps, total_features))
    Y = []
    
    row_index = 0
    for run in raw_data:
        for step in run:
            # The target Y is the outcome of the entire run, applied to every time step
            Y.append(step['y_target']) 
            
            # Populate the feature vector (row) X[row_index, :]
            for h_top, p_top, idx in step['time_step_data']:
                # Get the fixed column index for this node 'idx'
                h_col = idx_to_h_col[idx]
                p_col = idx_to_p_col[idx]
                
                # Assign the observed values. If a node in I_master was NOT in the 
                # top 1% for this step, it remains 0.0 in the matrix.
                X[row_index, h_col] = h_top
                X[row_index, p_col] = p_top
                
            row_index += 1
            
    X = pd.DataFrame(X)
    Y = np.array(Y)
    
    print(f"   -> Final X Matrix Shape: {X.shape} (Time Steps x Features)")
    print(f"   -> Final Y Vector Shape: {Y.shape} (Target Outcomes)")
    return X, Y

def train_and_analyze_lasso(X, Y, master_index_list):
    """
    Implements Step 5: Trains a Lasso model and analyzes the resulting coefficients.
    """
    print("\n4. Training Lasso Model and Analyzing Feature Importance...")
    
    # Split data for proper validation
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Initialize Lasso model (alpha is the regularization strength)
    # A low alpha gives high feature selection, a higher alpha gives more sparsity.
    # We use a non-zero alpha to encourage sparsity and find the truly predictive nodes.
    lasso_model = Lasso(alpha=0.01, max_iter=10000)
    lasso_model.fit(X_train, Y_train)
    
    # --- Analysis ---
    K = len(master_index_list)
    h_coeffs = lasso_model.coef_[:K]
    p_coeffs = lasso_model.coef_[K:]
    
    # Find the indices of nodes with non-zero coefficients
    non_zero_h_indices = np.where(np.abs(h_coeffs) > 1e-9)[0]
    non_zero_p_indices = np.where(np.abs(p_coeffs) > 1e-9)[0]

    # Combine the features with the largest absolute coefficients
    important_nodes = []
    
    print(f"   -> Lasso R-squared on test set: {lasso_model.score(X_test, Y_test):.4f}")
    
    print("\n   --- Top 5 Predictive Nodes (by absolute coefficient sum) ---")
    
    # Calculate the importance score for each node (sum of its H and P coefficient magnitudes)
    node_importance = {}
    for i in range(K):
        node_importance[master_index_list[i]] = np.abs(h_coeffs[i]) + np.abs(p_coeffs[i])

    sorted_nodes = sorted(node_importance.items(), key=lambda item: item[1], reverse=True)
    
    for i, (idx, importance) in enumerate(sorted_nodes[:5]):
        h_coeff = h_coeffs[master_index_list.index(idx)]
        p_coeff = p_coeffs[master_index_list.index(idx)]
        print(f"   {i+1}. Node Index: {idx} | Total Importance: {importance:.4f}")
        print(f"      H_top Coeff (Magnitude): {h_coeff:.4f}")
        print(f"      P_top Coeff (Peak): {p_coeff:.4f}")
    
    # Interpretation: 
    # The coefficients above tell you the true, fixed set of nodes that predict the run outcome (Y).
    print("\n--- Interpretation ---")
    print(f"Out of K={K} consistently 'important' nodes, Lasso selected {len(non_zero_h_indices) + len(non_zero_p_indices)} non-zero coefficients.")
    print("These non-zero coefficients correspond to the nodes that are truly predictive of the target outcome (Y) over time.")
    print("This matrix X is now stable and includes the 'good nodes' you were worried about losing.")


def main():
    """Main execution function."""
    
    # 1. Simulate Raw Data
    raw_data = simulate_raw_data(N_RUNS, N_TIMESTEPS, N_TOTAL_NODES, TOP_PERCENT)
    
    # 2. Build Master Index Set (I_master)
    master_index_list, K, idx_to_h_col, idx_to_p_col = build_master_index_set(raw_data)
    
    # 3. Build Fixed Feature Matrix X and Target Y
    X, Y = build_fixed_feature_matrix(raw_data, K, idx_to_h_col, idx_to_p_col)
    
    # 4. Train Lasso
    train_and_analyze_lasso(X, Y, master_index_list)


if __name__ == '__main__':
    main()
