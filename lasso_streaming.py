import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from io import StringIO # Used for capturing results as a string

# --- Configuration ---
NUM_SAMPLES = 3_500_000  # 3.5 million samples
NUM_FEATURES = 200       # The number of brain nodes (features)
BATCH_SIZE = 1000        # Smaller batches ensure lower peak memory usage
NODE_NAMES = [f'Node_{i+1}' for i in range(NUM_FEATURES)]
ALPHA = 0.0001           # Regularization strength (L1 penalty weight)

print(f"Configuration: {NUM_SAMPLES} samples, {NUM_FEATURES} features.")
print(f"Using BATCH_SIZE={BATCH_SIZE} for out-of-core learning.")
print("-" * 50)

# 1. Mock Data Generator (Simulates Streaming from Disk)
def data_generator(num_samples, num_features, batch_size):
    """Generates synthetic data batches (X, y) to simulate reading from disk."""
    total_batches = (num_samples + batch_size - 1) // batch_size
    print(f"Generating {total_batches} total batches...")
    
    # We define the true weights here again so the generator uses them
    true_weights = np.zeros(num_features)
    true_weights[4] = 2.5  # Node 5
    true_weights[19] = -1.0 # Node 20
    true_weights[149] = 1.5 # Node 150
    
    for i in range(total_batches):
        current_batch_size = min(batch_size, num_samples - i * batch_size)
        
        # Simulate 'Entropy Features' (X): values between 0 and 1
        X_batch = np.random.rand(current_batch_size, num_features)
        
        # Simulate the 'Outcome' (y): simple linear model + noise
        y_batch = X_batch @ true_weights + np.random.normal(0, 0.5, current_batch_size)
        
        yield X_batch, y_batch, i + 1

# 2. Model Setup
model = SGDRegressor(
    loss='squared_error',   # Standard linear regression loss
    penalty='l1',           # Lasso regularization (L1 penalty)
    alpha=ALPHA,            # Regularization parameter
    max_iter=1000,          
    tol=1e-3,
    random_state=42,
    warm_start=True
)
scaler = StandardScaler()
X_batch_scaled_final = None # Will store the final batch data for scoring

# 3. Out-of-Core Training Loop (Guaranteed Low Memory)
X_init, y_init, _ = next(data_generator(NUM_SAMPLES, NUM_FEATURES, BATCH_SIZE))
print("Fitting initial scaler on first batch for feature standardization...")

scaler.fit(X_init)
model.partial_fit(scaler.transform(X_init), y_init)
print("Model initialized and first batch trained.")

NUM_EPOCHS = 5
print(f"Starting {NUM_EPOCHS} epochs of training...")

for epoch in range(NUM_EPOCHS):
    print(f"\n--- EPOCH {epoch + 1}/{NUM_EPOCHS} ---")
    
    data_stream = data_generator(NUM_SAMPLES, NUM_FEATURES, BATCH_SIZE)
    
    for X_batch, y_batch, batch_num in data_stream:
        # Scale the batch using the fitted scaler
        X_batch_scaled = scaler.transform(X_batch)
        
        # Train the model with the current batch
        model.partial_fit(X_batch_scaled, y_batch)
        
        # Keep track of the last processed batch for final scoring
        if epoch == NUM_EPOCHS - 1 and batch_num == (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE:
             X_batch_scaled_final = X_batch_scaled
             y_batch_final = y_batch

        if batch_num % 1000 == 0:
            print(f"  > Batch {batch_num} trained...")

print("\n--- Training Complete ---")

# 4. Analysis Function (The 'reverse map analysis' step)
def analyze_and_report_results(model, X_scaled_data, y_true_data, node_names):
    """
    Extracts non-zero coefficients (Exceptional Locus) and calculates performance.
    Returns the formatted analysis report as a single string.
    """
    output = StringIO()
    
    try:
        coefficients = model.coef_
    except NotFittedError:
        output.write("Error: Model was not fitted properly.")
        return output.getvalue()

    # Filter for coefficients significant enough to be non-zero
    THRESHOLD = 1e-4 
    significant_coeffs = coefficients[np.abs(coefficients) > THRESHOLD]
    significant_indices = np.where(np.abs(coefficients) > THRESHOLD)[0]

    # Create a DataFrame for easy reading
    results = pd.DataFrame({
        'Node Name': [node_names[i] for i in significant_indices],
        'Coefficient': significant_coeffs.round(4)
    }).sort_values(by='Coefficient', ascending=False)

    output.write(f"\nExceptional Locus (Non-Zero Coefficients): {len(significant_coeffs)} nodes found\n")
    output.write(results.to_markdown(index=False))
    output.write("\n" + "-" * 50 + "\n")

    # Check performance (R-squared)
    score = model.score(X_scaled_data, y_true_data)
    output.write(f"R-squared (on final batch): {score:.4f}\n")
    output.write("\nSuccessfully performed Lasso GLM training in a memory-safe, out-of-core manner.")
    
    return output.getvalue()


# Run the batch analysis and print the result
print(analyze_and_report_results(model, X_batch_scaled_final, y_batch_final, NODE_NAMES))
