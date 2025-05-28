import numpy as np

def generate_synthetic_data(num_samples=1000, sequence_length=1, num_features=128, num_outputs=10, noise_level=0.1):
    """
    Generates synthetic neural data (X) and behavioral outputs (Y).
    """
    # Ensure X has the correct shape: (num_samples, sequence_length, num_features)
    X = np.random.randn(num_samples, sequence_length, num_features) * 0.5  # Ensure last dim = num_features
    
    # True weights for transformation
    true_weights = np.random.randn(num_features, num_outputs) * 0.1
    
    # Compute Y using only the last time step
    Y = np.tanh(X[:, -1, :] @ true_weights)  
    
    # Add noise
    Y += noise_level * np.random.randn(num_samples, num_outputs)
    
    return X, Y

# Generate training and validation datasets
train_X, train_Y = generate_synthetic_data(num_samples=5000)
valid_X, valid_Y = generate_synthetic_data(num_samples=1000)

# Save to .npy files
# Save using np.savez to store multiple arrays in a single file
np.savez("train.npz", X=train_X, Y=train_Y)
np.savez("valid.npz", X=valid_X, Y=valid_Y)

print("Synthetic train.npz and valid.npz files created.")