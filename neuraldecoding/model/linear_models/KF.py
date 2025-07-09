import numpy as np
import pickle
import torch
from tqdm import tqdm
from neuraldecoding.model.linear_models.LinearModel import LinearModel

class KalmanFilter(LinearModel):
    def __init__(self, model_params):
        """
        Constructs a Kalman Filter decoder object (can use numpy arrays, torch arrays, or torch dataloaders). 
        
        Parameters:
            model_params (dict) containing three keys, 'append_ones_y' (bool) which specifies whether or not to add column of ones for bias, 
            'device' (bool) which specifies what device to train/run the model on, "num_channels" (int) which specifies the number of channels in the input data, 
            "num_outputs" (int) which specifies the number of outputs, "start_y" (list) which specifies the initial yhat, and "zero_position_uncertainty" 
            (bool) which specifies whether to zero position uncertainty for online predictions
        """

        self.A, self.C, self.W, self.Q = None, None, None, None
        self.Pt = None
        self.append_ones_y = model_params.get("append_ones_y", True)
        self.device = model_params.get("device", "cpu")
        self.input_size = model_params.get("num_channels", 10)
        self.output_size = model_params.get("num_outputs", 3)
        self.start_y = torch.tensor(model_params.get("start_y", [0] * self.output_size), device=self.device, dtype=torch.float32)
        self.last_yhat = None
        self.zero_position_uncertainty = model_params.get("zero_position_uncertainty", True)

    def __call__(self, data):
        """
        Makes the instance callable and returns the result of forward pass.

        Parameters:
            data (ndarray): Observation data for prediction, expected size [batch_size, num_inputs, sequence_length]

        Returns:
            ndarray: Prediction results, size [batch_size, num_outputs, sequence_length]
        """
        return self.forward(data)
    

    def train_step(self, input_data: tuple[torch.Tensor, torch.Tensor]):
        """
        Trains the matrices in the model. If append_ones_y is true, a column of ones is added to calculate bias. 

        Parameters:
            input_data (tuple): A tuple of (x, y)
                - x (torch.Tensor) size [num_samples, num_inputs]: observation features (neural, EMG, etc.)
                - y (torch.Tensor) size [num_samples, num_outputs]: ground truth output (kinematic, etc.)

        """
        # unpack input data
        x, y = input_data

        if self.A is not None:
            raise ValueError("Tried to train a model that's already trained ")
        
        if self.append_ones_y:
            y = torch.cat((y, torch.ones([y.shape[0], 1], device=y.device, dtype=y.dtype)), dim=1)
        
        # Build the state transition matrix A.
        self.A = torch.zeros((self.output_size, self.output_size), device=self.device, dtype=y.dtype)
        if self.append_ones_y:
            self.A = torch.zeros((self.output_size + 1, self.output_size + 1), device=self.device, dtype=y.dtype)
        pos_idx = list(range(self.output_size // 2))

        # Identity for position states.
        for i in pos_idx:
            self.A[i, i] = 1.0
        # Set cross term: A[corresponding position, velocity] = 1.0.
        vel_idx = list(range(self.output_size // 2, self.output_size))
        for i, j in zip(pos_idx, vel_idx):
            self.A[i, j] = 1.0

        # Only add bias if append_ones_y is true
        if self.append_ones_y:
            self.A[-1, -1] = 1.0 # Bias state remains constant.

        # Get the velocity transition submatrix
        vel_transition = (y[1:, vel_idx].T @ y[:-1, vel_idx]) @ torch.linalg.inv(
            y[:-1, vel_idx].T @ y[:-1, vel_idx]
        )
        # Fill the velocity submatrix in A using the velocity indices
        self.A[vel_idx[0] : vel_idx[-1] + 1, vel_idx[0] : vel_idx[-1] + 1] = vel_transition
        
        # Compute trajectory noise covariance W.
        num_samples = y.shape[0]
        y_tm1 = y[:-1, :].T  # shape: (output_size, num_samples - 1)
        y_t = y[1:, :].T  # shape: (output_size, num_samples - 1)
        resid = y_t - self.A @ y_tm1
        self.W = (resid @ resid.T) / (num_samples - 1)

        # Zero out position and bias parts of W
        if self.append_ones_y:
            pos_and_bias_idx = pos_idx + [self.output_size]
        else:
            pos_and_bias_idx = pos_idx
        self.W[pos_and_bias_idx, :] = 0
        self.W[:, pos_and_bias_idx] = 0

        # Compute the neural observation matrix C via least squares:
        # C = (x.T @ y) @ inv(y.T @ y)
        self.C = (x.T @ y) @ torch.linalg.inv(y.T @ y)

        # Compute observation noise covariance Q.
        Q_resid = x - y @ self.C.T
        self.Q = (Q_resid.T @ Q_resid) / num_samples
        
        self.reinitialize()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass, by calling a predict method (torch). 
        The method depends on the dimension of the input tensor.

        Parameters:
            input (ndarray) size [batch_size, num_inputs, sequence_length] or [sequence_length, num_inputs] or [num_inputs]
        
        Returns:
            yhat (ndarray) prediction of size [batch_size, num_outputs, sequence_length] or [sequence_length, num_outputs] or [num_outputs]
        """
        yhat = None
        if input.dim() == 1:
            yhat = self.predict(input.unsqueeze(0)).squeeze(0)
            # Here, Pt is not reinitialized to allow for online prediction
        elif input.dim() == 2:
            yhat = self.predict(input)
            # Reinitialize after predicting many datapoints
            self.reinitialize()
        elif input.dim() == 3:
            yhat = self.predict_batch(input)
            # Reinitialize after a full batch
            self.reinitialize()
        else:
            raise ValueError(f"Input tensor must be 1D, 2D, or 3D, got shape {input.shape}")
            
        return yhat
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass, returning a prediction for all input datapoints.

        Parameters:
            x (ndarray) size [sequence_length, num_inputs]

        Returns:
            yhat (ndarray) prediction of size [sequence_length, num_outputs]
        """
        if x.dim() != 2:
            raise ValueError(f"Input tensor must be 2D (sequence_length, num_inputs), got shape {x.shape}")
        
        if x.shape[1] != self.input_size:
            raise ValueError(f"Input feature dimension mismatch: expected {self.input_size}, got {x.shape[1]}")

        if self.append_ones_y:
            # Add the ones
            all_yhat = torch.zeros((x.shape[0], self.output_size + 1), device=self.device, dtype=x.dtype)
            all_yhat[:, -1] = 1.0
        else:
            all_yhat = torch.zeros((x.shape[0], self.output_size), device=self.device, dtype=x.dtype)
            
        if self.last_yhat is None:
            if self.append_ones_y:
                self.last_yhat = torch.cat((self.start_y, torch.ones(1, device=self.device, dtype=x.dtype)), dim=0)
            else:
                self.last_yhat = self.start_y.clone().to(dtype=x.dtype)

        for t in range(x.shape[0]):
            yt = self.last_yhat @ self.A.T                                # predict new state
            self.Pt = self.A @ self.Pt @ self.A.T + self.W                        # compute error covariance
            if self.zero_position_uncertainty:
                # Assuming the first half of the outputs are position and the second half are velocity
                self.Pt[:self.output_size // 2, :self.output_size // 2] = 0
            K = torch.linalg.lstsq((self.C @ self.Pt @ self.C.T + self.Q).T,
                                (self.Pt @ self.C.T).T, rcond=None)[0].T     # compute kalman gain, where B/A = (A'\B')'
            all_yhat[t, :] = yt + K @ (x[t, :] - self.C @ yt)	        # update state estimate
            self.Pt = (torch.eye(self.A.shape[0], device=self.A.device, dtype=self.A.dtype) - K @ self.C) @ self.Pt
            self.last_yhat = all_yhat[t, :].clone()

        if self.append_ones_y:
            # Remove the ones
            all_yhat = all_yhat[:, :-1]

        return all_yhat
        
        

    def predict_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        Runs a forward pass, returning a prediction for all input datapoints.

        Parameters:
            x (torch.Tensor) size [batch_size, num_inputs, sequence_length]

        Returns:
            yhat (torch.Tensor) prediction of size [batch_size, num_outputs, sequence_length]
        """

        if x.dim() != 3:
            raise ValueError(f"Input tensor must be 3D (batch_size, num_inputs, sequence_length), got shape {x.shape}")
        if x.shape[1] != self.input_size:
            raise ValueError(f"Input feature dimension mismatch: expected {self.input_size}, got {x.shape[1]}")

        all_yhat = torch.zeros((x.shape[0], self.output_size, x.shape[2]), device=self.device, dtype=x.dtype)
        for b in range(x.shape[0]):
            yhat = self.predict(x[b, :, :].T)
            all_yhat[b, :, :] = yhat.T
        return all_yhat

    def save_model(self, fpath):
        """
        Saves the model in its current state at the specified filepath

        Parameters:
            filepath (path-like object) indicates the file path to save the model in
        """
        model_dict = {
            "A": self.A,
            "C": self.C,
            "W": self.W,
            'Q': self.Q,
            "Model": "KF"
        }

        with open(fpath, "wb") as file:
            pickle.dump(model_dict, file)

    def load_model(self, fpath):
        """
        Load model parameters from a specified location

        Parameters:
            filepath (path-like object) indicates the file path to load the model from
        """

        with open(fpath, "rb") as file:
           model_dict = pickle.load(file)

        if model_dict["Model"] != "KF":
            raise Exception("Tried to load model that isn't a Kalman Filter Instance")
        
        self.A = model_dict['A']
        self.C = model_dict['C']
        self.W = model_dict['W']
        self.Q = model_dict['Q']
        self.reinitialize()

    def reinitialize(self):
        self.Pt = self.W.clone()
        self.last_yhat = None

if __name__ == "__main__":
    print("=" * 60)
    print("KALMAN FILTER DEMONSTRATION AND TESTING")
    print("=" * 60)
    
    # Initialize the Kalman Filter with example parameters
    print("\n1. INITIALIZING KALMAN FILTER")
    print("-" * 30)
    model_params = {
        "append_ones_y": True,              # Add bias term
        "device": "cpu",                    # Use CPU for computation
        "num_channels": 10,                 # 10 input channels (e.g., neural channels)
        "num_outputs": 4,                   # 4 outputs (e.g., 2D position + 2D velocity)
        "start_y": [0, 0, 0, 0],           # Initial state estimate
        "zero_position_uncertainty": True   # Zero position uncertainty for online predictions
    }
    
    model = KalmanFilter(model_params=model_params)
    print(f"✓ Created KalmanFilter with {model_params['num_channels']} input channels")
    print(f"✓ Output dimensions: {model_params['num_outputs']} (likely position + velocity)")
    print(f"✓ Device: {model_params['device']}")
    print(f"✓ Initial state: {model_params['start_y']}")
    
    # Generate synthetic training data and train the model
    print("\n2. TRAINING THE MODEL")
    print("-" * 30)
    train_x = torch.randn(1000, 10)  # 1000 time points, 10 neural channels
    train_y = torch.randn(1000, 4)   # 1000 time points, 4 kinematic outputs
    
    print(f"Training data: X={train_x.shape}, Y={train_y.shape}")
    print("Training Kalman Filter parameters (A, C, W, Q matrices)...")
    
    model.train_step((train_x, train_y))
    print("✓ Model training completed successfully")
    print(f"✓ State transition matrix A: {model.A.shape}")
    print(f"✓ Observation matrix C: {model.C.shape}")
    print(f"✓ Process noise W: {model.W.shape}")
    print(f"✓ Observation noise Q: {model.Q.shape}")
    
    # Test different input formats
    print("\n3. TESTING DIFFERENT INPUT FORMATS")
    print("-" * 30)
    
    # Test 1: Batched input
    print("Test 1: Batched input")
    test_x = torch.randn(2, 10, 100)  # 2 batches, 10 channels, 100 time points
    yhat = model.forward(test_x)
    print(f"  Input shape: {test_x.shape} (batch_size=2, channels=10, time_points=100)")
    print(f"  Output shape: {yhat.shape} (batch_size=2, outputs=4, time_points=100)")
    
    # Test 2: Single trial (no batch dimension)
    print("\nTest 2: Single trial (no batch dimension)")
    test_x_nobatch = torch.randn(100, 10)  # 100 time points, 10 channels
    yhat_nobatch = model.forward(test_x_nobatch)
    print(f"  Input shape: {test_x_nobatch.shape} (time_points=100, channels=10)")
    print(f"  Output shape: {yhat_nobatch.shape} (time_points=100, outputs=4)")
    
    # Test 3: Single time point
    print("\nTest 3: Single time point (online prediction)")
    test_x_singlepoint = torch.randn(10)  # Just 10 channels
    yhat_singlepoint = model.forward(test_x_singlepoint)
    print(f"  Input shape: {test_x_singlepoint.shape} (channels=10)")
    print(f"  Output shape: {yhat_singlepoint.shape} (outputs=4)")
    
    # Test 4: Consistency check - manual vs batch prediction
    print("\n4. CONSISTENCY VERIFICATION")
    print("-" * 30)
    print("Verifying that single-point predictions match batch processing...")
    print("(Note: Model state is reset before each test for fair comparison)")
    
    # Reset model state before batch prediction
    model.reinitialize()
    yhat_nobatch_fresh = model.forward(test_x_nobatch)
    
    # Reset model state before manual prediction (ensures same starting conditions)
    model.reinitialize()
    yhat_nobatch_manual = torch.zeros((test_x_nobatch.shape[0], 4))
    for i in range(test_x_nobatch.shape[0]):
        yhat_nobatch_manual[i, :] = model.forward(test_x_nobatch[i, :])
    
    # Check if manual predictions match batch predictions
    is_close = torch.isclose(yhat_nobatch_manual, yhat_nobatch_fresh).all()
    
    print(f"Manual prediction shape: {yhat_nobatch_manual.shape}")
    print(f"Batch prediction shape: {yhat_nobatch_fresh.shape}")
    print(f"Predictions match within tolerance: {'✓ YES' if is_close else '✗ NO'}")
    
    if not is_close:
        max_diff = torch.max(torch.abs(yhat_nobatch_manual - yhat_nobatch_fresh)).item()
        print(f"Maximum difference: {max_diff:.2e}")
        raise AssertionError("Consistency test failed!")
    
    # Test 5: Parameter Variations
    print("\n5. PARAMETER VARIATION TESTS")
    print("-" * 30)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    devices_to_test = ["cpu"]
    if cuda_available:
        devices_to_test.append("cuda")
        print("✓ CUDA detected - will test both CPU and CUDA")
    else:
        print("! CUDA not available - testing CPU only")
    
    # Test different parameter combinations
    test_configs = [
        {"append_ones_y": True, "zero_position_uncertainty": True, "name": "With bias + zero pos uncertainty"},
        {"append_ones_y": True, "zero_position_uncertainty": False, "name": "With bias, no zero pos uncertainty"}, 
        {"append_ones_y": False, "zero_position_uncertainty": True, "name": "No bias + zero pos uncertainty"},
        {"append_ones_y": False, "zero_position_uncertainty": False, "name": "No bias, no zero pos uncertainty"},
    ]
    
    test_results = []
    
    for device in devices_to_test:
        print(f"\nTesting on device: {device.upper()}")
        
        for config in test_configs:
            print(f"  Testing: {config['name']}")
            
            try:
                # Create model with specific parameters
                test_params = {
                    "append_ones_y": config["append_ones_y"],
                    "device": device,
                    "num_channels": 8,  # Slightly different size for variety
                    "num_outputs": 2,   # 1D position + 1D velocity for simpler testing
                    "start_y": [0, 0],
                    "zero_position_uncertainty": config["zero_position_uncertainty"]
                }
                
                test_model = KalmanFilter(model_params=test_params)
                
                # Generate test data on appropriate device
                if device == "cuda":
                    train_x_test = torch.randn(500, 8, device=device)
                    train_y_test = torch.randn(500, 2, device=device)
                    pred_x_test = torch.randn(50, 8, device=device)
                else:
                    train_x_test = torch.randn(500, 8)
                    train_y_test = torch.randn(500, 2)
                    pred_x_test = torch.randn(50, 8)
                
                # Train and predict
                test_model.train_step((train_x_test, train_y_test))
                pred_y = test_model.forward(pred_x_test)
                
                # Verify output shapes and properties
                expected_output_shape = (50, 2)
                shape_correct = pred_y.shape == expected_output_shape
                is_finite = torch.isfinite(pred_y).all()
                on_correct_device = pred_y.device.type == device
                
                # Check matrix dimensions based on append_ones_y
                if config["append_ones_y"]:
                    expected_A_size = (3, 3)  # 2 outputs + 1 bias
                    expected_C_size = (8, 3)  # 8 inputs, 3 states
                else:
                    expected_A_size = (2, 2)  # 2 outputs only
                    expected_C_size = (8, 2)  # 8 inputs, 2 states
                
                A_shape_correct = test_model.A.shape == expected_A_size
                C_shape_correct = test_model.C.shape == expected_C_size
                
                all_checks_passed = all([shape_correct, is_finite, on_correct_device, A_shape_correct, C_shape_correct])
                
                if all_checks_passed:
                    print(f"    ✓ PASSED - Output: {pred_y.shape}, A: {test_model.A.shape}, C: {test_model.C.shape}")
                    test_results.append(f"✓ {device.upper()} + {config['name']}: PASSED")
                else:
                    issues = []
                    if not shape_correct:
                        issues.append(f"shape {pred_y.shape} != {expected_output_shape}")
                    if not is_finite:
                        issues.append("non-finite values")
                    if not on_correct_device:
                        issues.append(f"wrong device {pred_y.device}")
                    if not A_shape_correct:
                        issues.append(f"A shape {test_model.A.shape} != {expected_A_size}")
                    if not C_shape_correct:
                        issues.append(f"C shape {test_model.C.shape} != {expected_C_size}")
                    
                    print(f"    ✗ FAILED - Issues: {', '.join(issues)}")
                    test_results.append(f"✗ {device.upper()} + {config['name']}: FAILED")
                    
            except Exception as e:
                print(f"    ✗ ERROR - {str(e)}")
                test_results.append(f"✗ {device.upper()} + {config['name']}: ERROR - {str(e)}")
    
    # Test summary for parameter variations
    print(f"\n6. PARAMETER VARIATION SUMMARY")
    print("-" * 30)
    for result in test_results:
        print(result)
    
    failed_tests = [r for r in test_results if "✗" in r]
    if failed_tests:
        print(f"\n⚠️  {len(failed_tests)} parameter variation test(s) failed!")
    else:
        print(f"\n🎉 ALL {len(test_results)} parameter variations passed!")

    # Final Summary
    print("\n7. FINAL TEST SUMMARY")
    print("-" * 30)
    print("✓ Model initialization: PASSED")
    print("✓ Training: PASSED") 
    print("✓ Batched prediction: PASSED")
    print("✓ Single trial prediction: PASSED")
    print("✓ Single point prediction: PASSED")
    print("✓ Consistency verification: PASSED")
    if not failed_tests:
        print("✓ Parameter variations: PASSED")
        print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("✓ KalmanFilter is ready for neural decoding applications")
    else:
        print("✗ Parameter variations: SOME FAILED")
        print(f"\n⚠️  {len(failed_tests)} tests failed - check parameter variation results above")
    print("=" * 60)