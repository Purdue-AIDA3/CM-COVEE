import numpy as np
import os
import torch
import scipy.io
from torch.utils.data import Dataset, DataLoader


class MultiModalDataset(Dataset):
    """Dataset for multi-modal cognitive load data"""
    
    def __init__(self, data_dict, labels, modality):
        """
        Args:
            data_dict: Dictionary containing data for each modality
                      Keys: 'psd', 'pupil_diam', 'gaze_speed', 'eye_landmarks', 'face_landmarks'
            labels: Array of labels
            modality: String specifying which modalities to use
                     Options: 'eeg', 'eye', 'face', 'eeg_and_eye', 'eeg_and_face', 
                             'eye_and_face', 'eeg_and_eye_and_face'
        """
        self.data_dict = data_dict
        self.labels = torch.FloatTensor(labels)
        self.modality = modality
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx].float()
        
        # Return data based on modality
        if self.modality == 'eeg':
            return torch.FloatTensor(self.data_dict['psd'][idx]), label
        
        elif self.modality == 'eye':
            # Concatenate eye tracking features
            pupil = self.data_dict['pupil_diam'][idx]
            gaze = self.data_dict['gaze_speed'][idx]
            eye_data = np.concatenate([pupil, gaze], axis=-1)
            return torch.FloatTensor(eye_data), label
        
        elif self.modality == 'face':
            face_landmarks = self.data_dict['face_landmarks'][idx]
            eye_landmarks = self.data_dict['eye_landmarks'][idx]
            face_data = np.concatenate([face_landmarks, eye_landmarks], axis=-1)
            return torch.FloatTensor(face_data), label
        
        elif self.modality == 'eeg_and_eye':
            eeg_data = torch.FloatTensor(self.data_dict['psd'][idx])
            pupil = self.data_dict['pupil_diam'][idx]
            gaze = self.data_dict['gaze_speed'][idx]
            eye_data = torch.FloatTensor(np.concatenate([pupil, gaze], axis=-1))
            return eeg_data, eye_data, label
        
        elif self.modality == 'eeg_and_face':
            eeg_data = torch.FloatTensor(self.data_dict['psd'][idx])
            face_landmarks = self.data_dict['face_landmarks'][idx]
            eye_landmarks = self.data_dict['eye_landmarks'][idx]
            face_data = torch.FloatTensor(np.concatenate([face_landmarks, eye_landmarks], axis=-1))
            return eeg_data, face_data, label
        
        elif self.modality == 'eye_and_face':
            pupil = self.data_dict['pupil_diam'][idx]
            gaze = self.data_dict['gaze_speed'][idx]
            eye_data = torch.FloatTensor(np.concatenate([pupil, gaze], axis=-1))
            face_landmarks = self.data_dict['face_landmarks'][idx]
            eye_landmarks = self.data_dict['eye_landmarks'][idx]
            face_data = torch.FloatTensor(np.concatenate([face_landmarks, eye_landmarks], axis=-1))
            return eye_data, face_data, label
        
        elif self.modality == 'eeg_and_eye_and_face':
            eeg_data = torch.FloatTensor(self.data_dict['psd'][idx])
            pupil = self.data_dict['pupil_diam'][idx]
            gaze = self.data_dict['gaze_speed'][idx]
            eye_data = torch.FloatTensor(np.concatenate([pupil, gaze], axis=-1))
            face_landmarks = self.data_dict['face_landmarks'][idx]
            eye_landmarks = self.data_dict['eye_landmarks'][idx]
            face_data = torch.FloatTensor(np.concatenate([face_landmarks, eye_landmarks], axis=-1))
            return eeg_data, eye_data, face_data, label


class DataProcesser:
    """Main data processor for loading and preparing multi-modal data"""
    
    def __init__(self, directory, user_directory, label_directory):
        """
        Args:
            directory: Base directory containing user folders (e.g., 'COVEE/simple_datasets/')
            user_directory: Directory containing user label files (e.g., 'COVEE/user_data/')
            label_directory: Alternative directory for labels (e.g., 'COVEE/window_10_shift_10/')
        """
        self.directory = directory
        self.user_directory = user_directory
        self.label_directory = label_directory
        
        # Task file mappings
        self.task_mappings = {
            'one_platform': ('Task_2_one_platform', 'Trial 1'),
            'one_platform_secondary': ('Task_2_one_platform_secondary', 'Trial 2'),
            'two_platforms': ('Task_2_two_platforms', 'Trial 3'),
            'two_platforms_secondary': ('Task_2_two_platforms_secondary', 'Trial 4')
        }
        
        # Get list of all user label files
        self.user_files = [f for f in os.listdir(user_directory) 
                          if '_Task_2_' in f and f.endswith('.npy')]
        
        # Extract unique user IDs
        self.user_ids = []
        for f in self.user_files:
            user_id = f.split('_Task_2_')[0]
            if user_id not in self.user_ids:
                self.user_ids.append(user_id)
        
        print(f"Found {len(self.user_ids)} users: {self.user_ids}")
        print(f"Total label files: {len(self.user_files)}")
    
    def load_eeg_data(self, user_id, task_file):
        """
        Load EEG data from .mat file
        
        Args:
            user_id: User identifier
            task_file: Task filename (e.g., 'Task_2_one_platform')
            
        Returns:
            numpy array of EEG features with shape (time_windows, channels * frequencies)
        """
        # Correct path structure
        mat_path = os.path.join(self.directory, user_id, "EEG/min_processed/log_ps/Task 2/", f"{task_file}.mat")
        
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"EEG file not found: {mat_path}")
        
        # Load .mat file
        mat_data = scipy.io.loadmat(mat_path)
        
        # Extract 'all_bands' with correct slicing
        # Shape: (time_length, 30, 40) where 30 channels, 40 frequencies
        # We want first 31 channels (0:31) and all 40 frequencies (0:40)
        eeg_data = mat_data['all_psd'][:, 0:31, 0:40]
        
        # Current shape: (time_windows, 31, 40)
        num_windows, num_channels, num_freqs = eeg_data.shape
        
        # Keep 3D shape (time_windows, 31, 40) for normalization
        # Will be flattened after normalization in load_all_data
        return eeg_data
    
    def load_eye_data(self, user_id, trial_name):
        """
        Load eye tracking data from .txt files
        
        Args:
            user_id: User identifier
            trial_name: Trial name (e.g., 'Trial 1')
            
        Returns:
            Tuple of (gaze_speed, pupil_diam) arrays
        """
        eye_dir = os.path.join(self.directory, user_id, 'Eye Tracker', 'Task 2', trial_name)
        
        gaze_path = os.path.join(eye_dir, 'gaze_speed.txt')
        pupil_path = os.path.join(eye_dir, 'pupil_diam.txt')
        
        if not os.path.exists(gaze_path):
            raise FileNotFoundError(f"Gaze speed file not found: {gaze_path}")
        if not os.path.exists(pupil_path):
            raise FileNotFoundError(f"Pupil diameter file not found: {pupil_path}")
        
        # Load data from txt files
        gaze_speed = np.loadtxt(gaze_path)
        pupil_diam = np.loadtxt(pupil_path)
        
        return gaze_speed, pupil_diam
    
    def load_face_data(self, user_id, trial_num):
        """
        Load face landmarks data from .txt file
        
        Args:
            user_id: User identifier
            trial_num: Trial number (1, 2, 3, or 4)
            
        Returns:
            numpy array of face landmarks
        """
        face_path = os.path.join(self.directory, user_id, 'Video', 'Task 2', 
                                'Face Landmarks', f'Trial_{trial_num}.txt')
        
        if not os.path.exists(face_path):
            raise FileNotFoundError(f"Face landmarks file not found: {face_path}")
        
        # Load face landmarks - each line contains landmark coordinates
        # Format can be comma-separated or space-separated
        face_data = []
        with open(face_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Try comma-separated first
                    if ',' in line:
                        landmarks = [float(x) for x in line.split(',')]
                    # Try space-separated
                    elif ' ' in line:
                        landmarks = [float(x) for x in line.split()]
                    else:
                        # Single value
                        landmarks = [float(line)]
                    
                    face_data.append(landmarks)
                except ValueError as e:
                    print(f"      Warning: Could not parse line {line_num} in {face_path}")
                    print(f"      Error: {e}")
                    print(f"      Line content (first 100 chars): {line[:100]}")
                    # Skip this line and continue
                    continue
        
        if not face_data:
            raise ValueError(f"No valid face landmark data found in {face_path}")
        
        face_data = np.array(face_data)
        
        return face_data

    def load_eye_landmark_data(self, user_id, trial_num):
        """
        Load face landmarks data from .txt file
        
        Args:
            user_id: User identifier
            trial_num: Trial number (1, 2, 3, or 4)
            
        Returns:
            numpy array of face landmarks
        """
        eye_path = os.path.join(self.directory, user_id, 'Video', 'Task 2', 
                                'Eye Landmarks', f'Trial_{trial_num}.txt')
        
        if not os.path.exists(eye_path):
            raise FileNotFoundError(f"Face landmarks file not found: {eye_path}")
        
        # Load face landmarks - each line contains landmark coordinates
        # Format can be comma-separated or space-separated
        eye_data = []
        with open(eye_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Try comma-separated first
                    if ',' in line:
                        landmarks = [float(x) for x in line.split(',')]
                    # Try space-separated
                    elif ' ' in line:
                        landmarks = [float(x) for x in line.split()]
                    else:
                        # Single value
                        landmarks = [float(line)]
                    
                    eye_data.append(landmarks)
                except ValueError as e:
                    print(f"      Warning: Could not parse line {line_num} in {eye_path}")
                    print(f"      Error: {e}")
                    print(f"      Line content (first 100 chars): {line[:100]}")
                    # Skip this line and continue
                    continue
        
        if not eye_data:
            raise ValueError(f"No valid face landmark data found in {eye_path}")
        
        eye_data = np.array(eye_data)
        
        return eye_data
    
    def load_labels(self, user_id, task_type='one_platform', secondary=False):
        """
        Load labels for a specific user from user_directory
        
        Args:
            user_id: User identifier string
            task_type: 'one_platform' or 'two_platforms'
            secondary: If True, load secondary labels
            
        Returns:
            Array of scores/labels
        """
        # Construct filename based on task type and secondary flag
        if secondary:
            label_file = os.path.join(self.user_directory, 
                                     f"{user_id}_Task_2_{task_type}_secondary.npy")
        else:
            label_file = os.path.join(self.user_directory, 
                                     f"{user_id}_Task_2_{task_type}.npy")
        
        if not os.path.exists(label_file):
            # Try alternative location in label_directory
            if secondary:
                label_file = os.path.join(self.label_directory, 
                                         f"{user_id}_Task_2_{task_type}_secondary.npy")
            else:
                label_file = os.path.join(self.label_directory, 
                                         f"{user_id}_Task_2_{task_type}.npy")
        
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found for user {user_id}, task {task_type}, secondary={secondary}")
        
        labels = np.load(label_file, allow_pickle=True)
        
        return labels

    def normalize_landmarks(self, landmarks_data, modality_name='landmarks'):
        """
        Normalize landmark data using min-max normalization
        
        Args:
            landmarks_data: List of arrays or stacked array with shape (N, 10, features)
            modality_name: Name of the modality for logging purposes
            
        Returns:
            Normalized data with shape (N, 10, features)
        """
        # Stack if it's a list
        if isinstance(landmarks_data, list):
            landmarks_data = np.concatenate(landmarks_data, axis=0)
        
        # Current shape: (N, 10, features)
        N, time_steps, num_features = landmarks_data.shape
        
        # Reshape to (N*10, features) for normalization
        data_reshaped = landmarks_data.reshape(N * time_steps, num_features)
        
        # Calculate global min and max across all samples and time steps for each feature
        min_vals = np.min(data_reshaped, axis=0)  # (features,)
        max_vals = np.max(data_reshaped, axis=0)  # (features,)
        
        # Normalize each feature
        data_normalized = np.zeros_like(data_reshaped)
        for i in range(num_features):
            if (max_vals[i] - min_vals[i]) == 0:
                # If max == min, set to 0 (or could set to 0.5)
                data_normalized[:, i] = 0
            else:
                data_normalized[:, i] = (data_reshaped[:, i] - min_vals[i]) / (max_vals[i] - min_vals[i] + 1e-8)
        
        # Reshape back to (N, 10, features)
        data_normalized = data_normalized.reshape(N, time_steps, num_features)
        
        print(f"  {modality_name} normalized: min={data_normalized.min():.4f}, max={data_normalized.max():.4f}")
    
        return data_normalized

    def normalize_labels(self, labels):
        """
        Normalize continuous labels/scores using min-max normalization
        
        Args:
            labels: Array of continuous label values
            
        Returns:
            Normalized labels with values in [0, 1] range
        """
        # Convert to numpy array if not already
        labels = np.array(labels)
        
        # Get min and max values
        min_val = np.min(labels)
        max_val = np.max(labels)
        
        # Check if all values are the same
        if (max_val - min_val) == 0:
            print(f"Warning: All label values are the same ({min_val}), setting normalized values to 0.5")
            normalized_labels = np.ones_like(labels) * 0.5
        else:
            # Apply min-max normalization
            normalized_labels = (labels - min_val) / (max_val - min_val)
        
        print(f"\nLabel normalization:")
        print(f"  Original range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"  Normalized range: [{normalized_labels.min():.4f}, {normalized_labels.max():.4f}]")
        print(f"  Original mean: {labels.mean():.4f}, Normalized mean: {normalized_labels.mean():.4f}")
        
        return normalized_labels
    
    def normalize_eeg_data(self, data):
        """
        Normalize EEG data using min-max normalization
        
        Args:
            data: List of arrays, each with shape (n_windows, 10, 31, 40)
                  OR stacked array with shape (total_windows, 10, 31, 40)
            
        Returns:
            Normalized data with shape (total_windows, 10, 1240)
        """
        # Stack if it's a list
        if isinstance(data, list):
            data = np.concatenate(data, axis=0)  # (N, 10, 31, 40)
        
        # Current shape: (N, 10, 31, 40)
        N = data.shape[0]
        
        # Reshape for normalization: (N, 10, 31, 40) â†’ (N*10, 31, 40)
        # This treats each 1-second window separately
        data_reshaped = data.reshape(N * 10, 31, 40)
        
        # Now apply your normalization code
        # Transpose: (N*10, 31, 40) â†’ (N*10, 40, 31)
        data_norm = np.transpose(data_reshaped, (0, 2, 1))
        original_shape = data_norm.shape  # (N*10, 40, 31)
        
        # Reshape: (N*10, 40, 31) â†’ (N*10, 40, 31) [already this shape]
        # Calculate stats per frequency bin
        min_val = np.min(data_norm.reshape(-1, 40, 31), axis=(0, 2))  # (40,)
        max_val = np.max(data_norm.reshape(-1, 40, 31), axis=(0, 2))  # (40,)
        
        # Normalize
        for i in range(data_norm.shape[0]):
            for j in range(40):  # For each frequency bin
                data_norm[i, j] = (data_norm[i, j] - min_val[j]) / (max_val[j] - min_val[j] + 1e-8)
        
        # Transpose back: (N*10, 40, 31) â†’ (N*10, 31, 40)
        data_norm = np.transpose(data_norm, (0, 2, 1))
        
        # Flatten last two dimensions: (N*10, 31, 40) â†’ (N*10, 1240)
        data_norm = data_norm.reshape(N * 10, 31 * 40)
        
        # Reshape back to include 10-second structure: (N*10, 1240) â†’ (N, 10, 1240)
        data_final = data_norm.reshape(N, 10, 1240)
        
        return data_final
    
    def normalize_eye_data(self, gaze_speed_data, pupil_diam_data):
        """
        Normalize eye tracking data (gaze speed and pupil diameter) using min-max normalization
        
        Args:
            gaze_speed_data: List of arrays or stacked array with shape (N, 10, 1)
            pupil_diam_data: List of arrays or stacked array with shape (N, 10, 1)
            
        Returns:
            Tuple of (normalized_gaze_speed, normalized_pupil_diam) with shape (N, 10, 1) each
        """
        # Stack if it's a list
        if isinstance(gaze_speed_data, list):
            gaze_speed_data = np.concatenate(gaze_speed_data, axis=0)
        if isinstance(pupil_diam_data, list):
            pupil_diam_data = np.concatenate(pupil_diam_data, axis=0)
        
        # Normalize gaze speed
        gaze_normalized = self._normalize_single_eye_modality(gaze_speed_data, modality='gaze_speed')
        
        # Normalize pupil diameter
        pupil_normalized = self._normalize_single_eye_modality(pupil_diam_data, modality='pupil_diam')
        
        return gaze_normalized, pupil_normalized
    
    def _normalize_single_eye_modality(self, data, modality='gaze_speed'):
        """
        Normalize a single eye modality
        
        Args:
            data: Array with shape (N, 10, 1)
            modality: 'gaze_speed' or 'pupil_diam'
            
        Returns:
            Normalized data with shape (N, 10, 1)
        """
        # Flatten to 1D for processing: (N, 10, 1) -> (N*10,)
        N = data.shape[0]
        data_flat = data.reshape(-1)  # (N*10*30,)
        
        # Apply clipping for gaze_speed
        if modality == 'gaze_speed':
            data_flat[data_flat > np.sqrt(2)] = np.sqrt(2)
            max_val = np.sqrt(2)
        else:
            max_val = np.max(data_flat)
        
        min_val = np.min(data_flat)
        
        # Normalize each sample
        data_normalized = np.zeros_like(data_flat)
        for i in range(len(data_flat)):
            if (max_val - min_val) == 0:
                print(f"Warning: max_val == min_val for {modality}, setting normalized value to 0")
                data_normalized[i] = 0
            else:
                data_normalized[i] = (data_flat[i] - min_val) / (max_val - min_val)
        
        # Reshape back to (N, 10, 1)
        data_normalized = data_normalized.reshape(N, 10, 30)
        
        return data_normalized
    
    def resample_to_windows(self, data, target_length, method='average'):
        """
        Resample data to match target length (number of windows)
        
        Args:
            data: Input data array (samples, features) or (samples,)
            target_length: Target number of windows
            method: 'average' for averaging, 'interpolate' for interpolation, 
                   'repeat' for label expansion
            
        Returns:
            Resampled data with shape (target_length, features) or (target_length,)
        """
        current_length = len(data)
        
        if current_length == target_length:
            return data
        
        # If expanding labels (target > current), use repeat method
        if method == 'repeat' or (target_length > current_length * 2 and data.ndim == 1):
            # This is likely label expansion - repeat each label for multiple windows
            repeats_per_sample = target_length / current_length
            expanded = []
            for i in range(current_length):
                # Calculate how many times to repeat this label
                start_idx = int(i * repeats_per_sample)
                end_idx = int((i + 1) * repeats_per_sample)
                n_repeats = end_idx - start_idx
                if data.ndim == 1:
                    expanded.extend([data[i]] * n_repeats)
                else:
                    expanded.extend([data[i]] * n_repeats)
            
            # Handle any rounding issues
            result = np.array(expanded)
            if len(result) < target_length:
                # Pad with last value
                if data.ndim == 1:
                    result = np.pad(result, (0, target_length - len(result)), mode='edge')
                else:
                    padding = np.repeat([result[-1]], target_length - len(result), axis=0)
                    result = np.vstack([result, padding])
            elif len(result) > target_length:
                result = result[:target_length]
            
            return result
        
        if method == 'average':
            # Average pooling (for downsampling features)
            window_size = current_length / target_length
            resampled = []
            
            for i in range(target_length):
                start_idx = int(i * window_size)
                end_idx = int((i + 1) * window_size)
                if end_idx > current_length:
                    end_idx = current_length
                if start_idx >= current_length:
                    start_idx = current_length - 1
                    
                if data.ndim == 1:
                    if end_idx > start_idx:
                        resampled.append(np.mean(data[start_idx:end_idx]))
                    else:
                        resampled.append(data[start_idx])
                else:
                    if end_idx > start_idx:
                        resampled.append(np.mean(data[start_idx:end_idx], axis=0))
                    else:
                        resampled.append(data[start_idx])
            
            return np.array(resampled)
        
        elif method == 'interpolate':
            # Linear interpolation
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, current_length)
            x_new = np.linspace(0, 1, target_length)
            
            if data.ndim == 1:
                f = interp1d(x_old, data, kind='linear')
                return f(x_new)
            else:
                resampled = []
                for feature_idx in range(data.shape[1]):
                    f = interp1d(x_old, data[:, feature_idx], kind='linear')
                    resampled.append(f(x_new))
                return np.array(resampled).T
    
    def load_user_data(self, user_id, task_key):
        """
        Load all modality data for a specific user and task
        
        Data collection rates:
        - EEG: 1Hz (1 sample per second)
        - Eye: 30Hz (30 samples per second) 
        - Face: 30Hz (30 samples per second)
        - Labels: 10-second windows
        
        Reshaping:
        - EEG (T, 31, 40) â†’ (T//10, 10, 1240) where T//10 = number of 10-sec windows
        - Eye gaze (T*30,) + pupil (T*30,) â†’ (T*30//300, 10, 2) = (T//10, 10, 2)
        - Face (T*30, 136) â†’ (T*30//300, 10, 136) = (T//10, 10, 136)
        
        Args:
            user_id: User identifier
            task_key: One of 'one_platform', 'one_platform_secondary', 
                     'two_platforms', 'two_platforms_secondary'
            
        Returns:
            Dictionary with aligned data for all modalities
        """
        task_file, trial_name = self.task_mappings[task_key]
        trial_num = trial_name.split()[-1]  # Extract number from "Trial 1"
        
        print(f"    Loading {task_key}: {task_file}, {trial_name}")
        
        # Load EEG data (1Hz)
        try:
            eeg_data = self.load_eeg_data(user_id, task_file)
            print(f"      EEG raw: {eeg_data.shape}")
            
            # Reshape EEG: (T, 31, 40) â†’ (T//10, 10, 31, 40)
            eeg_samples = len(eeg_data)
            n_windows_eeg = eeg_samples // 10
            eeg_trimmed = eeg_data[:n_windows_eeg * 10]  # Trim to multiple of 10
            eeg_reshaped = eeg_trimmed.reshape(n_windows_eeg, 10, 31, 40)
            
            print(f"      EEG reshaped: {eeg_reshaped.shape} ({n_windows_eeg} windows of 10 seconds)")
            
        except Exception as e:
            print(f"      EEG loading error: {e}")
            return None
        
        # Load eye tracking data (30Hz)
        try:
            gaze_speed, pupil_diam = self.load_eye_data(user_id, trial_name)
            print(f"      Eye raw: gaze_speed {gaze_speed.shape}, pupil_diam {pupil_diam.shape}")
            
            # Calculate number of 10-second windows from eye data
            # At 30Hz, 10 seconds = 300 samples
            eye_samples = min(len(gaze_speed), len(pupil_diam))
            n_windows_eye = eye_samples // 300
            
            # Trim and reshape
            gaze_trimmed = gaze_speed[:n_windows_eye * 300]
            pupil_trimmed = pupil_diam[:n_windows_eye * 300]
            
            gaze_reshaped = gaze_trimmed.reshape(n_windows_eye, 10, 30)  # 10 seconds, 30Hz
            pupil_reshaped = pupil_trimmed.reshape(n_windows_eye, 10, 30)
            
            # Average within each second to get (n_windows, 10, 1) for each
            # gaze_per_sec = np.mean(gaze_reshaped, axis=2, keepdims=True)  # (n_windows, 10, 1)
            # pupil_per_sec = np.mean(pupil_reshaped, axis=2, keepdims=True)  # (n_windows, 10, 1)
            
            # Concatenate to get (n_windows, 10, 2)
            eye_data = np.concatenate([gaze_reshaped, pupil_reshaped], axis=2)
            
            print(f"      Eye reshaped: {eye_data.shape} ({n_windows_eye} windows of 10 seconds)")
                
        except Exception as e:
            print(f"      Eye loading error: {e}")
            eye_data = np.zeros((n_windows_eeg, 10, 30))
            n_windows_eye = n_windows_eeg
        
        # Load face landmarks (30Hz)
        try:
            face_raw = self.load_face_data(user_id, trial_num)
            print(f"      Face raw: {face_raw.shape}")
            eye_raw = self.load_eye_landmark_data(user_id, trial_num)
            print(f"      Eye raw: {eye_raw.shape}")
            
            # Calculate number of 10-second windows from face data
            face_samples = len(face_raw)
            n_windows_face = face_samples // 300
            
            # Trim and reshape: (T*30, 136) â†’ (T, 300, 136) â†’ (T, 10, 30, 136)
            face_trimmed = face_raw[:n_windows_face * 300]
            face_reshaped = face_trimmed.reshape(n_windows_face, 10, 30*136)

            eye_trimmed = eye_raw[:n_windows_face * 300]
            eye_reshaped = eye_trimmed.reshape(n_windows_face, 10, 30*112)
            
            # Average within each second to get (n_windows, 10, 136)
            # face_data = np.mean(face_reshaped, axis=2)  # Average over 30 samples per second
            face_data = np.concatenate([face_reshaped, eye_reshaped], axis=2)
            
            print(f"      Face reshaped: {face_data.shape} ({n_windows_face} windows of 10 seconds)")
            
        except Exception as e:
            print(f"      Face loading error: {e}")
            face_data = np.zeros((n_windows_eeg, 10, 7440))
            n_windows_face = n_windows_eeg
        
        # Find minimum number of windows across all modalities
        min_windows = min(n_windows_eeg, n_windows_eye, n_windows_face)
        
        print(f"      Minimum windows across modalities: {min_windows}")
        
        # Trim all to minimum length
        eeg_data_final = eeg_reshaped[:min_windows]  # (min_windows, 10, 31, 40)
        eye_data_final = eye_data[:min_windows]      # (min_windows, 10, 60)
        face_data_final = face_data[:min_windows]    # (min_windows, 10, 7440)
        
        # Reshape EEG to flatten frequency and channel: (min_windows, 10, 31*40)
        eeg_data_final = eeg_data_final.reshape(min_windows, 10, 31 * 40)
        
        print(f"      Final shapes - EEG: {eeg_data_final.shape}, Eye: {eye_data_final.shape}, Face: {face_data_final.shape}")
        
        # Note: eye_landmarks are not available in the raw data, so we'll create placeholder
        eye_landmarks = np.zeros((min_windows, 10, 116))  # Placeholder
        
        return {
            'psd': eeg_data_final,          # (min_windows, 10, 1240)
            'pupil_diam': eye_data_final[:, :, 30:60],  # (min_windows, 10, 30)
            'gaze_speed': eye_data_final[:, :, 0:30],  # (min_windows, 10, 30)
            'eye_landmarks': eye_reshaped[:min_windows],  # (min_windows, 10, 116)
            'face_landmarks': face_reshaped[:min_windows],  # (min_windows, 10, 136)
            'n_windows': min_windows
        }
    
    def load_all_labels_for_user(self, user_id):
        """
        Load all available label files for a user
        
        Args:
            user_id: User identifier string
            
        Returns:
            Dictionary mapping task_key to labels array
        """
        all_labels = {}
        
        for task_key in ['one_platform', 'one_platform_secondary', 
                        'two_platforms', 'two_platforms_secondary']:
            task_type = 'one_platform' if 'one_platform' in task_key else 'two_platforms'
            secondary = 'secondary' in task_key
            
            try:
                labels = self.load_labels(user_id, task_type, secondary)
                all_labels[task_key] = labels
                print(f"  Loaded {task_key}: {len(labels)} samples")
            except FileNotFoundError:
                print(f"  {task_key}: not found")
                pass
        
        return all_labels
    
    def load_all_data(self):
        """
        Load data and labels for all users
        
        Returns:
            Tuple of (all_data_dict, all_labels)
        """
        all_psd = []
        all_pupil_diam = []
        all_gaze_speed = []
        all_eye_landmarks = []
        all_face_landmarks = []
        all_scores = []
        
        for user_id in self.user_ids:
            try:
                print(f"\n{'='*60}")
                print(f"Loading data for user {user_id}...")
                print(f"{'='*60}")
                
                # Load all labels for this user
                user_labels_dict = self.load_all_labels_for_user(user_id)
                
                # Load data for each task
                for task_key, labels in user_labels_dict.items():
                    user_data = self.load_user_data(user_id, task_key)
                    
                    if user_data is None:
                        continue
                    
                    # Get the number of 10-second windows
                    n_windows = user_data['n_windows']
                    n_labels = len(labels)
                    
                    # Trim labels to match data length
                    if n_labels > n_windows:
                        print(f"      INFO: Trimming labels from {n_labels} to {n_windows} (shortest modality)")
                        labels = labels[:n_windows]
                    elif n_labels < n_windows:
                        print(f"      INFO: Data has more windows ({n_windows}) than labels ({n_labels})")
                        print(f"      Trimming data to {n_labels} windows")
                        # Trim all data to match labels
                        user_data['psd'] = user_data['psd'][:n_labels]
                        user_data['pupil_diam'] = user_data['pupil_diam'][:n_labels]
                        user_data['gaze_speed'] = user_data['gaze_speed'][:n_labels]
                        user_data['eye_landmarks'] = user_data['eye_landmarks'][:n_labels]
                        user_data['face_landmarks'] = user_data['face_landmarks'][:n_labels]
                        n_windows = n_labels
                    
                    print(f"      Final: {n_windows} windows (10-sec each), matching {len(labels)} labels")
                    
                    # Append to lists
                    all_psd.append(user_data['psd'])
                    all_pupil_diam.append(user_data['pupil_diam'])
                    all_gaze_speed.append(user_data['gaze_speed'])
                    all_eye_landmarks.append(user_data['eye_landmarks'])
                    all_face_landmarks.append(user_data['face_landmarks'])
                    all_scores.append(labels)
                    
                    print(f"      Added {len(labels)} samples from {task_key}")
                
            except Exception as e:
                print(f"Error loading data for user {user_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Concatenate all data
        print(f"\n{'='*60}")
        print("Concatenating all data...")
        print(f"{'='*60}")
        
        # Check if any data was loaded
        if not all_scores:
            raise ValueError("No data was successfully loaded. Please check that:\n"
                           f"1. EEG files exist in {self.directory}/{{user_id}}/Task_2_*.mat\n"
                           f"2. Label files exist in {self.user_directory}\n"
                           f"3. Directory paths are correct")
        
        # Normalize EEG data before concatenating
        # all_psd is a list of arrays with shape (time_length, 31, 40)
        print("\nNormalizing EEG data...")
        if all_psd:
            normalized_psd = self.normalize_eeg_data(all_psd)
            print(f"  EEG normalized shape: {normalized_psd.shape}")
        else:
            normalized_psd = None
        
        # Normalize eye tracking data
        print("\nNormalizing eye tracking data...")
        if all_gaze_speed and all_pupil_diam:
            normalized_gaze, normalized_pupil = self.normalize_eye_data(all_gaze_speed, all_pupil_diam)
            print(f"  Gaze speed normalized shape: {normalized_gaze.shape}")
            print(f"  Pupil diameter normalized shape: {normalized_pupil.shape}")
        else:
            normalized_gaze = np.concatenate(all_gaze_speed, axis=0) if all_gaze_speed else None
            normalized_pupil = np.concatenate(all_pupil_diam, axis=0) if all_pupil_diam else None

        # Normalize eye landmarks data
        print("\nNormalizing eye landmarks data...")
        if all_eye_landmarks:
            normalized_eye = self.normalize_landmarks(all_eye_landmarks)
        else:
            RuntimeError('Empty eye landmakrs data')
        # Normalize face landmarks data
        print("\nNormalizing face landmarks data...")
        if all_face_landmarks:
            normalized_face = self.normalize_landmarks(all_face_landmarks)
        else:
            RuntimeError('Empty face landmakrs data')
        
        
        data_dict = {
            'psd': normalized_psd,
            'pupil_diam': normalized_pupil,
            'gaze_speed': normalized_gaze,
            'eye_landmarks': normalized_eye,
            'face_landmarks': normalized_face
        }
        
        all_scores = np.concatenate(all_scores, axis=0)
        all_scores = self.normalize_labels(all_scores)
        print(all_scores)
        
        print(f"\nFinal data shapes:")
        for key, val in data_dict.items():
            if val is not None:
                print(f"  {key}: {val.shape}")
        print(f"  scores: {all_scores.shape}")
        
        return data_dict, all_scores
    
    # def label_maker(self, scores):
    #     """
    #     Convert continuous scores to categorical labels
        
    #     Args:
    #         scores: Array of continuous scores
            
    #     Returns:
    #         Array of categorical labels
    #     """
    #     # Binary classification based on median
    #     median_score = np.median(scores)
    #     labels = (scores >= median_score).astype(int)
        
    #     # Alternative: Tertile classification
    #     # labels = np.digitize(scores, bins=np.percentile(scores, [33, 67]))
        
    #     return labels
    
    def save_all_data(self, all_data, all_scores, cache_path):
        """
        Save loaded data to cache file
        
        Args:
            all_data: Dictionary with all modality data
            all_scores: Array of all scores
            cache_path: Path to save the cache file
        """
        import pickle
        
        # Create directory if it doesn't exist
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        cache_data = {
            'all_data': all_data,
            'all_scores': all_scores
        }
        
        print(f"\nSaving data to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Data cached successfully!")
    
    def load_cached_data(self, cache_path):
        """
        Load data from cache file if it exists
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            Tuple of (all_data, all_scores) if cache exists, None otherwise
        """
        import pickle
        
        if not os.path.exists(cache_path):
            print(f"\nCache file not found: {cache_path}")
            return None
        
        print(f"\nLoading data from cache: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            all_data = cache_data['all_data']
            all_scores = cache_data['all_scores']
            
            print(f"Data loaded from cache successfully!")
            print(f"Total samples: {len(all_scores)}")
            print(f"\nCached data shapes:")
            for key, val in all_data.items():
                if val is not None:
                    print(f"  {key}: {val.shape}")
            print(f"  scores: {all_scores.shape}")
            
            return all_data, all_scores
            
        except Exception as e:
            print(f"Error loading cache: {e}")
            print("Will reload data from scratch...")
            return None
    
    def load_or_cache_data(self, cache_path):
        """
        Try to load from cache first, otherwise load from scratch and save to cache
        
        Args:
            cache_path: Path to the cache file
            
        Returns:
            Tuple of (all_data, all_scores)
        """
        # Try to load from cache
        cached_result = self.load_cached_data(cache_path)
        if cached_result is not None:
            return cached_result
        
        # Cache not found or failed, load from scratch
        print("\nLoading data from scratch...")
        all_data, all_scores = self.load_all_data()
        
        # Save to cache for next time
        self.save_all_data(all_data, all_scores, cache_path)
        
        return all_data, all_scores
    
    def prepare_data(self, fold_num, modality, batch_size,
                    train_levels, val_levels, test_levels,
                    train_psd, val_psd, test_psd,
                    train_pupil, val_pupil, test_pupil,
                    train_gaze, val_gaze, test_gaze,
                    train_eye_landmarks, val_eye_landmarks, test_eye_landmarks,
                    train_face_landmarks, val_face_landmarks, test_face_landmarks):
        """
        Prepare DataLoaders for training, validation, and testing
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Prepare data dictionaries
        train_data_dict = {
            'psd': train_psd,
            'pupil_diam': train_pupil,
            'gaze_speed': train_gaze,
            'eye_landmarks': train_eye_landmarks,
            'face_landmarks': train_face_landmarks
        }
        
        val_data_dict = {
            'psd': val_psd,
            'pupil_diam': val_pupil,
            'gaze_speed': val_gaze,
            'eye_landmarks': val_eye_landmarks,
            'face_landmarks': val_face_landmarks
        }
        
        test_data_dict = {
            'psd': test_psd,
            'pupil_diam': test_pupil,
            'gaze_speed': test_gaze,
            'eye_landmarks': test_eye_landmarks,
            'face_landmarks': test_face_landmarks
        }
        
        # Create datasets
        train_dataset = MultiModalDataset(train_data_dict, train_levels, modality)
        val_dataset = MultiModalDataset(val_data_dict, val_levels, modality)
        test_dataset = MultiModalDataset(test_data_dict, test_levels, modality)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, drop_last=False)
        
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the data loader
    directory = 'COVEE/simple_datasets/'
    user_directory = 'COVEE/user_data/'
    label_directory = 'COVEE/window_10_shift_10/'
    
    processor = DataProcesser(directory, user_directory, label_directory)
    
    # Test loading all data
    print("Loading all data...")
    all_data, all_scores = processor.load_all_data()
    
    print(f"\nLoaded data shapes:")
    for key, val in all_data.items():
        if val is not None:
            print(f"  {key}: {val.shape}")
    print(f"  scores: {all_scores.shape}")