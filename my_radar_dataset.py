import os
from scipy.io import loadmat
import numpy as np

class MyRadarDataset:
    def __init__(self):
        """
        Initialize the MyRadarDataset class.

        Attributes:
            data (list): List of numpy arrays containing the data samples.
            labels (list): List of numpy arrays containing the labels.
            files (list): List of strings containing the file names.
            _num_samples (int): Total number of samples in the dataset.
        """
        self.data = np.array([])
        self.labels = np.array([])
        self.files = []
        self._num_samples = 0

    def load(self, folder=None):
        """
        Load '.mat' files from the specified folder into the dataset.

        Args:
            folder (str): Path to the folder containing .mat files.

        Raises:
            ValueError: If folder path is not provided.
            FileNotFoundError: If the folder does not exist or no .mat files found.
            KeyError: If the expected keys ('X' and 'Y') are not found in the .mat files.
        """
        if folder is None:
            raise ValueError("Folder path must be provided")

        if not os.path.exists(folder):
            raise FileNotFoundError(f"The folder {folder} does not exist")
        
        mat_files = [file for file in os.listdir(folder) if file.endswith(".mat")]
        if not mat_files:
            raise FileNotFoundError(f"No .mat files found in the folder {folder}")
        
        all_data = []
        all_labels = []
        all_files = []
        
        for file_name in mat_files:
            file_path = os.path.join(folder, file_name)
            mat_data = loadmat(file_path)
            if 'X' not in mat_data or 'Y' not in mat_data:
                raise KeyError(f"File {file_name} does not contain 'X' and 'Y' keys")

            X = mat_data['X']
            Y = mat_data['Y'].flatten()
            if len(X) != len(Y):
                raise ValueError(f"Data and labels length mismatch in file {file_name}")
            
            all_data.append(X.reshape(-1, X.shape[1]))  # (num_samples, 8)
            all_labels.append(Y)  # (num_samples,)
            all_files.extend([file_name] * X.shape[0])
        
        # Combine all data and labels
        self.data = np.concatenate(all_data, axis=0).reshape(-1, 1, X.shape[1])  # (total_samples, 1, 8)
        self.labels = np.concatenate(all_labels, axis=0).reshape(-1, 1)  # (total_samples, 1)
        self.files = all_files
        self._num_samples = self.data.shape[0]

    @property
    def num_samples(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return int(self._num_samples)

    def get(self, sample_index):
        """
        Retrieve a sample by its index.

        Args:
            sample_index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple (x, y, file_name) where:
                - x (numpy array): Features of the sample of shape (1, 8).
                - y (numpy array): Label of the sample of shape (1,).
                - file_name (str): Name of the file the sample came from.

        Raises:
            ValueError: If the dataset is empty.
            IndexError: If the sample index is out of range.
        """
        if self._num_samples == 0:
            raise ValueError("The dataset is empty. Please load data first.")
        
        if sample_index < 0 or sample_index >= self._num_samples:
            raise IndexError("Sample index out of range")

        x = self.data[sample_index].reshape(1, -1)  # Ensure shape is (1, 8)
        y = self.labels[sample_index].reshape(1,)  # Ensure shape is (1,)
        file_name = self.files[sample_index]

        return x, y, file_name

    def split(self, slice_indexes):
        """
        Split the dataset into two datasets based on the provided slice indexes.

        Args:
            slice_indexes (list, tuple, or numpy array): Indices to include in the first dataset.

        Returns:
            tuple: Two MyRadarDataset instances. The first contains the specified indices, the second contains the rest.

        Raises:
            ValueError: If slice_indexes is not a list, tuple, or numpy array.
        """
        if isinstance(slice_indexes, (list, tuple, np.ndarray)):
            slice_indexes = np.array(slice_indexes)
        else:
            raise ValueError("slice_indexes must be a list, tuple, or numpy array")

        # Create a mask for the indices
        mask = np.zeros(self.num_samples, dtype=bool)
        mask[slice_indexes] = True

        # Create new datasets
        dataset1 = MyRadarDataset()
        dataset2 = MyRadarDataset()

         # Using the mask to split data and labels
        dataset1.data = self.data[mask].reshape(-1, 1, self.data.shape[2])
        dataset1.labels = self.labels[mask].reshape(-1, 1)
        dataset1.files = [self.files[i] for i in np.where(mask)[0]]
        dataset1._num_samples = np.sum(mask)

        dataset2.data = self.data[~mask].reshape(-1, 1, self.data.shape[2])
        dataset2.labels = self.labels[~mask].reshape(-1, 1)
        dataset2.files = [self.files[i] for i in np.where(~mask)[0]]
        dataset2._num_samples = np.sum(~mask)
            
        return dataset1, dataset2
    
    def random_split(self, ratio=0.7):
        """
        Randomly split the dataset into two datasets based on the specified ratio.

        Args:
            ratio (float): Ratio of the first dataset (e.g., 0.7 means 70% for the first dataset, 30% for the second).

        Returns:
            tuple: Two MyRadarDataset instances, split according to the ratio.

        Raises:
            ValueError: If the ratio is not between 0 and 1.
        """
        if not (0 < ratio < 1):
            raise ValueError("ratio must be between 0 and 1")

        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        split_index = int(self.num_samples * ratio)
        dataset1_indices = indices[:split_index]

        return self.split(dataset1_indices)
    
    def get_generator(self, batch_size=1, shuffle=True):
        """
        Return a generator that yields batches of samples from the dataset.

        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the samples at the beginning of each epoch.

        Yields:
            tuple: A tuple (x_batch, y_batch) where:
                - x_batch (numpy array): Batch of features of shape (batch_size, 8).
                - y_batch (numpy array): Batch of labels of shape (batch_size,).
        """
        indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.num_samples, batch_size):
            end_idx = min(start_idx + batch_size, self.num_samples)
            batch_indices = indices[start_idx:end_idx]
            x_batch = np.concatenate([self.data[i] for i in batch_indices], axis=0).reshape(-1, 8)
            y_batch = np.concatenate([self.labels[i] for i in batch_indices], axis=0).flatten()
            yield x_batch, y_batch



def main():
    # Create a folder for dummy .mat files
    test_folder = 'mat_files'
    
    # Initialize MyRadarDataset
    radar_dataset = MyRadarDataset()
    
    # Load the dataset
    print("\nLoading dataset...")
    try:
        radar_dataset.load(folder=test_folder)
        print(f"Total samples loaded: {radar_dataset.num_samples}")
    except Exception as e:
        print(f"Error loading data: {e}")
    
    # Get a sample
    print("\nRetrieving a sample...")
    try:
        sample_index = 5
        x, y, file_name = radar_dataset.get(sample_index)
        assert x.shape == (1, 8), f"Incorrect shape for x: {x.shape}"
        assert y.shape == (1,), f"Incorrect shape for y: {y.shape}"
        print(f"Sample {sample_index} retrieved successfully.")
        print(f"  x: {x}")
        print(f"  y: {y}")
        print(f"  file_name: {file_name}")
    except (ValueError, IndexError) as e:
        print(f"Error retrieving sample {sample_index}: {e}")
    
    # Test bad sample index
    print("\nTesting bad sample indices...")
    try:
        bad_sample_index = -1
        radar_dataset.get(bad_sample_index)
    except Exception as e:
        print(f"Correctly handled bad sample index {bad_sample_index}: {e}")
    
    try:
        bad_sample_index = radar_dataset.num_samples + 1
        radar_dataset.get(bad_sample_index)
    except Exception as e:
        print(f"Correctly handled bad sample index {bad_sample_index}: {e}")
    
    # Split the dataset
    print("\nSplitting the dataset...")
    try:
        split_index = np.arange(radar_dataset.num_samples // 2)
        dataset1, dataset2 = radar_dataset.split(split_index)
        print(f"Dataset 1 samples: {dataset1.num_samples}")
        print(f"Dataset 2 samples: {dataset2.num_samples}")
        assert dataset1.data.shape == (dataset1.num_samples, 1, 8), "Incorrect shape for dataset1 data"
        assert dataset1.labels.shape == (dataset1.num_samples, 1), "Incorrect shape for dataset1 labels"
        assert dataset2.data.shape == (dataset2.num_samples, 1, 8), "Incorrect shape for dataset2 data"
        assert dataset2.labels.shape == (dataset2.num_samples, 1), "Incorrect shape for dataset2 labels"
    except Exception as e:
        print(f"Error splitting dataset: {e}")
    
    # Test bad split indexes
    print("\nTesting bad split indexes...")
    try:
        bad_split_index = "not a list"
        radar_dataset.split(bad_split_index)
    except Exception as e:
        print(f"Correctly handled bad split indexes: {e}")
    
    # Random split the dataset
    print("\nRandomly splitting the dataset...")
    try:
        dataset3, dataset4 = radar_dataset.random_split() # default is 0.7
        print(f"Random split - Dataset 3 samples: {dataset3.num_samples}")
        print(f"Random split - Dataset 4 samples: {dataset4.num_samples}")
        assert dataset3.data.shape == (dataset3.num_samples, 1, 8), "Incorrect shape for dataset3 data"
        assert dataset3.labels.shape == (dataset3.num_samples, 1), "Incorrect shape for dataset3 labels"
        assert dataset4.data.shape == (dataset4.num_samples, 1, 8), "Incorrect shape for dataset4 data"
        assert dataset4.labels.shape == (dataset4.num_samples, 1), "Incorrect shape for dataset4 labels"
    except Exception as e:
        print(f"Error in random split: {e}")
    
    # Test bad ratio for random split
    print("\nTesting bad ratios for random split...")
    try:
        bad_ratio = -0.5
        radar_dataset.random_split(ratio=bad_ratio)
    except Exception as e:
        print(f"Correctly handled bad ratio {bad_ratio}: {e}")
    
    try:
        bad_ratio = 1.5
        radar_dataset.random_split(ratio=bad_ratio)
    except Exception as e:
        print(f"Correctly handled bad ratio {bad_ratio}: {e}")
    
    # Generate batches
    print("\nGenerating batches...")
    try:
        batch_size = 3
        generator = radar_dataset.get_generator(batch_size=batch_size, shuffle=True)
        print(f"Generating batches of size {batch_size}:")
        for i, (x_batch, y_batch) in enumerate(generator):
            assert x_batch.shape[1] == 8, f"Incorrect shape for x_batch: {x_batch.shape}"
            assert y_batch.shape[0] == x_batch.shape[0], f"y_batch and x_batch size mismatch: {y_batch.shape}, {x_batch.shape}"
            print(f"  Batch {i} retrieved successfully.")
            print(f"    x_batch: {x_batch}")
            print(f"    y_batch: {y_batch}")
            if i >= 1:  # Limit the number of batches printed
                break
    except Exception as e:
        print(f"Error generating batches: {e}")

if __name__ == "__main__":
    main()
