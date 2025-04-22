from torch.utils.data import Dataset
import numpy as np
import torch


class RadarDataset(Dataset):
    def __init__(self, interference_data, clean_data):
        """
        interference_data: Complex radar data with interference (num_samples, num_chirps, num_frames)
        clean_data: Clean radar data without interference (num_samples, num_chirps, num_frames)
        """
        # Convert to numpy arrays if they aren't already
        self.interference_data = np.array(interference_data, dtype=np.complex128)
        self.clean_data = np.array(clean_data, dtype=np.complex128)

        # Convert complex data to real-valued representation [real; imag]
        self.interference_data = self.complex_to_real(self.interference_data)
        self.clean_data = self.complex_to_real(self.clean_data)

        # Store number of frames
        self.num_frames = self.interference_data.shape[3]  # Should be the last dimension after complex_to_real

        print(f"Dataset shapes after processing:")
        print(f"Interference data shape: {self.interference_data.shape}")
        print(f"Clean data shape: {self.clean_data.shape}")
        print(f"Number of frames: {self.num_frames}")

    def complex_to_real(self, data):
        # # Normalize complex data
        # max_abs = np.max(np.abs(data))
        # data = data / max_abs

        # Combine real and imaginary parts along new dimension
        real_part = np.real(data)
        imag_part = np.imag(data)
        # Shape: (num_samples, num_chirps, num_frames) -> (2, num_samples, num_chirps, num_frames)
        # Verify no data is lost in conversion
        print("complex_to_real() triggered:")
        print(f"Real part range: [{np.min(real_part):.3f}, {np.max(real_part):.3f}]")
        print(f"Imag part range: [{np.min(imag_part):.3f}, {np.max(imag_part):.3f}]")
        return np.stack((real_part, imag_part), axis=0)

    def __len__(self):
        return self.num_frames  # Return integer number of frames

    def __getitem__(self, idx):
        # Return shapes: (2, num_samples, num_chirps)
        x = torch.FloatTensor(self.interference_data[:, :, :, idx])
        y = torch.FloatTensor(self.clean_data[:, :, :, idx])
        return x, y
