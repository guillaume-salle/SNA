import os
import h5py
import time
from tqdm import tqdm
from BaseDataGenerator import BaseDataGenerator
import torch


def append_chunk_to_hdf5(dset_X, dset_y, data_chunk, has_y, x_dim, y_dim):
    """Helper function to append a chunk of data to HDF5 datasets."""
    current_size = dset_X.shape[0]
    if has_y:
        X_chunk, y_chunk = data_chunk
        chunk_len = X_chunk.shape[0]
        dset_X.resize((current_size + chunk_len, x_dim))
        dset_y.resize((current_size + chunk_len, y_dim))
        dset_X[current_size:] = X_chunk.cpu().numpy()
        dset_y[current_size:] = y_chunk.cpu().numpy()
    else:
        X_chunk = data_chunk
        chunk_len = X_chunk.shape[0]
        dset_X.resize((current_size + chunk_len, x_dim))
        dset_X[current_size:] = X_chunk.cpu().numpy()
    return chunk_len


def save_data_to_hdf5(generator: BaseDataGenerator, filepath, num_samples, chunk_size):
    """
    Generates data using the provided generator instance and saves it to HDF5.

    Args:
        generator: An instance of a BaseDataGenerator subclass.
        filepath (str): Path to save the HDF5 file.
        num_samples (int): Total number of samples to generate.
        chunk_size (int): Size of chunks for generation and saving.
    """
    print(f"--- Starting HDF5 Saving ---")
    print(f"Generator type: {generator.__class__.__name__}")
    print(f"Output file: {filepath}")

    try:
        # --- Prepare HDF5 File ---
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with h5py.File(filepath, "w") as f:
            # Determine data structure and dimensions based on first chunk
            print("Generating first chunk to determine structure...")
            first_chunk_data = generator.generate_chunk(min(chunk_size, num_samples))
            has_y = isinstance(first_chunk_data, tuple)

            # Initialize datasets
            x_dim = first_chunk_data[0].shape[1] if has_y else first_chunk_data.shape[1]
            hdf5_chunk_shape_X = (min(chunk_size, num_samples), x_dim)
            dset_X = f.create_dataset(
                "X",
                shape=(0, x_dim),
                maxshape=(num_samples, x_dim),
                dtype="float32",
                chunks=hdf5_chunk_shape_X,
                compression="gzip",
            )
            dset_y = None
            if has_y:
                y_dim = first_chunk_data[1].shape[1]
                hdf5_chunk_shape_y = (min(chunk_size, num_samples), y_dim)
                dset_y = f.create_dataset(
                    "y",
                    shape=(0, y_dim),
                    maxshape=(num_samples, y_dim),
                    dtype="float32",
                    chunks=hdf5_chunk_shape_y,
                    compression="gzip",
                )
                print(f"Detected structure with y (shape: {y_dim})")
            else:
                print(f"Detected structure with X only (shape: {x_dim})")

            # Process first chunk
            num_generated = append_chunk_to_hdf5(
                dset_X, dset_y, first_chunk_data, has_y, x_dim, y_dim if has_y else None
            )
            del first_chunk_data

            # --- Generate Remaining Data in Chunks ---
            with tqdm(
                total=num_samples,
                initial=num_generated,
                desc=f"Saving {os.path.basename(filepath)}",
            ) as pbar:
                while num_generated < num_samples:
                    current_chunk_size = min(chunk_size, num_samples - num_generated)
                    if current_chunk_size <= 0:
                        break

                    data_chunk = generator.generate_chunk(current_chunk_size)
                    processed_len = append_chunk_to_hdf5(
                        dset_X, dset_y, data_chunk, has_y, x_dim, y_dim if has_y else None
                    )

                    num_generated += processed_len
                    pbar.update(processed_len)
                    del data_chunk

            # --- Store Metadata and True Parameters ---
            metadata = generator.get_metadata()
            metadata["num_samples"] = num_samples
            metadata["chunk_size"] = chunk_size
            metadata["creation_timestamp"] = time.time()
            for key, value in metadata.items():
                try:
                    f.attrs[key] = value
                except TypeError:  # Handle non-native HDF5 types like lists
                    f.attrs[key] = str(value)

            true_params = generator.get_true_parameters()
            if true_params:
                param_group = f.create_group("true_parameters")
                for key, value in true_params.items():
                    if isinstance(value, torch.Tensor):
                        param_group.create_dataset(key, data=value.cpu().numpy())

    except Exception as e:
        print(f"Error during HDF5 saving: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        raise

    print(f"Finished saving dataset: {filepath}")
    print("-" * 30)
