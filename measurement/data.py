
import os
import numpy as np
from imageio import imread
import glob
from skimage.util import random_noise

def read_emsa_spectrum(filepath: str, header_lines: int = 31) -> np.ndarray:
    """
    Reads a .emsa file and extracts the numerical data spectrum.

    Args:
        filepath (str): The full path to the .emsa file.
        header_lines (int): The number of header lines to skip.

    Returns:
        np.ndarray: A 1D NumPy array containing the spectrum data.
    """
    try:
        spectrum = np.loadtxt(filepath, skiprows=header_lines)
        return spectrum
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return np.array([]) # Return an empty array on error
    except ValueError:
        print(f"Error: Could not convert data to numbers after the header.")
        print("Please check the file to ensure all lines after the header are numeric.")
        return np.array([])
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return np.array([])
    
def spectrums_from_directory(directory):
    emsa_files = sorted(glob.glob(os.path.join(directory, '*.emsa')))
    if emsa_files:
        list_of_spectra = [read_emsa_spectrum(f) for f in emsa_files]
        spectra_array = np.array([s for s in list_of_spectra if s.size > 0])
        
        if spectra_array.size == 0:
            print("Could not read any valid spectra from the files found.")
        
        return spectra_array
    else:
        print(f"No .emsa files found in the directory: {directory}")

def generate_training_data(image_directory, target_size: tuple, spectrum_directories: list, spectrum_range = None) -> np.ndarray:
    """Generates training data for the CNN classifier network.

    Args:
        image_folder (str, Path, etc.): the directory in which the SEM images to be preprocessed into training data are located
        L (int, optional): Number of classes driving dimension of spectral vector subspace. Defaults to 2.
    """
    
    L = len(spectrum_directories)
    
    if not os.path.isdir(image_directory):
        raise FileNotFoundError(f"Error: The directory '{image_directory}' was not found.")
    
    if not (isinstance(target_size, tuple) and len(target_size) == 2 and 
            all(isinstance(x, int) and x > 0 for x in target_size)):
        raise ValueError("Error: target_size must be a tuple of two positive integers (height, width).")

    processed_images = []
    supported_extensions = ('.jpg', '.jpeg', '.png')

    print(f"Scanning directory: {image_directory}")
    print(f"Looking for files with extensions: {supported_extensions}")
    print(f"All images will be resized to: {target_size}")

    for filename in os.listdir(image_directory):        
        # Check if file has one of the supported extensions
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(image_directory, filename)
            
            try:
                img_gray = imread(file_path, mode='L')
                processed_images.append(img_gray)
            except Exception as e:
                print(f"  - Could not read or process {filename}. Reason: {e}")

    if not processed_images:
        print("\nWarning: No valid images were found or processed. Returning an empty array.")
        return np.array([])

    # now with shape of (number_of_images, height, width)
    print(f"\nSuccessfully compiled {len(processed_images)} images.")
    images = np.stack(processed_images, axis=0)
    
    # We now segregate the images into phases
    # They are already split into black / white so we just need to normalize
    images[images > 128] = 255
    images[images <= 128] = 0
    images = images // 255

    # We now assign spectra to each pixel
    # The way I'm doing it is the easiest but not super efficient, I'm just randomly patterning each spectra over an array with shape equal to the iamge stack
    # Then I'm stitching them together using masks according to the class labels
    spec_lists = [spectrums_from_directory(directory) for directory in spectrum_directories]
    if spectrum_range:
        for idx in range(len(spec_lists)):
            spec_lists[idx] = spec_lists[idx][spectrum_range[0]:spectrum_range[1]]

    def spectral_vol(spec_list):
        random_indices = np.random.randint(0, L - 1, size = images.shape)
        spec_vol = spec_list[random_indices]
        
        # we pass in some poisson noise
        random_noise(spec_vol, mode='poisson', clip=True)
        spec_vol = spec_vol + np.random.poisson(spec_vol)
        return spec_vol
        
    HSI = None
    for idx in range(L):
        spec = spec_lists[idx]
        spec_vol = spectral_vol(spec)
        if HSI is None:
            HSI = np.zeros(spec_vol.shape)
        print(f"Masking over data for class {idx}...")
        HSI[images == idx] = spec_vol[images == idx]
        del spec_vol
    
    # Replace 1% of spectrums with ill-spectrums
    spatial_size = np.prod(HSI.shape[:-1])
    N_replace = int(0.01 * spatial_size)
    spec_size = HSI.shape[-1]
    ill_spectrum = np.random.poisson(lam=20, size=(N_replace, spec_size))
    flat_indices = np.random.choice(spatial_size, N_replace, replace=False)
    spatial_indices_to_replace = np.unravel_index(flat_indices, HSI.shape[:-1])
    HSI[spatial_indices_to_replace] = ill_spectrum
    
    del ill_spectrum

    return HSI, images