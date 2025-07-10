
import numpy as np
from model import CNN_Classifier
from torch import softmax, Tensor, load
import matplotlib.pyplot as plt

class EDS_Simulator:
    
    def __init__(self, spectrums):

        self.spectrums = spectrums
        self.image = None
        self.classifier = CNN_Classifier(2)
        self.classifier.load_state_dict(
            load(r"C:\Users\nolan\OneDrive\Desktop\College\Research\ANL\SLADS\SLADS-Net\measurement\pretrained_models\CNN.pt", weights_only=True)
        )
    
    def generate_classifications(self):
        probabilities = softmax(
                            self.classifier(
                                Tensor(self.spectrums).float().reshape(-1, 1, 2048)
                            ).reshape(256, 256, 2), 2
                        ).detach().numpy()
            
        self.image = np.argmax(probabilities, axis=2)
    
    def get_measured_values(self, mask):
        pixel_coords = np.array(mask)
        row_indices, col_indices = pixel_coords.T
        return self.image[row_indices, col_indices]
    
    def plot_classifications(self):
        plt.imshow(self.image)
        plt.colorbar()
    
class EDS_Manager:
    
    def __init__(self, image_shape, spectrum_length = 2048):
        
        self.image = np.zeros(image_shape) - 1  # -1 -> undefined
        self.spectrums = np.zeros()
        self.classifier = CNN_Classifier(2)
        self.classifier.load_state_dict(
            load(r"C:\Users\nolan\OneDrive\Desktop\College\Research\ANL\SLADS\SLADS-Net\measurement\pretrained_models\CNN.pt", weights_only=True)
        )
    
    def update(self, position, spectrum):
        classification = np.argmax(softmax(self.classifier(Tensor(spectrum)))).detach().numpy()
        self.image[position] = classification
        return classification
    
    def get_position(self, position):
        res = self.image[position]
        return res if (res != -1) else None
    
    def get_measured_values(self, mask):
        return self.image[mask]
    
    def override_data(self, image, spectrum):
        pass