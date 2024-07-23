import torch
import numpy as np
import os
from src.domain import Domain
import json
import re

def getRandomAffine(scale_range=(0.95,1.3), rotation_range=(-60,60),
                    translation_range=(-0.15,0.15)):
 
    sx, sy = np.random.uniform(scale_range[0], scale_range[1], 2)

    angle = np.random.uniform(rotation_range[0], rotation_range[1])
    theta = np.radians(angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    
    tx, ty = np.random.uniform(translation_range[0], translation_range[1], 2)

    translate_to_origin = np.array([[1, 0, -10],
                                    [0, 1, -12.5],
                                    [0, 0,  1]])
    
    translate_back = np.array([[1, 0, 10],
                               [0, 1, 12.5],
                               [0, 0, 1]])
                               
    rotation_and_scale = np.array([[cos_theta * sx, -sin_theta * sy, 0],
                                   [sin_theta * sx, cos_theta * sy,  0],
                                   [0,               0,              1]])
    
    translation = np.array([[1, 0, tx],
                            [0, 1, ty],
                            [0, 0, 1]])
    
    test = np.dot(np.dot(translate_back, rotation_and_scale), translate_to_origin)

    rotation_and_scale = torch.tensor(test[:2, :2],dtype=torch.float32)
    
    translation = torch.tensor(test[:2,2],dtype=torch.float32)

    return rotation_and_scale, translation


def down_sample(img, factor=3):
    """
    Downsamples an image using average pooling.

    This function applies 2D average pooling to an input image `img` to reduce its dimensions by a specified factor. 
    The image is temporarily expanded with unsqueezed dimensions to fit the expected input shape of the average pooling operation 
    (which requires a 4D tensor), and then the extra dimensions are removed after pooling.

    Parameters:
    - img (torch.Tensor): The input image tensor. Expected to be a 2D tensor (height x width).
    - factor (int, optional): The factor by which to downsample the image. The kernel size and stride for the average pooling 
                              will both be equal to this factor. Default value is 3.

    Returns:
    - torch.Tensor: The downsampled image as a 2D tensor.
    """
    return torch.nn.functional.avg_pool2d(img.unsqueeze(0).unsqueeze(0), factor).squeeze(0).squeeze(0)

def save_tensors(tensors, base_folder, prefix="tensor", format=".pt", suffixes=None):
    """
    Saves a batch of tensors directly to files in a specified directory.

    Args:
    - tensors (torch.Tensor or list of torch.Tensor): The tensors to save.
    - base_folder (str): The base directory to save the tensors.
    - prefix (str): The prefix for each tensor file name.
    - format (str): The file extension, typically '.pt' for PyTorch tensors.

    Example usage:
    save_tensors(template_imgs, 'data/low_res_templates')
    """
    # Ensure the directory exists
    os.makedirs(base_folder, exist_ok=True)

    # Save each tensor in the directory
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]  # Wrap single tensor in a list for uniform handling
    for i, tensor in enumerate(tensors):
        if suffixes == None:
            file_path = os.path.join(base_folder, f"{prefix}_{i}{format}")
        else:
            file_path = os.path.join(base_folder, f"{prefix}_{suffixes[i]}{format}")
        torch.save(tensor, file_path)

def load_tensors(base_folder, pattern="*.pt"):
    """
    Loads all tensor files matching a pattern from a specified directory.

    Args:
    - base_folder (str): The directory from which to load the tensors.
    - pattern (str): The glob pattern to match files.
    
    Returns:
    - list of torch.Tensor: The loaded tensors.

    Example usage:
    loaded_tensors = load_tensors('data/low_res_templates')
    """
    from glob import glob
    import torch

    file_paths = glob(os.path.join(base_folder, pattern))
    tosort = [(file, float(file.split('.pt')[0].split('/')[-1])) for file in file_paths]
    tosort.sort(key = lambda a: a[1])
    file_paths = [tup[0] for tup in tosort]
    times = torch.tensor([tup[1] for tup in tosort])
    tensors = [torch.load(fp) for fp in file_paths]
    return tensors, times


def load_tensors2(base_folder, pattern="*.pt"):
    """
    Loads all tensor files matching a pattern from a specified directory.

    Args:
    - base_folder (str): The directory from which to load the tensors.
    - pattern (str): The glob pattern to match files.
    
    Returns:
    - list of torch.Tensor: The loaded tensors.

    Example usage:
    loaded_tensors = load_tensors('data/low_res_templates')
    """
    from glob import glob
    import re
    
    # Escape special characters in pattern except *
    escaped_pattern = re.escape(pattern).replace("\\*", "(.*)")
    regex_pattern = re.compile(escaped_pattern)
    
    file_paths = glob(os.path.join(base_folder, pattern))
    
    def extract_number(file):
        match = regex_pattern.search(file)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return match.group(1)
        return file
    
    tosort = [(file, extract_number(file)) for file in file_paths]
    
    tosort.sort(key=lambda x: x[1])
    
    loaded_tensors = [torch.load(file) for file, _ in tosort]
    
    return loaded_tensors

def load_tensors3(base_folder, pattern="*.pt"):
    """
    Loads all tensor files matching a pattern from a specified directory.

    Args:
    - base_folder (str): The directory from which to load the tensors.
    - pattern (str): The glob pattern to match files.
    
    Returns:
    - list of torch.Tensor: The loaded tensors.

    Example usage:
    loaded_tensors = load_tensors('data/low_res_templates')
    """
    from glob import glob
    import torch

    file_paths = glob(os.path.join(base_folder, pattern))
    tosort = [(f, float(re.search('(\d|\.)+', f.split('/')[-1]).group(0)[:-1])) for f in file_paths]
    tosort.sort(key = lambda a: a[1])
    file_paths = [tup[0] for tup in tosort]
    times = torch.tensor([tup[1] for tup in tosort])
    tensors = [torch.load(fp) for fp in file_paths]
    return tensors, times

class SuperResolutionProblem:

    def __init__(self, data):
        self.downsample_factor = data['downsample_factor']
        self.omega_R = data['omega_R']
        self.omega_T = data['omega_T']
        self.theta_R = data['theta_R']
        self.theta_T = data['theta_T']

        # load
        self.path_R = data['path_R']
        self.path_T = data['path_T']
        self.R = torch.load(data['path_R'])
        self.T, self.times = load_tensors3(data['path_T'], '*.pt')
        self.T = torch.stack(self.T, dim=0)
        self.times.requires_grad = False

        self.domain_R = Domain(torch.tensor(self.omega_R), torch.tensor((self.R.shape[0], self.R.shape[1])))
        self.domain_T = Domain(torch.tensor(self.omega_T), torch.tensor((self.T[0].shape[0], self.T[0].shape[1])))

    def get_templates(self, n=4):
        """return first n tensors with corresponding times"""
        return self.T[:n], self.times[:n]
    
    def save(self, filepath):
        with open(filepath, 'w') as f:
            f.write(json.dumps(
                {'downsample_factor': self.downsample_factor,
                 'omega_R': self.omega_R,
                 'omega_T': self.omega_T,
                 'theta_R': self.theta_R,
                 'theta_T': self.theta_T,
                 'path_R': self.path_R,
                 'path_T': self.path_T,
                 }
            ))
    
    def load(filepath):
        f = open(filepath, 'r').read()
        return SuperResolutionProblem(json.loads(f))


if __name__ == '__main__':
    path = os.path.abspath(os.getcwd())

    brain_data = SuperResolutionProblem({'downsample_factor': 4,
        'domain_R': {'omega': [0, 20, 0, 25], 'm': [128, 128]},
        'domain_T': {'omega': [0, 20, 0, 25], 'm': [128, 128]},
        'theta_R': 1,
        'theta_T': 1,
        'path_R': 'data/super_resolution/brainz/R_0.pt',
        'path_T': 'data/super_resolution/brainz/templates',
        'omega_R': [0, 20, 0, 25],
        'omega_T': [0, 20, 0, 25]
        })

    brain_data.save(path + 'data/super_resolution/brain_problem.json')



    brain_data = SuperResolutionProblem.load(path + 'data/super_resolution/brain_problem.json')
    print(brain_data.get_templates(4))