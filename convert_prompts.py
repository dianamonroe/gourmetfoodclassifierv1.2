import torch
import numpy as np

def convert_prompts():
    # Load the PyTorch tensor
    learned_prompts = torch.load("/workspace/gourmetfoodclassifierv1.2/learned_prompts.pt", map_location=torch.device('cpu'))
    
    # Convert to NumPy array
    learned_prompts_np = learned_prompts.detach().numpy()
    
    # Save as .npy file
    np.save("/workspace/gourmetfoodclassifierv1.2/learned_prompts.npy", learned_prompts_np)
    
    print("Learned prompts converted and saved as learned_prompts.npy")

if __name__ == "__main__":
    convert_prompts()

