import torch
from transformers import CLIPModel, CLIPProcessor
import os

def convert_clip_to_onnx(learned_prompts_path, output_path):
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load learned prompts
    learned_prompts = torch.load(learned_prompts_path, map_location=torch.device('cpu'))

    # Create a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Define the custom forward function
    def custom_forward(self, pixel_values):
        image_features = self.get_image_features(pixel_values)
        similarities = torch.matmul(image_features, learned_prompts.t())
        return similarities

    # Replace the forward method of the model
    model.forward = custom_forward.__get__(model)

    # Export the model to ONNX
    torch.onnx.export(model,
                      dummy_input,
                      output_path,
                      input_names=['pixel_values'],
                      output_names=['similarities'],
                      dynamic_axes={'pixel_values': {0: 'batch_size'},
                                    'similarities': {0: 'batch_size'}},
                      opset_version=12)

    print(f"Model converted to ONNX and saved at {output_path}")

if __name__ == "__main__":
    learned_prompts_path = "/workspace/gourmetfoodclassifierv1.2/learned_prompts.pt"
    output_path = "models/clip_model_with_prompts.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    convert_clip_to_onnx(learned_prompts_path, output_path)

