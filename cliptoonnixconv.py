import torch
from transformers import CLIPProcessor, CLIPModel

def convert_clip_to_onnx():
    # Load the CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model
    torch.onnx.export(model.vision_model, 
                      dummy_input, 
                      "clip_vision_model.onnx", 
                      input_names=['input'], 
                      output_names=['output'], 
                      dynamic_axes={'input' : {0 : 'batch_size'}, 
                                    'output' : {0 : 'batch_size'}},
                      opset_version=11)

    print("CLIP vision model converted to ONNX format")

if __name__ == "__main__":
    convert_clip_to_onnx()

