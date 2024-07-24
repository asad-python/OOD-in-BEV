import torch

# Load the model onto CPU
model_path = 'model525000.pt'
model = torch.load(model_path, map_location=torch.device('cpu'))

# Check if the model is a state dictionary
if isinstance(model, dict):
    # Print the keys in the dictionary to understand its structure
    print("Keys in the loaded model dictionary:", model.keys())

    # Summarize model architecture parameters
    print("\nModel Architecture Summary:")
    for key, value in model.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.size()}")
else:
    # Print the model architecture
    print("Model Architecture:")
    print(model)

# Check if classes are stored in the model
if isinstance(model, dict):
    if 'classes' in model:
        classes = model['classes']
        print("Classes:", classes)
else:
    if hasattr(model, 'classes'):
        print("Classes:", model.classes)
    else:
        print("Class information not found in the model.")
