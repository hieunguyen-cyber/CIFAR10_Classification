import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
from train import CIFAR10_CNN  # Import model

# Danh sách nhãn của CIFAR-10
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Khởi tạo bộ biến đổi ảnh
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 có ảnh 32x32
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Chuẩn hóa
])

def predict(image_path, device=None):
    """
    Dự đoán lớp của ảnh đầu vào bằng mô hình CNN.

    Args:
        image_path (str): Đường dẫn đến ảnh cần dự đoán.
        device (str hoặc torch.device, optional): Thiết bị sử dụng ('cpu', 'cuda', hoặc 'mps').

    Returns:
        str: Nhãn dự đoán.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                              "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load mô hình (mặc định là 'model.pth')
    model_path = "./model/model.pth"
    model = CIFAR10_CNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load ảnh và tiền xử lý
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Thêm batch dimension

    # Dự đoán
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return CLASSES[predicted.item()]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [device]")
        sys.exit(1)

    image_path = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else None

    result = predict(image_path, device)
    print(f"Predicted class: {result}")