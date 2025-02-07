import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from train import CIFAR10_CNN  # Import model của bạn

# Danh sách nhãn của CIFAR-10
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Khởi tạo bộ biến đổi ảnh
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 có ảnh 32x32
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # Chuẩn hóa
])

# Load mô hình
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model = CIFAR10_CNN()
    model.load_state_dict(torch.load("./model/model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

def predict(image, model, device):
    image = transform(image).unsqueeze(0).to(device)  # Thêm batch dimension
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)  # Chuẩn hóa thành xác suất
    return probabilities.cpu().numpy().flatten()

# === STREAMLIT UI ===
st.title("CIFAR-10 Image Classifier")
st.write("Tải lên một ảnh, mô hình sẽ dự đoán lớp tương ứng!")

uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    model, device = load_model()
    probabilities = predict(image, model, device)

    # Lấy top-5 class dự đoán
    top5_indices = probabilities.argsort()[-5:][::-1]
    top5_labels = [CLASSES[i] for i in top5_indices]
    top5_probs = [probabilities[i] * 100 for i in top5_indices]  # Đổi thành %

    # Hiển thị kết quả
    st.subheader("🎯 Dự đoán:")
    for label, prob in zip(top5_labels, top5_probs):
        st.write(f"**{label.capitalize()}**: {prob:.2f}%")

    # Vẽ đồ thị
    fig, ax = plt.subplots()
    ax.barh(top5_labels[::-1], top5_probs[::-1], color='skyblue')
    ax.set_xlabel("Xác suất (%)")
    ax.set_title("Top-5 Dự Đoán")
    st.pyplot(fig)

st.write("🚀 Được phát triển với PyTorch & Streamlit!")
st.write("Author: hieunguyen-cyber")