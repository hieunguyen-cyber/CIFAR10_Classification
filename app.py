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
# UI chính
st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🚀", layout="wide")

# Tiêu đề chính với hiệu ứng
st.markdown(
    "<h1 style='text-align: center; color: #4A90E2;'>🚀 CIFAR-10 Image Classifier</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Tải lên một ảnh, mô hình sẽ dự đoán lớp tương ứng!</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# Upload ảnh
uploaded_file = st.file_uploader("📤 Chọn một ảnh...", type=["jpg", "png", "jpeg"], help="Chỉ hỗ trợ JPG, PNG, JPEG")
st.write("Các lớp có thể dự đoán là: plane, car, bird, cat, deer, dog, frog, horse, ship, truck")
if uploaded_file:
    col1, col2 = st.columns([1, 2])  # Chia bố cục 2 cột

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼 Ảnh đã tải lên", use_column_width=True)

    with col2:
        with st.spinner("⏳ Đang dự đoán..."):
            model, device = load_model()
            probabilities = predict(image, model, device)

            # Lấy top-5 class dự đoán
            top5_indices = probabilities.argsort()[-5:][::-1]
            top5_labels = [CLASSES[i] for i in top5_indices]
            top5_probs = [probabilities[i] * 100 for i in top5_indices]  # Đổi thành %

            # Hiển thị kết quả
            st.markdown("<h3 style='color: #27AE60;'>🎯 Kết quả dự đoán:</h3>", unsafe_allow_html=True)

            for label, prob in zip(top5_labels, top5_probs):
                st.markdown(f"<p style='font-size:18px;'>✅ <b>{label.capitalize()}</b>: <span style='color:#E74C3C;'>{prob:.2f}%</span></p>", unsafe_allow_html=True)

            # Vẽ biểu đồ
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(top5_labels[::-1], top5_probs[::-1], color=['#4A90E2', '#50E3C2', '#F5A623', '#E74C3C', '#8B572A'])
            ax.set_xlabel("Xác suất (%)")
            ax.set_title("📊 Top-5 Dự Đoán")
            st.pyplot(fig)

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        🚀 Được phát triển với <b>PyTorch</b> & <b>Streamlit</b> | 
        👨‍💻 Author: <b>hieunguyen-cyber</b> | 
        🔗 <a href="https://github.com/hieunguyen-cyber" target="_blank" style="color:#4A90E2; text-decoration:none;">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)