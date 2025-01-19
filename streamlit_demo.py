import streamlit as st
from PIL import Image, ImageEnhance
import torch.backends
import torch.backends.mps
from inference import TrainStylePredictor
from torchvision import transforms
import numpy as np
import torch
import json
from io import BytesIO

# 设置页面配置
st.set_page_config(
    page_title="Fast-CLIPStyler Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .uploadedFile {
            margin: 10px 0;
        }
        .stSelectbox {
            margin: 10px 0;
        }
        .stTextArea {
            margin: 10px 0;
        }
        .stImage {
            margin: 10px 0;
        }
        .css-1544g2n {
            padding: 1rem 1rem 1.5rem;
        }
        .css-ocqkz7 {
            gap: 0.5rem;
        }
        .css-1y4p8pa {
            max-width: 100%;
            padding: 0rem 1rem 1rem;
        }
        .main > div {
            padding: 1rem;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 0rem;
        }
        h1 {
            margin-bottom: 1rem;
        }
        h3 {
            margin: 0.5rem 0;
        }
        .stMarkdown {
            margin: 0.5rem 0;
        }
    </style>
""", unsafe_allow_html=True)

# 初始化 session state
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None
if 'input_image' not in st.session_state:
    st.session_state.input_image = None

device = torch.device("cpu")
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using MPS")
    device = torch.device("mps")

def adjust_image(image, r_factor, g_factor, b_factor, brightness_factor, contrast_factor):
    """调整图片亮度和对比度"""
    # 调整 rgb
    image = adjust_rgb(image, r_factor, g_factor, b_factor)
    # 先调整亮度
    enhancer = ImageEnhance.Brightness(image)
    img_adjusted = enhancer.enhance(brightness_factor)
    
    # 再调整对比度
    enhancer = ImageEnhance.Contrast(img_adjusted)
    return enhancer.enhance(contrast_factor)

def adjust_rgb(image, r_factor, g_factor, b_factor):
    """调整图片的 RGB 通道"""
    # 转换为numpy数组
    img_array = np.array(image)
    
    # 分别调整RGB通道
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * r_factor, 0, 255)  # R通道
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] * g_factor, 0, 255)  # G通道
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * b_factor, 0, 255)  # B通道

    image = Image.fromarray(img_array.astype('uint8'))
    
    return image

@st.cache_resource
def load_model():
    return TrainStylePredictor()

trainer = load_model()

def style_transfer(input_image, style_description):
    output_image = trainer.test(input_image, style_description)
    return output_image

def main():
    st.title("Fast-CLIPStyler Demo")
    
    # 读取预设 prompts
    with open('prompts/prompts.json', 'r', encoding='utf-8') as f:
        preset_prompts = json.load(f)
    
    # 创建左右两列布局
    left_col, right_col = st.columns([1.2, 2])
    
    with left_col:
        with st.container():
            st.markdown("### 风格设置")
            # 添加预设 prompt 下拉框
            selected_prompt = st.selectbox(
                "选择预设风格",
                options=preset_prompts,
                format_func=lambda x: x
            )
            
            # 文本输入框，可以使用预设 prompt 或自定义
            prompt = st.text_area("自定义风格描述", value=selected_prompt, height=80)
            
            # st.markdown("### 上传图片")

            # 上传图片组件
            uploaded_file = st.file_uploader(
                "上传图片 支持 JPG, PNG, JPEG 格式",
                type=["jpg", "png", "jpeg"]
            )
            
            # 生成按钮
            generate_button = st.button("生成风格化图片", use_container_width=True)
            if generate_button and not uploaded_file:
                st.error("请先上传图片")
    
    with right_col:
        # 创建两列用于显示原图和结果
        img_col1, img_col2 = st.columns(2)

        with img_col1:
            st.markdown("### 原始图片")
        
            # 处理上传的图片
            if uploaded_file is not None:
                file_type = uploaded_file.name.split('.')[-1]
                if file_type in ["jpg", "png", "jpeg"]:
                    image = Image.open(uploaded_file)
                    with img_col1:
                        st.image(image, use_column_width=True)
                    transform = transforms.Compose([transforms.ToTensor()])
                    input_image = transform(np.array(image))[:3, :, :]
                    st.session_state.input_image = input_image.unsqueeze(0).to(device)
                else:
                    st.error("不支持的文件格式")

        # 将调整控件移到主列布局中
        if uploaded_file is not None:
            adj_col1, adj_col2 = st.columns(2)
            
            with adj_col1:
                brightness = st.slider("调整亮度", 0.1, 3.0, 1.0, 0.1)
                contrast = st.slider("调整对比度", 0.1, 3.0, 1.0, 0.1)

            with adj_col2:
                r_factor = st.slider("红色通道", 0.0, 2.0, 1.0, 0.1)
                g_factor = st.slider("绿色通道", 0.0, 2.0, 1.0, 0.1)
                b_factor = st.slider("蓝色通道", 0.0, 2.0, 1.0, 0.1)

        
        # 处理生成结果
        with img_col2:
            st.markdown("### 生成结果")
            if generate_button and uploaded_file is not None:
                with st.spinner('正在生成中...'):
                    output_image = style_transfer(st.session_state.input_image, prompt)[0]
                    to_pil = transforms.ToPILImage()
                    output_image = to_pil(output_image)
                    st.session_state.generated_image = output_image

            
            # 如果存在生成的图片，显示图片和下载按钮
            if st.session_state.generated_image is not None:
                # # 添加亮度调节滑动条
                # brightness = st.slider("调整亮度", 0.1, 3.0, 1.0, 0.1)
                # contrast = st.slider("调整对比度", 0.1, 3.0, 1.0, 0.1)
                # st.markdown("### 颜色调整")
                # r_factor = st.slider("红色通道", 0.0, 2.0, 1.0, 0.1)
                # g_factor = st.slider("绿色通道", 0.0, 2.0, 1.0, 0.1)
                # b_factor = st.slider("蓝色通道", 0.0, 2.0, 1.0, 0.1)
                # 应用亮度调整
                adjusted_image = adjust_image(
                    st.session_state.generated_image, 
                    r_factor, g_factor, b_factor, 
                    brightness, contrast
                )

                st.image(adjusted_image, use_column_width=True)
                
                # # 添加保存按钮
                # buf = BytesIO()
                # st.session_state.generated_image.save(buf, format='PNG')
                # byte_im = buf.getvalue()
                
                # st.download_button(
                #     label="保存生成的图片",
                #     data=byte_im,
                #     file_name="styled_image.png",
                #     mime="image/png",
                #     use_container_width=True
                # )
        # 将下载按钮移到滑块下方
        if st.session_state.generated_image is not None:
            # 添加保存按钮
            buf = BytesIO()
            st.session_state.generated_image.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            st.download_button(
                label="保存生成的图片",
                data=byte_im,
                file_name="styled_image.png",
                mime="image/png",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
