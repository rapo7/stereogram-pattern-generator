import numpy as np
import torch
from torchvision.transforms import Compose, ToTensor
from matplotlib import cm
import requests
import streamlit as st
import base64
from sklearn.cluster import KMeans
from PIL import Image
import io
import cv2
import os

# Cloudflare API Key and Account ID (replace with your actual details)

CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN")


def extract_color(img):
    """Extract dominant colors from the image using k-means clustering and return their hex codes."""

    img_rgb = np.array(img)

    # Reshaping the image to a 2D array of pixels
    pixels = img_rgb.reshape((-1, 3))

    # Apply k-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)

    # Get the dominant colors
    dominant_colors = kmeans.cluster_centers_.astype(int)

    # Convert RGB to hex
    hex_colors = [
        "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])
        for color in dominant_colors
    ]

    return hex_colors


def extract_shapes(img):
    """Extract shapes from the image and list them."""
    gray = np.array(img.convert("L"))

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = {"Triangle": 0, "Rectangle": 0, "Square": 0, "Circle": 0}

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        num_vertices = len(approx)

        if num_vertices == 3:
            shapes["Triangle"] += 1
        elif num_vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 < aspect_ratio < 1.05:
                shapes["Square"] += 1
            else:
                shapes["Rectangle"] += 1
        elif num_vertices > 4:
            shapes["Circle"] += 1

    return shapes


def generate_prompt(dominant_colors, shapes):

    prompt = f"""
    Convert the following hex colors to human-readable colors: {', '.join(dominant_colors)}. Identify the most occurring shapes from the list: {', '.join([f'{k}: {v}' for k, v in shapes.items()])}.
    Using this data, create an image generator prompt for a tile pattern that will be repeated. Incorporate the identified colors and shapes, including circular patterns. Blend the shapes seamlessly into the pattern, 
    prioritizing the most frequent colors and shapes.
    Please provide the final image generator prompt only.
    """

    st.write(prompt)
    response = requests.post(
        f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/google/gemma-7b-it-lora",
        headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
        json={"prompt": prompt},
    )
    result = response.json()
    return result["result"]["response"]


# Function to generate pattern image from text prompt using Cloudflare API
def generate_pattern_from_prompt(prompt):
    try:
        data = {
            "prompt": prompt,
        }

        response = requests.post(
            f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/black-forest-labs/flux-1-schnell",
            headers={"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"},
            json=data,
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            base64_image = data.get("result", {}).get("image", None)
            if base64_image is None:
                print("Error: No 'image' field in response data")
                return None

            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))
            image = image.resize((384, 384))
            return image
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"Error generating pattern from prompt using Cloudflare AI: {e}")
        return None


# Load MiDaS model for depth estimation
@st.cache_resource
def load_midas_model(model_type="MiDaS_small"):
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    transform = Compose([ToTensor()])
    return midas, transform


# Function to estimate depth map using MiDaS
def estimate_depth(image, model, transform):
    original_size = image.size
    image = image.resize((384, 384))
    input_batch = transform(image).unsqueeze(0)
    with torch.no_grad():
        depth_map = model(input_batch).squeeze().cpu().numpy()

    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = (depth_map * 255).astype(np.uint8)
    depth_map = cv2.resize(depth_map, original_size, interpolation=cv2.INTER_LINEAR)
    return depth_map


def process_uploaded_file(uploaded_file):
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        return img

    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return None


# Streamlit UI
st.title("Autostereogram Generator")

model_type = st.sidebar.selectbox(
    "Select Depth Model", ["MiDaS_small", "DPT_Large", "DPT_Hybrid"]
)


midas, transform = load_midas_model(model_type)

uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = process_uploaded_file(uploaded_file)

    if input_image is not None:
        st.image(input_image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Estimating depth map..."):
            depth_map = estimate_depth(input_image, midas, transform)

        if depth_map is not None:
            depth_map_colored = cm.get_cmap("gray")(depth_map / 255)[:, :, :3] * 255
            depth_map_colored = depth_map_colored.astype(np.uint8)
            st.image(depth_map_colored, caption="Depth Map", use_container_width=True)

        with st.spinner("Extracting dominant colors..."):
            dominant_colors = extract_color(input_image)
        st.write("Dominant Colors:")
        for color in dominant_colors:
            st.markdown(
                f'<div style="background-color: {color}; padding: 10px;">{color}</div>',
                unsafe_allow_html=True,
            )

        with st.spinner("Extracting shapes..."):
            shapes = extract_shapes(input_image)

        st.text("There are the following shapes in the image:")
        for shape, count in shapes.items():
            st.text(f"{shape}: {count}")

        shapes = dict(sorted(shapes.items(), key=lambda item: item[1], reverse=True))

        with st.spinner("Generating prompt..."):
            prompt = generate_prompt(dominant_colors, shapes)
        st.text_area(label="Prompt", value=prompt)

        with st.spinner("Generating pattern..."):
            pattern = generate_pattern_from_prompt(prompt)
        if pattern is not None:
            st.image(pattern, caption="Generated Pattern", use_container_width=True)
        else:
            st.error("Failed to generate pattern from prompt")
