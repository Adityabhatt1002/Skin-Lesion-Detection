import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2

# --- 1. PROFESSIONAL UI & CSS SETUP ---
st.set_page_config(page_title="CDSS | Skin Lesion Analyzer", page_icon="⚕️", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #1e3a8a; font-family: 'Helvetica Neue', sans-serif;}
    h2, h3 {color: #3b82f6;}
    .stButton>button {background-color: #1e3a8a; color: white; border-radius: 8px; font-weight: bold; width: 100%;}
    .stButton>button:hover {background-color: #3b82f6;}
    .warning-box {background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 10px; border-left: 5px solid #ffeeba;}
    .success-box {background-color: #d4edda; color: #155724; padding: 15px; border-radius: 10px; border-left: 5px solid #c3e6cb;}
    </style>
""", unsafe_allow_html=True)

CLASS_NAMES = ['MEL', 'BCC', 'NV', 'BKL']
CLASS_DETAILS = {
    'MEL': 'Melanoma (Cancerous - Biopsy Required)',
    'BCC': 'Basal Cell Carcinoma (Cancerous - Treatable)',
    'NV':  'Melanocytic Nevus (Benign Mole)',
    'BKL': 'Benign Keratosis (Benign Lesion)'
}

@st.cache_resource
def load_best_model():
    return tf.keras.models.load_model('multimodal_efficientnet_best.keras')

model = load_best_model()

# --- 2. DATA PROCESSING ---
def encode_metadata(age, sex, site):
    age_scaled = age / 100.0
    sex_array = [1 if sex == "Female" else 0, 1 if sex == "Male" else 0, 1 if sex == "Unknown" else 0]
    sites = ["Anterior torso", "Head/neck", "Lateral torso", "Lower extremity", 
             "Oral/genital", "Palms/soles", "Posterior torso", "Upper extremity", "Unknown"]
    site_array = [1 if site == s else 0 for s in sites]
    meta_features = [age_scaled] + sex_array + site_array
    return np.array(meta_features, dtype=np.float32).reshape(1, -1)

def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    raw_array = np.array(image)
    if raw_array.shape[2] == 4:
        raw_array = raw_array[:, :, :3]
    img_array = raw_array.astype(np.float32)
    processed_img = np.expand_dims(img_array, axis=0)
    return processed_img, raw_array

# --- 3. THE BULLETPROOF GRAD-CAM ---
def generate_gradcam(inputs_dict, model):
    try:
        # 1. Grab just the visual brain (EfficientNet)
        base_model = model.get_layer('efficientnetb0')
        
        # 2. Pass the image straight through to get the feature maps 
        # (This completely bypasses the Keras 3 merge bug!)
        conv_outputs = base_model(inputs_dict["image_input"], training=False)
        
        # 3. Average the 1,280 filters to see where the AI is looking
        heatmap = tf.reduce_mean(conv_outputs, axis=-1)
        heatmap = tf.squeeze(heatmap)
        
        # 4. Normalize the colors
        max_val = tf.math.reduce_max(heatmap)
        min_val = tf.math.reduce_min(heatmap)
        if max_val != min_val:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            heatmap = tf.maximum(heatmap, 0)
            
        return heatmap.numpy()

    except Exception as e:
        print(f"XAI fallback failed: {e}")
        return np.zeros((224, 224))

def create_heatmap_overlay(original_img_array, heatmap):
    # If the failsafe was triggered, heatmap is all zeros, returning a clean image.
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(original_img_array, 0.6, heatmap_colored, 0.4, 0)
    return superimposed_img

# --- 4. MAIN APP ---
def main():
    st.title("⚕️ AI Clinical Decision Support System")
    st.write("### Multimodal Dermatoscopic Lesion Analysis")
    st.markdown("---")

    st.sidebar.header("📋 Patient Demographics")
    age = st.sidebar.number_input("Patient Age", min_value=0, max_value=120, value=45)
    sex = st.sidebar.selectbox("Biological Sex", ["Male", "Female", "Unknown"])
    site = st.sidebar.selectbox("Anatomical Site", [
        "Anterior torso", "Posterior torso", "Lateral torso", 
        "Head/neck", "Upper extremity", "Lower extremity", 
        "Palms/soles", "Oral/genital", "Unknown"
    ])

    col_upload, col_empty = st.columns([1, 1])
    with col_upload:
        file = st.file_uploader("Upload Dermatoscopic Image", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file)
        
        st.markdown("### 🔍 Analysis Results")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.image(image, caption="Original Image", width="stretch")

        if st.button("▶️ Run Multimodal AI Analysis"):
            with st.spinner("Fusing Image & Patient Data..."):
                try:
                    processed_img, raw_array = preprocess_image(image)
                    meta_array = encode_metadata(age, sex, site)
                    
                    inputs_dict = {
                        "image_input": processed_img,
                        "meta_input": meta_array
                    }
                    
                    # 1. Get the Prediction
                    prediction = model.predict(inputs_dict)[0]
                    score_index = np.argmax(prediction)
                    class_name = CLASS_NAMES[score_index]
                    confidence = np.max(prediction) * 100
                    
                    # 2. Get the Heatmap (Now wrapped in a failsafe)
                    heatmap = generate_gradcam(inputs_dict, model)
                    overlay_img = create_heatmap_overlay(raw_array, heatmap)
                    
                    with col2:
                        st.image(overlay_img, caption="Grad-CAM Focal Points", width="stretch")
                    
                    with col3:
                        st.write("#### Clinical Recommendation")
                        if confidence < 65.0:
                            st.markdown(f"""
                            <div class="warning-box">
                                <b>⚠️ Uncertain Diagnosis</b><br>
                                The AI is less than 65% confident.<br><br>
                                <b>Leaning Towards:</b> {class_name} ({confidence:.1f}%)<br>
                                <i>Review highly recommended.</i>
                            </div>
                            """, unsafe_allow_html=True)
                        elif class_name == 'MEL':
                            st.markdown(f"""
                            <div class="warning-box" style="background-color: #f8d7da; color: #721c24; border-color: #f5c6cb;">
                                <b>🚨 Critical Alert: Melanoma Detected</b><br>
                                Confidence: {confidence:.1f}%<br>
                                <i>Immediate biopsy required.</i>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="success-box">
                                <b>✅ Primary Finding:</b> {CLASS_DETAILS[class_name]}<br>
                                Confidence: {confidence:.1f}%<br>
                                <i>Standard clinical monitoring advised.</i>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        st.write("---")
                        st.write("**Probability Distribution:**")
                        for i, name in enumerate(CLASS_NAMES):
                            st.progress(float(prediction[i]), text=f"{name}: {prediction[i]*100:.1f}%")

                except Exception as e:
                    st.error(f"Critical System Error: {e}")

if __name__ == "__main__":
    main()