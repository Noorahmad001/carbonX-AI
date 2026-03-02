import streamlit as st
import plotly.graph_objects as go
import random
from PIL import Image
import time
import torch
from torchvision import models, transforms
import requests

# Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.splitlines()

# --- CONFIGURATION & THEME ---
st.set_page_config(page_title="carbonX AI", page_icon="🌿", layout="wide")

# Custom CSS for Eco-Theme
st.markdown("""
    <style>
    .main { background-color: #f0f7f4; }
    .stButton>button { background-color: #2d6a4f; color: white; border-radius: 8px; width: 100%; }
    .eco-card { 
        background-color: white; 
        padding: 20px; 
        border-radius: 15px; 
        border-left: 5px solid #52b788;
        box-shadow: 2px 2px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .points-badge {
        background-color: #d8f3dc;
        color: #1b4332;
        padding: 10px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOCK ML LOGIC ---

# Load MobileNetV2 using PyTorch (Much lighter and faster to load)
model = models.mobilenet_v2(weights='DEFAULT')
model.eval()

def classify_waste_logic(img):
    img = img.convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) 

    with torch.no_grad():
        output = model(input_batch)
    
    # Get the top prediction
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    conf, index = torch.max(probabilities, 0)
    
    # Get the actual word from our labels list
    raw_name = labels[index.item()].lower() 

    # Map that word to a waste category
    label, tip = map_to_waste_category(raw_name)
    
    return label, conf.item() * 100, tip, raw_name

def map_to_waste_category(text):
    # This now checks the actual word returned by the AI
    if any(x in text for x in ['bottle', 'plastic', 'cup', 'container', 'poly']):
        return "Plastic", "Rinse and dry. Check for resin codes 1 or 2."
    if any(x in text for x in ['box', 'paper', 'cardboard', 'envelope', 'carton']):
        return "Paper", "Flatten and keep dry. No food-stained paper!"
    if any(x in text for x in ['computer', 'keyboard', 'phone', 'mouse', 'laptop', 'screen']):
        return "E-waste", "Contains toxic metals. Take to a tech recycler."
    if any(x in text for x in ['apple', 'banana', 'food', 'orange', 'meat', 'pot', 'broccoli']):
        return "Organic", "Compost or place in the green bin."
    if any(x in text for x in ['glass', 'jar', 'wine', 'beer', 'bottle']):
        return "Glass", "100% recyclable. Remove metal caps first."
    
    return "General Waste", f"Detected as '{text}'. Not sure? Check local guidelines."
# --- SIDEBAR / USER PROFILE ---
with st.sidebar:
    st.title("🌿 carbonX AI")
    st.markdown("### User: **EcoWarrior_01**")
    st.markdown('<div class="points-badge">Current Points: 1,420 XP</div>', unsafe_allow_html=True)
    st.divider()
    st.info("💡 **Smart Tip:** Switching to LED bulbs reduces your lighting CO2 by 80%.")

# --- MAIN DASHBOARD ---
tab1, tab2, tab3 = st.tabs(["🗑️ Waste AI", "📊 Footprint Calc", "🏆 Leaderboard"])

# TAB 1: AI WASTE CLASSIFIER
with tab1:
    st.header("AI-Powered Waste Segregation")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload or Capture")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Waste Item', use_container_width=True)
            if st.button("Analyze Waste Material"):
                with st.spinner('AI analyzing material composition...'):
                    label, conf, tip,raw_name = classify_waste_logic(image)
                    st.session_state['last_result'] = (label, conf, tip, raw_name)

    with col2:
        st.subheader("Segregation Result")
        # Check if key exists AND isn't None
        if st.session_state.get('last_result'):
            label, conf, tip, raw_name = st.session_state['last_result']
            st.markdown(f"""
                <div class="eco-card">
                    <h2 style="color:#2d6a4f;">{label}</h2>
                    <p><b>Object Detected:</b> {raw_name.title()}</p>
                    <p><b>AI Confidence:</b> {conf:.1f}%</p>
                    <hr>
                    <p><b>How to Recycle:</b> {tip}</p>
                </div>
            """, unsafe_allow_html=True)
            st.success(f"Success! +10 XP added to EcoWarrior_01")
        else:
            st.info("Upload an image and click 'Analyze' to start.")

# TAB 2: CARBON FOOTPRINT CALCULATOR
with tab2:
    st.header("Carbon Footprint Analyzer")
    
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            km = st.number_input("Transport (KM driven per week)", 0, 2000, 50)
            kwh = st.number_input("Electricity (kWh per month)", 0, 5000, 150)
        with c2:
            meals = st.slider("Meat-based meals per week", 0, 21, 7)
            shopping = st.slider("New products purchased/month", 0, 50, 5)

    # Calculation (kg CO2)
    # Average factors: Car 0.12kg/km, Grid 0.45kg/kWh, Meat 2.5kg/meal, Shopping 5kg/item
    total = (km * 0.12 * 4) + (kwh * 0.45) + (meals * 2.5 * 4) + (shopping * 5)
    trees = round(total / 21, 1) # 1 tree absorbs ~21kg/year

    st.divider()
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.metric("Estimated Monthly CO₂", f"{round(total, 2)} kg")
        st.metric("Trees Required to Offset", f"{trees} Trees")
        
        if total < 200:
            st.balloons()
            st.write("🌟 **Status: Eco-Warrior.** Your footprint is significantly lower than average!")
        else:
            st.warning("⚠️ **Status: High Impact.** Consider reducing meat intake or carpooling.")

    with res_col2:
        # Visualizing the breakdown
        fig = go.Figure(data=[go.Pie(
            labels=['Transport', 'Electricity', 'Diet', 'Consumption'],
            values=[km*0.12*4, kwh*0.45, meals*2.5*4, shopping*5],
            hole=.4,
            marker_colors=['#1b4332', '#2d6a4f', '#52b788', '#95d5b2']
        )])
        fig.update_layout(title_text="Emission Distribution", margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

# TAB 3: LEADERBOARD
with tab3:
    st.header("Community Leaderboard")
    data = {
        "Rank": [1, 2, 3, 4],
        "User": ["GreenQueen", "SolarSam", "You", "EarthFirst"],
        "Points": [4500, 3200, 1420, 980],
        "Badges": ["🏅 Planet Protector", "🥈 Carbon Neutral", "🥉 Recycler", "🌱 Seedling"]
    }
    st.table(data)