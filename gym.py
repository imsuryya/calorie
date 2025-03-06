import streamlit as st
import cv2
import base64
import requests
import os
import json
import time
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Food Analyzer",
    page_icon="🍽️",
    layout="wide"
)

# Header
st.title("🍽️ Real-time Food Analyzer")
st.markdown("This application uses AI to identify food items and estimate their nutritional content.")

# Function to encode the image to base64
def encode_image(image):
    # Convert OpenCV image to PIL Image
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Create a bytes buffer for the image
    buffered = io.BytesIO()
    
    # Save the image to the buffer as JPEG
    pil_img.save(buffered, format="JPEG")
    
    # Encode the image as base64
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_str

# Function to analyze the food in the image using OpenRouter API
def analyze_food(image):
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        st.error("OpenRouter API key is missing. Please set it in your .env file.")
        return None
    
    # Encode the image to base64
    base64_image = encode_image(image)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = ("Analyze the given food image and estimate the calorie content. "
              "Break down the calorie values based on individual food items if multiple items are detected. "
              "Also, provide an estimated macronutrient distribution (carbohydrates, proteins, and fats) if possible. "
              "Return your response in JSON format with the following structure: "
              "{'food_items': [{'name': 'item name', 'calories': X, 'carbs': X, 'protein': X, 'fat': X}], "
              "'total_calories': X, 'total_carbs': X, 'total_protein': X, 'total_fat': X}")
    
    data = {
        "model": "qwen/qwen2.5-vl-72b-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ],
        "max_tokens": 1024
    }
    
    try:
        with st.spinner("Analyzing food..."):
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # Try to extract JSON from the response
                try:
                    # Look for JSON in the response
                    import re
                    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # If no code block, try to find JSON directly
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                        else:
                            # If JSON parsing fails, return the text response
                            return {
                                "raw_response": content,
                                "food_items": [{"name": "Food Analysis", "calories": "See details", "carbs": "-", "protein": "-", "fat": "-"}],
                                "total_calories": "See details"
                            }
                    
                    return json.loads(json_str)
                except Exception as e:
                    # If JSON parsing fails, return the text response in a user-friendly format
                    return {
                        "error": str(e),
                        "raw_response": content,
                        "food_items": [{"name": "Food Analysis", "calories": "See details", "carbs": "-", "protein": "-", "fat": "-"}],
                        "total_calories": "See details"
                    }
            else:
                return {"error": f"API request failed with status code {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": str(e)}

# Create two columns with better proportions
col1, col2 = st.columns([0.6, 0.4])

# Initialize session state for camera
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False

if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = 0

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

if 'frame_to_analyze' not in st.session_state:
    st.session_state.frame_to_analyze = None

# Toggle camera function
def toggle_camera():
    st.session_state.camera_on = not st.session_state.camera_on
    if not st.session_state.camera_on:
        st.session_state.analysis_result = None

# Button to toggle camera
with col1:
    st.button(
        "Start Camera" if not st.session_state.camera_on else "Stop Camera",
        on_click=toggle_camera,
        key="camera_button",
        use_container_width=True
    )
    
    # Camera feed placeholder
    camera_placeholder = st.empty()
    
    # Analyze button
    if st.session_state.camera_on:
        if st.button("Analyze Current Frame", use_container_width=True):
            if st.session_state.frame_to_analyze is not None:
                st.session_state.analysis_result = analyze_food(st.session_state.frame_to_analyze)
                st.session_state.last_analysis_time = time.time()

# Results in the second column
with col2:
    st.header("Analysis Results")
    result_placeholder = st.container()  # Use container instead of empty to ensure visibility

# Display camera feed if the camera is on
if st.session_state.camera_on:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera connection.")
        st.session_state.camera_on = False
    else:
        try:
            while st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image")
                    break
                
                # Store the current frame for analysis
                st.session_state.frame_to_analyze = frame.copy()
                
                # Convert the color from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the camera feed - fixed deprecated parameter
                camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Wait for a short time to update the UI
                time.sleep(0.1)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            cap.release()

# Display analysis results if available - always show the container
with result_placeholder:
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        if "error" in result:
            st.error(f"Analysis Error: {result['error']}")
            if "raw_response" in result:
                with st.expander("Show raw response", expanded=True):
                    st.write(result["raw_response"])
        
        # Display food items
        if "food_items" in result:
            for item in result["food_items"]:
                with st.expander(f"{item['name']} - {item.get('calories', 'N/A')} kcal", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Carbs", f"{item.get('carbs', 'N/A')}g")
                    col2.metric("Protein", f"{item.get('protein', 'N/A')}g")
                    col3.metric("Fat", f"{item.get('fat', 'N/A')}g")
        
        # Display total nutrition if available
        if "total_calories" in result:
            st.subheader("Total Nutrition")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Calories", f"{result.get('total_calories', 'N/A')} kcal")
            col2.metric("Carbs", f"{result.get('total_carbs', 'N/A')}g")
            col3.metric("Protein", f"{result.get('total_protein', 'N/A')}g")
            col4.metric("Fat", f"{result.get('total_fat', 'N/A')}g")
        
        # Display raw response if JSON parsing failed but we have a response
        if "raw_response" in result and not result.get("error"):
            with st.expander("Detailed Analysis", expanded=True):
                st.write(result["raw_response"])
        
        # Display when the analysis was done
        st.caption(f"Analysis completed {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_analysis_time))}")
    else:
        # Show a placeholder message when no analysis has been performed
        st.info("No analysis results yet. Start the camera and click 'Analyze Current Frame' to see results.")

# Instructions
st.markdown("---")
st.subheader("How to use")
st.markdown("""
1. Click the 'Start Camera' button to turn on your webcam
2. Position your food in the camera view
3. Click 'Analyze Current Frame' to get nutritional information
4. Review the analysis results in the right panel
""")

# Environment setup instructions
st.sidebar.title("Setup Instructions")
st.sidebar.info("""
Before running this app, you need to:

1. Create a `.env` file in the same directory as your script
2. Add your OpenRouter API key to the `.env` file:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```
3. Install required dependencies:
   ```
   pip install streamlit opencv-python python-dotenv requests pillow
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```
""")