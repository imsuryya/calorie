import streamlit as st
import cv2
import numpy as np
import time
import os
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Real-time Food Detector",
    page_icon="🍽️",
    layout="wide"
)

# Header
st.title("🍽️ Real-time Food Detector")
st.markdown("This application uses YOLO to identify food items in your camera feed in real-time.")

# Food categories that YOLO can detect
FOOD_CLASSES = [
    'apple', 'banana', 'orange', 'broccoli', 'carrot', 'hot dog', 
    'pizza', 'donut', 'cake', 'sandwich', 'bowl', 'cup'
]

# Dictionary with approximate calories for common foods
FOOD_CALORIES = {
    'apple': '95 calories',
    'banana': '105 calories',
    'orange': '62 calories',
    'broccoli': '55 calories per cup',
    'carrot': '50 calories per cup',
    'hot dog': '150 calories',
    'pizza': '285 calories per slice',
    'donut': '195 calories',
    'cake': '350 calories per slice',
    'sandwich': '200-500 calories',
    'bowl': 'Depends on contents',
    'cup': 'Depends on contents'
}

# Function to load YOLO model
@st.cache_resource
def load_yolo_model():
    # Load YOLO weights and config
    weights_path = os.path.join(os.path.dirname(__file__), "yolov3.weights")
    config_path = os.path.join(os.path.dirname(__file__), "yolov3.cfg")
    
    # Check if files exist
    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        st.error(f"YOLO files not found. Please download yolov3.weights and yolov3.cfg to {os.path.dirname(__file__)}")
        return None
    
    # Load YOLO network
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    
    # Try to use GPU if available
    try:
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    except:
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Get layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, output_layers

# Function to detect objects in an image
def detect_objects(frame, net, output_layers):
    height, width, _ = frame.shape
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    # Set input for the network
    net.setInput(blob)
    
    # Run forward pass
    outputs = net.forward(output_layers)
    
    # Initialize lists for detected objects
    class_ids = []
    confidences = []
    boxes = []
    
    # Process each output layer
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Filter out weak predictions
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-max suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Load COCO class names
    with open(os.path.join(os.path.dirname(__file__), "coco.names"), "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Prepare results
    results = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            results.append((label, confidence, (x, y, w, h)))
    
    return results

# Function to process frame and draw results
def process_frame(frame, net, output_layers):
    # Detect objects
    results = detect_objects(frame, net, output_layers)
    
    # Draw bounding boxes
    for label, confidence, (x, y, w, h) in results:
        if label in FOOD_CLASSES:  # Only show food items
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label with confidence
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Return processed frame and food items
    food_items = [item for item in results if item[0] in FOOD_CLASSES]
    return frame, food_items

# Function to generate analysis text
def generate_analysis(food_items):
    if not food_items:
        return "No food items detected in the frame."
    
    # Create a list of detected food items
    foods = {}
    for label, confidence, _ in food_items:
        if label in foods:
            foods[label] = max(foods[label], confidence)
        else:
            foods[label] = confidence
    
    # Generate analysis text
    analysis = "### Detected Food Items:\n\n"
    for food, confidence in foods.items():
        analysis += f"- **{food.title()}** (confidence: {confidence:.2f})\n"
        if food in FOOD_CALORIES:
            analysis += f"  - Approx. {FOOD_CALORIES[food]}\n"
    
    return analysis

# Initialize session state
if 'camera_on' not in st.session_state:
    st.session_state.camera_on = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'last_analysis_time' not in st.session_state:
    st.session_state.last_analysis_time = 0

# Create two columns with better proportions
col1, col2 = st.columns([0.6, 0.4])

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

# Results in the second column
with col2:
    st.header("Real-time Analysis")
    result_placeholder = st.container()

# Display camera feed if the camera is on
if st.session_state.camera_on:
    # Load YOLO model
    model_result = load_yolo_model()
    
    if model_result is None:
        st.error("Failed to load YOLO model. Please check if the model files are available.")
    else:
        net, output_layers = model_result
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam. Please check your camera connection.")
            st.session_state.camera_on = False
        else:
            try:
                stframe = camera_placeholder.empty()
                last_analysis_time = 0
                
                while st.session_state.camera_on:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture image")
                        break
                    
                    # Process frame
                    current_time = time.time()
                    if current_time - last_analysis_time >= 1:  # Update analysis every second
                        processed_frame, food_items = process_frame(frame.copy(), net, output_layers)
                        
                        # Generate analysis
                        analysis = generate_analysis(food_items)
                        st.session_state.analysis_result = analysis
                        st.session_state.last_analysis_time = current_time
                        
                        last_analysis_time = current_time
                    else:
                        # Just process frame for display without updating analysis
                        processed_frame, _ = process_frame(frame.copy(), net, output_layers)
                    
                    # Convert the color from BGR to RGB
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display the camera feed
                    stframe.image(frame_rgb, caption="Camera Feed", use_container_width=True)
                    
                    # Wait for a short time to update the UI
                    time.sleep(0.03)
                    
            except Exception as e:
                st.error(f"Camera Error: {str(e)}")
            finally:
                cap.release()

# Display analysis results if available
with result_placeholder:
    if st.session_state.analysis_result:
        st.markdown(st.session_state.analysis_result)
        st.caption(f"Last updated: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_analysis_time))}")
    else:
        st.info("No analysis results yet. Real-time analysis will appear here once the camera captures frames.")

# Add instructions in the sidebar
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Click the "Start Camera" button to begin
    2. Position food items in front of your camera
    3. The app will detect and analyze food items in real-time
    4. Nutritional information will be displayed when available
    """)
    
    st.header("Note")
    st.markdown("""
    This application uses YOLOv3 for object detection. It can detect these food items:
    - Apple, Banana, Orange
    - Broccoli, Carrot
    - Hot Dog, Pizza, Donut, Cake, Sandwich
    - Bowl, Cup (containers)
    
    For more accurate nutritional analysis, consider using a specialized food recognition API.
    """)