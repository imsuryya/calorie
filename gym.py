import streamlit as st
import cv2
import numpy as np
import requests
import base64
import json
import os
import time
from dotenv import load_dotenv

class FoodCalorieDetector:
    def __init__(self):
        # Streamlit page configuration
        st.set_page_config(page_title="Food Calorie Detector", page_icon="🍽️")
        st.title("Real-Time Food Calorie Detector")
       
        # API Key Configuration
        self.api_key = self.get_api_key()
       
        # Initialize session state variables
        if 'capture' not in st.session_state:
            st.session_state.capture = False
        if 'last_analysis_time' not in st.session_state:
            st.session_state.last_analysis_time = 0
   
    @staticmethod
    def get_api_key():
        load_dotenv()
        return os.environ.get('OPENROUTER_API_KEY')
    
    def encode_image(self, image):
        """Convert OpenCV image to base64 encoded string"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def get_food_analysis(self, image):
        """Analyze food image and estimate calories"""
        try:
            # Prepare API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/your-username/food-calorie-detector",
                "X-Title": "Food Calorie Detector"
            }
            
            payload = {
                "model": "qwen/qwen2.5-vl-72b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Identify the food items in this image and estimate their calories. Provide a detailed breakdown of each food item, its approximate portion size, and total calories. If multiple food items are present, list them separately."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image}"
                                }
                            }
                        ]
                    }
                ]
            }
            
            # Send request to OpenRouter
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                     headers=headers, 
                                     json=payload)
            
            if response.status_code == 200:
                analysis = response.json()['choices'][0]['message']['content']
                return analysis
            else:
                error_message = f"Error: {response.status_code} - {response.text}"
                st.error(error_message)
                return error_message
        
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            st.error(error_message)
            return error_message
    
    def run(self):
        """Main Streamlit application"""
        # Sidebar for controls
        st.sidebar.header("Webcam Food Calorie Detector")
        
        # Webcam capture
        run = st.sidebar.checkbox('Start/Stop Camera', key='camera_toggle')
        
        # Video capture
        if run:
            # Open webcam
            cap = cv2.VideoCapture(0)
            
            # Create placeholders
            frame_placeholder = st.empty()
            analysis_placeholder = st.empty()
            
            # Capture and process frames
            analysis_interval = 10  # seconds between analyses
            
            while run:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Convert frame from BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                frame_placeholder.image(frame_rgb, channels="RGB")
                
                # Automatic periodic analysis
                current_time = time.time()
                if (current_time - st.session_state.last_analysis_time) >= analysis_interval:
                    # Encode image
                    encoded_image = self.encode_image(frame)
                    
                    # Show loading
                    with st.spinner('Analyzing food...'):
                        # Get food analysis
                        food_analysis = self.get_food_analysis(encoded_image)
                        
                        # Display analysis
                        analysis_placeholder.write(food_analysis)
                        
                        # Update last analysis time
                        st.session_state.last_analysis_time = current_time
                
                # Manual analysis button with unique key
                if st.sidebar.button('Detect Food Calories', key=f'detect_calories_{int(current_time)}'):
                    # Encode image
                    encoded_image = self.encode_image(frame)
                    
                    # Show loading
                    with st.spinner('Analyzing food...'):
                        # Get food analysis
                        food_analysis = self.get_food_analysis(encoded_image)
                        
                        # Display analysis
                        analysis_placeholder.write(food_analysis)
                
                # Break if run is unchecked
                run = st.sidebar.checkbox('Start/Stop Camera', value=True, key='camera_toggle_2')
            
            # Release webcam
            cap.release()
        
        # Additional instructions
        st.sidebar.info("1. Start the camera\n2. Position food in frame\n3. Click 'Detect Food Calories'")

# Run the application
if __name__ == "__main__":
    detector = FoodCalorieDetector()
    detector.run()