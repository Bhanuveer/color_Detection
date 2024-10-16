import cv2
import numpy as np #A powerful library used for array operations. In this case, it's used to handle color boundaries and masks.

# Function to get the color name based on the HSV value
def get_color_name(hsv_value): 
    h, s, v = hsv_value    #(Hue, Saturation, Value) value of a pixel as input.

    '''Based on the h (hue) value, the function returns a color name.
    Red is detected when the hue is close to 0 or greater than 170.
    Green is detected for hues between 35 and 85.
    Blue is detected for hues between 85 and 130.
    You can modify the ranges to detect more colors if needed.'''

    if h < 10 or h > 170:  # Red
        return "Red"
    elif 20 < h < 35:  # Yellow
        return "Yellow"
    elif 35 < h < 85:  # Green
        return "Green"
    elif 85 < h < 130:  # Blue
        return "Blue"
    elif 130 < h < 170:  # Purple
        return "Purple"
    else:
        return "Unknown"

# Function to detect colors
def detect_color(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for red, green, and blue (in HSV)
    colors = {
        "Red": ([0, 120, 70], [10, 255, 255]),
        "Green": ([35, 100, 100], [85, 255, 255]),
        "Blue": ([100, 150, 0], [140, 255, 255])
    }
    
    # Iterate over defined colors and create masks
    for color_name, (lower, upper) in colors.items():
        lower = np.array(lower)
        upper = np.array(upper)
        
        # Create a mask for the current color
        mask = cv2.inRange(hsv_frame, lower, upper)
        
        # Find contours to detect objects of that color
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter out small contours
                # Draw a bounding box around the detected color
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
                
                # Get the color name and print it
                hsv_val = hsv_frame[y + h // 2, x + w // 2]  # Take the HSV value at the center of the contour
                detected_color = get_color_name(hsv_val)
                
                # Display the detected color name on the frame
                cv2.putText(frame, detected_color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Call the color detection function and show result in the same frame
    result_frame = detect_color(frame)
    
    # Display the frame with detected colors
    cv2.imshow("Color Detection", result_frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
