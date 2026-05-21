from gpiozero import LED, Buzzer
import cv2
import numpy as np
import time

RED_LED = LED(17)
YELLOW_LED = LED(27)
GREEN_LED = LED(22)
BUZZER = Buzzer(18)  # Connect buzzer to GPIO pin 18

RED_LED.on()
YELLOW_LED.off()
GREEN_LED.off()

# Load YOLO model
net = cv2.dnn.readNet('yolov3_training_last_2.weights', 'yolov3_testing.cfg')
classes = ["Ambulance"]


colors = np.random.uniform(0, 255, size=(len(classes), 3))


cap = cv2.VideoCapture("4.mp4")


def calculate_distance(width_in_pixels):
   
    real_width_of_ambulance = 2.5  
    focal_length = 700  
   
    
    distance = (real_width_of_ambulance * focal_length) / width_in_pixels
    return distance

try:
    while True:
        ret, img = cap.read()
        if not ret:
            print("Video ended or no frame captured.")
            break

        
        img = cv2.resize(img, (640, 480))  
        height, width, channels = img.shape

       
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        
        class_ids = []
        confidences = []
        boxes = []
        ambulance_detected = False
        distance_to_ambulance = 0  
       
        for out in layerOutputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.7:  
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                   
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    
                    if class_ids[-1] == 0:  
                        ambulance_detected = True
                        
                        distance_to_ambulance = calculate_distance(w)

       
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

       
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

       
        if ambulance_detected:
            print(f"Ambulance detected! Range: {distance_to_ambulance:.2f} meters. Switching traffic light to GREEN.")
            RED_LED.off()
            YELLOW_LED.off()
            GREEN_LED.on()
            BUZZER.on()  
            time.sleep(2)  
            BUZZER.off()  # Deactivate buzzer

            
            cv2.putText(img, f"Ambulance Detected! Range: {distance_to_ambulance:.2f} meters",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            print("No ambulance detected. Traffic light is RED.")
            RED_LED.on()
            YELLOW_LED.off()
            GREEN_LED.off()

        
        cv2.imshow("Ambulance Detection", img)

        
        ambulance_detected = False

       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program interrupted by user.")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()