# Gender and Age Recognition

import cv2
from fer import FER

age_protocol_buffer = "age_deploy.prototxt"
age_model = "age_net.caffemodel"
face_protocol_buffer = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
gender_protocol_buffer = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"
padding = 20
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_range = ['(0-2)',
             '(4-6)',
             '(8-12)',
             '(15-20)',
             '(25-32)',
             '(38-43)',
             '(48-53)',
             '(60-100)']
gender_list = ['Male', 'Female']


'''
STEPS INVOLVED

1. Detect face
2. Classify into Male/Female
3. Classify into one of the (0–2), (4–6), (8–12), (15–20), (25–32), (38–43), (48–53), (60–100) age ranges
4. Displaying the Result
'''

# Getting path of image from the user
path_to_image = str(input("Enter Path to Image: "))

# Loading the networks
age_network = cv2.dnn.readNet(age_model, age_protocol_buffer)
face_network = cv2.dnn.readNet(face_model, face_protocol_buffer)
gender_network = cv2.dnn.readNet(gender_model, gender_protocol_buffer)

# Reading the image
frame = cv2.imread(path_to_image)

confidence_threshold = 0.7
image = frame.copy()
frame_height = image.shape[0]
frame_width = image.shape[1]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [
    104, 117, 123], True, False)

face_network.setInput(blob)
detection = face_network.forward()
faces = []
for i in range(detection.shape[2]):
    confidence = detection[0, 0, i, 2]
    if confidence > confidence_threshold:
        x1 = int(detection[0, 0, i, 3]*frame_width)
        y1 = int(detection[0, 0, i, 4]*frame_height)
        x2 = int(detection[0, 0, i, 5]*frame_width)
        y2 = int(detection[0, 0, i, 6]*frame_height)
        faces.append([x1, y1, x2, y2])
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      (0, 255, 0), int(round(frame_height/150)), 8)

if len(faces) < 1:
    print("No face recognised")  # If no face is detected in frame
else:
    # If faces are detected in frame
    for face in faces:
        face_frame = frame[max(0, face[1]-padding):
                           min(face[3]+padding, frame.shape[0]-1), max(0, face[0]-padding):min(face[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(
            face_frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        '''
        Feeding the input and giving the network a forward pass. 
        The higher one is the gender of the person.
        '''
        gender_network.setInput(blob)
        gender_prediction = gender_network.forward()
        gender_prediction = gender_prediction.tolist()
        gender = gender_list[gender_prediction[0].index(
            max(gender_prediction[0]))]
        
        '''
        Feeding the input and giving the network a forward pass. 
        This is kind of the same thing that got us the gender, We do it again for age.
        '''
        age_network.setInput(blob)
        age_prediction = age_network.forward()
        age_prediction = age_prediction.tolist()
        age = age_range[age_prediction[0].index(max(age_prediction[0]))]
        detector = FER(mtcnn=True)
        emotion, score = detector.top_emotion(frame)
        print("\n\n\n\n\n\n")
        print('Gender of the person in image: {}'.format(gender))
        print('Age Range of the person in image: {}'.format(age[1:-1]))
        print('Emotion of the person : {}'.format(emotion))