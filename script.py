import cv2 
import numpy as np

#Inferencing module
def detect_age(frame,faceNet,ageNet):
    #Frame of detection and age 
    results=[]
    
    #Get the dimensions the frame captured
    (h,w)=frame.shape[:2]
    
    #Creating a blob to feed into the inference model 
    blob=cv2.dnn.blobFromImage(frame, 1.5 , (300,300), (104.0, 177.0, 123.0))
    
    #Pass through faceNet to obtain face detection window
    faceNet.setInput(blob)
    #Frames detected for faces
    detections = faceNet.forward()
    
    no_of_points_detected=detections.shape[2]
    
    #Now for each set of points we will measure confidence scores 
    #Then for best window we will inference age
    for i in range(no_of_points_detected):
        confidence_score=detections[0,0,i,2]
        
        #Filtering out the weak confidence
        if confidence_score>0.80:
            # compute the (x, y)-coordinates of the bounding box for
			# the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # extract the ROI of the face
            face = frame[startY:endY, startX:endX]

			# ensure the face ROI is sufficiently large
            if face.shape[0] < 50 or face.shape[1] < 50:
                continue
                
            #Creating blob for ageNet
            faceBlob = cv2.dnn.blobFromImage(face, 1.5, (227, 227),(78.4263377603, 87.7689143744, 114.895847746),swapRB=False)    
             
            # Get all the predictions probability on age bucket
            ageNet.setInput(faceBlob)
            preds = ageNet.forward()
           
            #Select largest probability label
            i = preds[0].argmax()
            age = bucket[i]
            ageConfidence = preds[0][i]

			# construct a dictionary consisting of both the face
			# bounding box location along with the age prediction,
			# then update our results list
            d = {
                "loc": (startX, startY, endX, endY),
                "age": (age, ageConfidence)
			}
            
            results.append(d)
    
    #Return to calling function
    return results

#Our caffe model will predict labels in following bucketized ranges
bucket = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)","(38-43)", "(48-53)", "(60-100)"]

#Load the face detector model and weights
model='face_detector/deploy.prototxt'
weights='face_detector/res10_300x300_ssd_iter_140000.caffemodel'

faceNet=cv2.dnn.readNet(model,weights)

#Load the age detector model and weights 
model='age_detector/age_deploy.prototxt'
weights='age_detector/age_net.caffemodel'

ageNet=cv2.dnn.readNet(model,weights)

#Cv2 object for recording footage via second internal webcam 
#cam=cv2.videoCapture(0)  (I will use external as internal webcam has low resolution)

#Cv2 object for recording footage via second external webcam 
cam=cv2.VideoCapture(1)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(True):
    read,frame=cam.read()
    
    #Call the inference model 
    detections = detect_age(frame,faceNet,ageNet)
    
    # loop over the results and display
    for r in detections:
		# draw the bounding box of the face along with the associated
		# predicted age
        text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
        #Creating rectangle window
        (startX, startY, endX, endY) = r["loc"]
		
        y = startY - 10 if startY - 10 > 10 else startY + 10
		
        #Displaying on frame
        cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    out.write(frame)
    cv2.imshow('camera',frame)
    if  cv2.waitKey(1) &0xFF == ord('q'):
        break
        
#Releasing all resources
cam.release()
cv2.destroyAllWindows()