import cv2
import numpy as np

#camera object initialized
camera = cv2.VideoCapture(0)
#opencv classifier-- haarCascade  object for extracting features automatically 
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print('Hey buddy! mind entering your name please ?')
personName = input()

#this will hold our training data
face_data = []
ix = 0  #current frame no

while True:
    ret, frame = camera.read()
    if ret == True:
        #to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #apply haar cascade to detect faces in current frame, it return face objects from the images as cords
        print(gray)
        faces = face_cas.detectMultiScale(gray, 1.3, 5)
        #for each face object we get, we have
        #the corner cordinates(x,y) and the width &height of the face in it
        
        for(x,y,w,h) in faces:
            #extracting just faces from frame using cordinates got (along with all the 3 bgr layer)
            face_component = frame[y:y+h, x:x+w, :]
            #resize tha face component from image to 50 * 50 * 3
            fc = cv2.resize(face_component, (50,50))
            #store the face data after every 10 frames only if the no. of entries is less than 20-- loop will go on until 20
            if ix%10 == 0 and len(face_data) < 20:
                face_data.append(fc)
                
            #for viz, drawing a rec around face using rectangle function
            # args - frame, (start pt), (end pt), color of rec, thickness
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        ix +=1 #increment the frame no, frame no. are just used for gaps between saving each frame that is 10.
        cv2.imshow('frame', frame)
        #if i press escape or face data completes 20 faces it'll stop recording
        if cv2.waitKey(1) == 27 or len(face_data) >= 20:
            break
    else:
        #camera is not working then error msg
        print('Camera not working')

#destroy all windows we created        
cv2.destroyAllWindows()
#convert the data into numpydata = np.asarray(face_data)
data = np.asarray(face_data)
print(data.shape)
#this saves an array to a binary file in NumPy .npy format.
np.save(personName, data)