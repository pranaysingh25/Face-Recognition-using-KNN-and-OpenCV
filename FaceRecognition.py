import numpy as np
import cv2

camera = cv2.VideoCapture(0)
#opencv classifier-- haarCascade  object for extracting features automatically 
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#declare the type of font to be used on the output window
font = cv2.FONT_HERSHEY_SIMPLEX

#loading our numpy object files
pranay = np.load('pranay.npy').reshape((20, 50*50*3))  #we will now have it reshaped into 20 flattened linear vectors. 
rahul = np.load('Rahul.npy').reshape((20, 50*50*3))
rohit = np.load('rohit.npy').reshape((20, 50*50*3))
# print(vishal.shape)
# print(pranay.shape)
#shape will be (20, 7500) where 7500 is pixels we had in every layer of brg total.. now they are features 
#so 20 instances of my face with 7500 features each

#create a look up dictionary for our labels
names = {
    0: 'Pranay',
    1: 'Rahul',
    2: 'Rohit',
}

#creating y_train manually
labels = np.zeros((60,1))
labels[:20, :] = 0.0  #1st 30 for me
labels[20:40, :] = 1.0   #rest all for others
labels[40:, :] = 2.0 
#combine all the face data into 1 array
data = np.concatenate([pranay, rahul, rohit])
print(data.shape)
print(labels.shape)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

while True:
    ret, frame = camera.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #apply haar cascade to detect faces in current frame, it return face objects from the images as cords
        faces = face_cas.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            #extracting just faces from frame using cordinates got (along with all the 3 bgr layer)
            face_component = frame[y:y+h, x:x+w, :]
            #resize tha face component from image to 50 * 50 * 3
            fc = cv2.resize(face_component, (50,50))
            
            #now instead of storing of data this time we will try to recognize them
            #after processing the image and rescaling convert to linear vector using .flatten() 
            #and pass to knn fun along with all the training data and training labels
            #to get the prediction label
            
#             lab = knn(fc.flatten(), data, labels)
            knn.fit(data, np.ravel(labels))
            y_predict = knn.predict(fc.flatten().reshape(1, -1))
            
            #now convert that label into name text using our name dictionary
            text = names[int(y_predict)]
            cv2.putText(frame, text, (x,y), font, 1, (255,255,0), 2)
            
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),2)
        cv2.imshow('face recognition', frame)
        
        if cv2.waitKey(1) == 27:
            break
    else:
        #camera is not working then error msg
        print('Camera not working')
        
cv2.destroyAllWindows()