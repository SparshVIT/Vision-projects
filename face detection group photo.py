import cv2
img = cv2.imread(r"C:\Users\hp\Downloads\istockphoto-1146473249-612x612.jpg")
print(img.shape)
"""
cv2.imshow("output",img)
cv2.waitKey(6000)
cv2.destroyAllWindows()
"""

# as we have downloaed the model hence we have to insert into
detector = cv2.CascadeClassifier(r"C:\Users\hp\Downloads\haarcascade_frontalface_default.xml")

# as the detector is imported now it has a function named detectMultiscale which will detect the faces in the image
faces = detector.detectMultiScale(img,1.3,4) # here 1.3 is the scale factor which determine image size reduction
#whereas 4 is the min neighbour that rectangle be only when it has 4 neighbour
print(faces.shape) # to look that how many faces it has detected

# so it will show the faces in the order (x,y) where x represents number of faces and y shows parameters
#x,y,w,h = faces[0]
#here now it will form the rectangle using
"""
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0))
cv2.imshow("output",img)
cv2.waitKey(6000)
cv2.destroyAllWindows()
"""

#now if we want all rectangles drawn on image then we use for loop
for face in faces:
    x,y,w,h = face
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)

cv2.imshow("output",img)
cv2.waitKey(6000)
cv2.destroyAllWindows()
