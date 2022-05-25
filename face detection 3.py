import cv2
img = cv2.imread(r"C:\Users\hp\Downloads\images (19).jpg")
"""
cv2.imshow("output",img)
cv2.waitKey(6000)
cv2.destroyAllWindows()
"""


# lets load the detector module
detector = cv2.CascadeClassifier(r"C:\Users\hp\Downloads\haarcascade_frontalface_default.xml")
faces = detector.detectMultiScale(img,1.03,3)
print(faces.shape)

# now lets draw a rectangle on the faces
for face in faces:
    x,y,w,h=face
    img = cv2.rectangle(img,(x,y),(x+y,w+h),(255,0,0),2)
cv2.imshow("output",img)
cv2.waitKey(6000)
cv2.destroyAllWindows()