import cv2
img = cv2.imread(r"C:\Users\hp\Downloads\images (20).jpg")


det = cv2.CascadeClassifier(r"C:\Users\hp\Downloads\haarcascade_frontalface_default.xml")
faces = det.detectMultiScale(img)
print(faces.shape)

for face in faces:
    x,y,w,h = face
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.imshow("output",img)
cv2.waitKey(6000)
cv2.destroyAllWindows()