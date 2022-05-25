import cv2
img = cv2.imread(r"C:\Users\hp\Downloads\images (18).jpg")

# now lets load the ml model named as haarcascade frontalface default
detector = cv2.CascadeClassifier(r"C:\Users\hp\Downloads\haarcascade_frontalface_default.xml")
face = detector.detectMultiScale(img)

# lets see that how many faces the detector has been detecting
print(face.shape)
