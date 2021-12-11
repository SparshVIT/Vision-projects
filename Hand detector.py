import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

#for detection of hand the class and object will be formed from the hand module
mpHands  = mp.solutions.hands
hands = mpHands.Hands()#the object declaration
#for drawing the points on the hand image
mpDraw = mp.solutions.drawing_utils

#for the frame rate
pTime = 0
cTime = 0
# for opening the camera

while True:
    success, img = cap.read()
    #As this hand module works on the rgb image hence the image has to be converted to rgb image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks: #here it will be checked that if hands are detected
        for handlms in results.multi_hand_landmarks:# for multiple hands
            
            #now we will see the x and y coordinate of the position points of hand
            #for that we will see the id of every position of point on hand and the coordinates corresponding to it
            for id,lm in enumerate(handlms.landmark):
               # print(id,lm)
                h, w, c = img.shape #dimension of image
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)#landamrk of every point
                #now showing on the display
                if id==0:
                    cv2.circle(img, (cx,cy), 25,(255, 1, 255), cv2.FILLED)
                
            
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)#mpHands.HAND_CONNECTIONS is used to connect all the points
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255), 3)#display the frame rate
    
        

    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
