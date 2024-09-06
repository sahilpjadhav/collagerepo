import cv2
import myMin as htm
import numpy as np
import autopy
import time

###################################
wCam, hCam = 640, 480
frameR = 100   # Frame reduction
smoothening = 10
right_click_threshold = 20
drag_threshold = 25
###################################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
dragging = False

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index, middle, and thumb fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:] # Middle finger tip
        x3, y3 = lmList[4][1:]  # Thumb tip
        # print(x1, y1, x2, y2, x3, y3)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:

            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Both Index and Middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between index and middle fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)
            # 10. Click mouse if distance is short
            if length < 30:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

        # 11. Pinky and Index finger are up : Right-Clicking Mode
        if fingers[4] == 1 and fingers[1] == 1 and fingers[2] == 0:
            autopy.mouse.click(autopy.mouse.Button.RIGHT)

        # 12. Thumb and Index fingers are up then check the distance and then perform drag and drop
        if fingers[0] == 1 and fingers[1] == 1:
            # 12. Find distance between thumb and index fingers
            length, img, lineInfo = detector.findDistance(4, 8, img)
            # print(length)
            # 13. If distance is short, initiate or continue dragging
            if length < drag_threshold:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 0, 255), cv2.FILLED)
                if not dragging:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, down=True)
                    dragging = True
            else:
                if dragging:
                    autopy.mouse.toggle(autopy.mouse.Button.LEFT, down=False)
                    dragging = False

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
