import cv2


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (160,120))

while True:

    ret, frame = cap.read()
    if not ret:
        break
    

    frame = cv2.resize(frame, (160, 120))
    # write the flipped frame
    out.write(frame)

    cv2.imshow('frame', frame) 
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()

out.release()
cv2.destroyAllWindows()
