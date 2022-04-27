from cmath import nan
import copy
import time
import argparse
import cv2

from Detector.detector import ObjectDetector
from Tracker.tracker import MultiObjectTracker


def main():
    CAP_DEVICE = 0
    USE_GPU = False
    CAP_FPS = 20
    TARGET_GESTURE_ID = 2


    cap = cv2.VideoCapture(CAP_DEVICE)
    # cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # Hand Gesture Detection
    gesture_detector = ObjectDetector(
        name= "hand_gesture",
        target_id= None,
        use_gpu=USE_GPU
    )
    gesture_detector.print_info()

    # People Detection
    people_detector = ObjectDetector(
        name= "yolox",
        target_id = None,
        use_gpu=USE_GPU,
    )
    people_detector.print_info()

    # Multi Object Tracking
    tracker = MultiObjectTracker(
        CAP_FPS,
        use_gpu=USE_GPU,
    )
    tracker.print_info()

    target_id = None
    target_bbox = []

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Flip image for selfie mod
        # TODO take in into account for robot direction
        frame = cv2.flip(frame, 1)
        debug_image = copy.deepcopy(frame)

        # Hand Gesture Detection
        hg_bboxes, hg_scores, hg_class_ids = gesture_detector(frame)

        # Draw gesture detection
        draw_debug_info_detector(debug_image, hg_bboxes, hg_scores, hg_class_ids)

        # Person  Detection
        d_bboxes, d_scores, d_class_ids = people_detector(frame)

        # Multi People Tracking
        track_ids, t_bboxes, t_scores, t_class_ids = tracker(
            frame,
            d_bboxes,
            d_scores,
            d_class_ids,
        )

        # If at least two target gesture detected
        if  hg_class_ids.count(TARGET_GESTURE_ID) > 1:
            
            # Compute center coords of target gestures
            hand_centers = []
            for hg_box, hg_id in zip(hg_bboxes, hg_class_ids):
                if hg_id == TARGET_GESTURE_ID:
                    hand_centers.append(calc_center(hg_box))

            # For each detected people count number of detected gesture in their box  
            for t_box, t_id in zip(t_bboxes, track_ids):
                count_gesture_in_box = 0
                for hand_center in hand_centers:
                  if in_bounding_box(hand_center, t_box):
                      count_gesture_in_box+=1
                # If more than one, set it as target people
                if count_gesture_in_box > 1:
                  target_id = t_id

        # Retrieve the current target bbox
        if target_id != None:
            has_been_detected = False
            for id, box in zip(track_ids, t_bboxes):
                if id == target_id:
                    has_been_detected = True
                    target_bbox = box
                    break
            if not has_been_detected:
                target_bbox = []

        if len(target_bbox) != 0:
            x, y = calc_center(target_bbox)
            height = target_bbox[3] -  target_bbox[1]
            cv2.circle(debug_image, (round(x), round(y)), round(height / 30),
                    (0, 0, 255), 1)
                
        elapsed_time = time.time() - start_time

        debug_image = draw_debug_info(
            debug_image,
            elapsed_time,
            track_ids,
            t_bboxes,
            t_scores,
            t_class_ids,
            target_id,
        )

    
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        cv2.imshow('MOT Tracking by Detection Pipeline Sample', debug_image)

    cap.release()
    cv2.destroyAllWindows()

def calc_center(brect):
    # calculate center of hand bounding box
    return (brect[0] + brect[2]) / 2, (brect[1] + brect[3]) / 2

def in_bounding_box(p, bbox):
    # check if point is in bounding box
    return p[0] >= bbox[0] and p[0] <= bbox[2] and p[1] >= bbox[1] and p[1] <= bbox[3]

def get_id_color(index, target_idx):
    # Green if target id with ow
    return (0,255,0) if (index == target_idx) else (255, 255,255)
   

def draw_bounding_box(image, brect, hand_sign_text = ""):
    # draw bounding box of hand
    # Outer rectangle
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (0, 0, 0), 3)

    # Text
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)
    cv2.putText(image, hand_sign_text, (brect[0] + 5, brect[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def draw_debug_info_detector(
    debug_image,
    d_bboxes,
    d_scores,
    d_class_ids,
):
    for i in range(len(d_bboxes)):
        draw_bounding_box(debug_image, d_bboxes[i], f'ID: {d_class_ids[i]} with score: {round(d_scores[i],2)}' )



def draw_debug_info(
    debug_image,
    elapsed_time,
    track_ids,
    bboxes,
    scores,
    class_ids,
    target_id
):
    for id, bbox, score, class_id in zip(track_ids, bboxes, scores, class_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        color = get_id_color(id, target_id if target_id != None else -1)

        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )

        score = '%.2f' % score
        text = 'TID:%s(%s)' % (str(int(id)), str(score))
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness=2,
        )

        text = 'CID:%s' % (str(int(class_id)))
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness=2,
        )

    cv2.putText(
        debug_image,
        "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return debug_image


if __name__ == '__main__':
    main()

     