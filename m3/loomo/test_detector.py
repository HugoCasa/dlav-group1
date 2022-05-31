from detector import Detector
import cv2
import copy
import numpy as np

last_bbox = [0,0, 10, 10, 0.0]

# If true, when person of interest is lost, keep his last position as current one
WITH_INERTIA = True
MIN_X = 400
MAX_X = 800
MARGIN = 0.3


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

def draw_target_bbox(debug_image, target_bbox):
    center_x, center_y, width, height, _ = target_bbox
    half_width = int(width/2)
    half_height = int(height/2)
    pt1 = (center_x - half_width, center_y - half_height)
    pt2 = (center_x + half_width, center_y + half_height)
    cv2.rectangle(debug_image, pt1, pt2, (0, 0, 255), 3)


def calc_center(brect):
    # calculate center of hand bounding box
    return (brect[0] + brect[2]) / 2, (brect[1] + brect[3]) / 2


def draw_debug_info_detector(
    debug_image,
    bboxes,
    scores,
    class_ids,
):
    
    for i in range(len(bboxes)):
        draw_bounding_box(debug_image, bboxes[i], f'ID: {class_ids[i]} with score: {round(scores[i],2)}' )

def get_id_color(index, target_idx):
    # Green if target id with ow
    return (0,255,0) if (index == target_idx) else (255, 255,255)


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


detector = Detector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # target_bbox, label = detector.forward(frame,  is_re_init_allowed=True)
    
    debug_image = copy.deepcopy(frame)

    # if len(target_bbox) > 0:
    #     draw_target_bbox(debug_image, target_bbox)


    # Empty bbox if person of interest not detected
    bbox, bbox_label = detector.forward(frame, is_re_init_allowed=True)
    
    # if bbox_label:
    #     print("BBOX: {}".format(bbox))
    #     print("BBOX_label: {}".format(bbox_label))
    # else:
    #     print("False")
    # print(bbox)

    if len(bbox) == 0:
        if WITH_INERTIA:
            # Robot follow last bbox (turn in circle probably)
            values = last_bbox
        else:
            # 0.0 confidence => Robot does not move
            values = [0, 0, 10, 10, 0.0]
    else:
        conf = 1.0
        width = frame.shape[1]
        min_x = int(MARGIN * width)
        max_x = int(width - MARGIN * width)
        if bbox[0] > min_x and bbox[0] < max_x:
            conf = 0.0
        last_bbox = [bbox[0], bbox[1], bbox[2], bbox[3], conf]
        print("Last_bbox", last_bbox)
        
        # According to the TA w, h = 10 is more robust for depth estimation
        values = [bbox[0], bbox[1], bbox[2], bbox[3], 1.0]
        print("Values", values)


    width = frame.shape[1]
    min_x = int(MARGIN * width)
    max_x = int(width - MARGIN * width)
    values[0] = max(min(values[0], max_x), min_x)

    draw_target_bbox(debug_image, values)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    cv2.imshow('test_detector', debug_image)

cap.release()
cv2.destroyAllWindows()

