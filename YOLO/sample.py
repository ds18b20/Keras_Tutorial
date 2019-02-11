"""Demo for use yolo v3
"""
import cv2

if __name__ == '__main__':
    img = cv2.imread("sample_w.png")
    anchors_13 = [(10, 13),
                  (16, 30),
                  (33, 23)]

    anchors_26 = [(30, 61),
                  (62, 45),
                  (59, 119)]

    anchors_52 = [(116, 90),
                  (156, 198),
                  (373, 326)]
    line = 1
    for anchor in anchors_13:
        cv2.rectangle(img, (0, 0), anchor, (255, 0, 0), line)
        line += 1
    line = 1
    for anchor in anchors_26:
        cv2.rectangle(img, (0, 0), anchor, (0, 255, 0), line)
        line += 1
    line = 1
    for anchor in anchors_52:
        cv2.rectangle(img, (0, 0), anchor, (0, 0, 255), line)
        line += 1

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
