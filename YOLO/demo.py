"""Demo for use yolov3
"""
import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO


def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_classes(file_name):
    """Get classes name.

    # Argument:
        file_name: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file_name) as fp:
        class_names = fp.readlines()
    class_names = [c.strip() for c in class_names]  # remove '¥n' from strings

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        left = max(0, np.floor(x + 0.5).astype(int))
        top = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # get the width and height of the text box
        # (text_width, text_height) = cv2.getTextSize('{0} {1:.2f}'.format(all_classes[cl], score), font, fontScale=1.0, thickness=1)[0]
        # if top < text_height+3:
        #     top = top
        # else:
        #     top = top - 6
        cv2.putText(image,
                    '{0} {1:.2f}'.format(all_classes[cl], score),
                    (left+3, top+17),
                    font,
                    0.6, (0, 255, 0), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))


def detect_image(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('Processing time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image


def detect_video(video, yolo, all_classes):
    """Use yolo v3 to detect video.

    # Argument:
        video: video file.
        yolo: YOLO, yolo model.
        all_classes: all classes name.
    """
    video_path = os.path.join("videos", "test", video)
    camera = cv2.VideoCapture(video_path)
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

    # Prepare for saving the detected video
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')

    vout = cv2.VideoWriter()
    vout.open(os.path.join("videos", "res", video), fourcc, 20, sz, True)

    while True:
        res, frame = camera.read()

        if not res:
            break

        image = detect_image(frame, yolo, all_classes)
        cv2.imshow("detection", image)

        # Save the video frame by frame
        vout.write(image)

        if cv2.waitKey(110) & 0xff == 27:
            break

    vout.release()
    camera.release()
    

if __name__ == '__main__':
    yolo = YOLO(0.6, 0.5)
    file = 'data/coco_classes.txt'
    all_classes = get_classes(file)

    # detect images in test folder.
    for (root, dirs, files) in os.walk('images/test'):
        if files:
            for f in files:
                if f.endswith('.jpg') or f.endswith('.png'):
                    print("Image to be detected:", f)
                    path = os.path.join(root, f)
                    image = cv2.imread(path)
                    image = detect_image(image, yolo, all_classes)
                    cv2.imwrite('images/res/' + f, image)
                else:
                    pass
        else:
            print("No files found!")

    # detect videos one at a time in videos/test folder
    # video = 'library1.mp4'
    # detect_video(video, yolo, all_classes)
