import datetime
import time
from collections import deque
from copy import deepcopy

import cv2
import numpy as np

class_path = "./classes.txt"
def list_classes():
    with open(class_path, "r") as fp:
        datas = fp.readlines()
    classes = []
    for cls in datas:
        classes.append(cls.strip('\n'))
        
    return classes

CLASSES = list_classes()
CLASSES.insert(0, "nothing")

class BboxAnnotator(object):
    """
    Annotates bounding boxes on an image.

    Args:
        annotations (list): List of existing annotations.
        img_id (int): ID of the image.
        img_path (str): Path to the image file.
        window_name (str, optional): Name of the window to display the image. Defaults to "Image".
        fps (int, optional): Frames per second for displaying the image. Defaults to 20.
        is_start (bool, optional): Flag indicating if it is the start of the annotation. Defaults to True.
    """

    def __init__(
        self, annotations, img_id, img_path, window_name="Image", fps=20, is_start=True
    ) -> None:
        self.started_at = time.time()
        # print(f'annotations: {annotations}')
        # self.img_path = img_path
        # print(f'img_path : {img_path}')
        self.img = cv2.imread(img_path)
        self.img_id = img_id
        self.cls_ids = []
        self.annotations = deepcopy(annotations)
        # print(f'self.annotations: {self.annotations}')
        self.starts = []
        self.stops = []
        for annotation in self.annotations:
            if annotation["bbox"] == []:
                # self.starts.append(annotation["bbox"][:2])
                # self.stops.append(
                #     [
                #         annotation["bbox"][0] + annotation["bbox"][2],
                #         annotation["bbox"][1] + annotation["bbox"][3],
                #     ]
                # )
                self.cls_ids.append(0)
                pass
            else:
                for bbox, cls_id in zip(annotation["bbox"], annotation["category_id"]):
                    self.starts.append(bbox[:2])
                    self.stops.append(
                        [
                            bbox[0] + bbox[2],
                            bbox[1] + bbox[3],
                        ]
                    )
                    self.cls_ids.append(cls_id)

        self.is_start = is_start
        self.current_keypoint = None
        self.dragging = False
        self.window_name = window_name
        self.pressed_at = 0
        self.fps = fps

        self.adjust_bbox = None
        self.resize_corner = None
        self.drag_offset = None


        self.distance_threshold = (self.img.shape[0] + self.img.shape[1]) / 2 * 0.025

        self.memory = deque(maxlen=100)
        self.memory.append(deepcopy((self.starts, self.stops)))

        self.show()

    def mouse_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.set_current_keypoint(x, y)
            self.dragging = True
            self.pressed_at = time.time()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.current_keypoint is not None:
                self.memory.append(deepcopy((self.starts, self.stops)))
            self.current_keypoint = None
            self.dragging = False
            self.pressed_at = 0
            self.adjust_bbox = None
            self.resize_corner = None
            self.drag_offset = None
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.current_keypoint is not None:
                i = self.current_keypoint
                if self.adjust_bbox == "adjust":
                    x1, y1 = self.starts[i]
                    x2, y2 = self.stops[i] # These x1,y1 and x2, y2 are the co-ordinates of current bounding box.
                    corners = [
                        [x1, y1],  # 0: top-left
                        [x2, y1],  # 1: top-right
                        [x1, y2],  # 2: bottom-left
                        [x2, y2],  # 3: bottom-right ----> For understaning.
                    ]
                    if self.resize_corner == 0:
                        self.starts[i] = [x, y] # If bbox is resizing from top left corner then the new [x,y] must be the current position of mouse.
                    elif self.resize_corner == 1:
                        self.starts[i] = [x1, y]
                        self.stops[i] = [x, y2]   # If bbox is resizing from top right corner then the 'x-coordinate' of top left(starts) and 'y-coordinate' bottom right(stops) of bbox must be same and  other co-ordinate must be changed as the mouse cursor is moved.
                    elif self.resize_corner == 2:
                        self.starts[i] = [x, y1]
                        self.stops[i] = [x2, y]  # If bbox is resizing from bottom left corner then the 'y-coordinate' of top left(starts) and 'x-coordinate' bottom right(stops) of bbox must be same and  other co-ordinate must be changed as the mouse cursor is moved.
                    elif self.resize_corner == 3:
                        self.stops[i] = [x, y] # If bbox is resizing from bottom right corner then the new [x, y] must be the current position of mouse.
                elif self.adjust_bbox == "move":
                    x_offset, y_offset = self.drag_offset
                    x1 = x - x_offset
                    y1 = y - y_offset # This offset is calculated in relative to top left corner in set_current_keypoint fucntion, so again it used to calculate the current top left corner.
                    width = self.stops[i][0] - self.starts[i][0]
                    height = self.stops[i][1] - self.starts[i][1]
                    self.starts[i] = [x1, y1]
                    self.stops[i] = [x1 + width, y1 + height] #bottom right corner is obtained by adding width and height of bbox to top left corner point.

                if time.time() - self.pressed_at > 1 / self.fps:
                    self.pressed_at = time.time()
                    self.show()

    def key_pressed(self, k):
        # Delete the current bbox
        total_cls = str(len(CLASSES))
        if k in range(ord('0'), ord(total_cls)+1):
            new_cls_idx = k - ord('0')
            if self.current_keypoint is not None:
                self.cls_ids[self.current_keypoint] = new_cls_idx
                print(f"Class changed to {CLASSES[new_cls_idx]} for bbox {self.current_keypoint}")
                self.show()
                
        elif k == ord("d"):
            if self.dragging and self.current_keypoint is not None:
                del self.starts[self.current_keypoint]
                del self.stops[self.current_keypoint]

                self.current_keypoint = None
                self.dragging = False
                self.pressed_at = 0
                self.current_is_start = None
                self.show()

        # Undo the last action
        elif k == ord("z"):
            self.undo()

        # Restart the annotation
        elif k == ord("r"):
            self.undo(all=True)
    
    def set_current_keypoint(self, x, y):
        draw_bbox = False # This variable is initially False because in new frame we have to draw the bounding box from scratch.

        for i, (start, stop) in enumerate(zip(self.starts, self.stops)):
            x1, y1 = min(start[0], stop[0]), min(start[1], stop[1])
            x2, y2 = max(start[0], stop[0]), max(start[1], stop[1]) #This done because top left corner must be smaller than bottom right corner.

            corners = [
                (x1, y1),  # top-left
                (x2, y1),  # top-right
                (x1, y2),  # bottom-left
                (x2, y2),  # bottom-right
            ]
            
            for corner_id, corner in enumerate(corners):
                dst = np.linalg.norm(np.array([x, y]) - np.array(corner))
                if dst < self.distance_threshold:
                    self.current_keypoint = i
                    self.adjust_bbox = "adjust"
                    self.resize_corner = corner_id
                    draw_bbox = True
                    break

            if not draw_bbox:
                if x1 <= x <= x2 and y1 <= y <= y2: #This comparison is to check that if mouse is clicked inside of drawn bbox or not?
                    self.current_keypoint = i
                    self.adjust_bbox = "move"
                    self.drag_offset = [x - x1, y - y1] # This offeset is relative to top left corner of bounding box.
                    draw_bbox = True
            if draw_bbox:
                break

        if not draw_bbox:
            # Create new box
            self.starts.append([x, y])
            self.stops.append([x, y])
            self.current_keypoint = len(self.starts) - 1
            self.adjust_bbox = "adjust"
            self.resize_corner = 3  # bottom-right by default
            self.cls_ids.append(0)


    def show(self):
        img = show_bboxes(
            self.img,
            starts=self.starts,
            stops=self.stops,
            cls_idxs=self.cls_ids
        )

        # Put the text on the image
        text = "{}".format(self.img_id)

        text_color = (0, 0, 255) if self.is_start else (0, 255, 0)
        img = cv2.putText(
            img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 1, cv2.LINE_AA
        )

        cv2.imshow(self.window_name, img)
        
    def undo(self, all=False):
        if len(self.memory) > 0:
            if all:
                self.starts, self.stops = self.memory[0]
                self.memory = deque(maxlen=100)
                self.memory.append(deepcopy((self.starts, self.stops)))
            else:
                self.starts, self.stops = self.memory.pop()

            self.show()

            if len(self.memory) == 0:
                self.memory.append(deepcopy((self.starts, self.stops)))

    def get_annotation(self):
        # Transfer corners to annotations
        # print(f'annotatins: {self.annotations}')
        for start, stop, cls_id in zip(self.starts, self.stops, self.cls_ids):
            start = list(start)
            stop = list(stop)
            for i in range(2):
                if start[i] > stop[i]:
                    tmp = start[i]
                    start[i] = stop[i]
                    stop[i] = tmp

            start[0] = max(0, start[0])
            start[1] = max(0, start[1])
            stop[0] = min(stop[0], self.img.shape[1])
            stop[1] = min(stop[1], self.img.shape[0])
            bbox_width = stop[0] - start[0]
            bbox_height = stop[1] - start[1]
            if (bbox_width > 1) and (bbox_height > 1):
                boxes = [start[0], start[1], stop[0] - start[0], stop[1] - start[1]]
                if boxes not in self.annotations[0]["bbox"]:
                    self.annotations[0]["bbox"].append(boxes)
                    self.annotations[0]["category_id"].append(cls_id)
                # if time.time() - self.started_at > 3:
                #     self.annotations[-1]["checked"] = datetime.datetime.now().strftime(
                #         "%Y-%m-%d_%H:%M:%S"
                #     )
            else:
                print(
                    f"Invalid bbox: {start}, {stop}, bbox_width: {bbox_width}, bbox_height: {bbox_height}"
                )
                print(f"Img_id: {self.img_id}")


        annotations = deepcopy(self.annotations)
        # for annotation in annotations:
        #     annotation["bbox"] = annotation["bbox"].flatten().tolist()
        return annotations

def show_bboxes(img, starts, stops, cls_idxs):
    if isinstance(img, str):
        print("Loading image")
        img = cv2.imread(img)
    else:
        img = img.copy()

    for start, stop, cls_idx in zip(starts, stops, cls_idxs):
        img = cv2.rectangle(
            img,
            (int(start[0]), int(start[1])),
            (int(stop[0]), int(stop[1])),
            color=(0, 255, 0),
            thickness=2,
        )
        img = cv2.circle(
            img,
            (int(start[0]), int(start[1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )
        img = cv2.circle(
            img,
            (int(stop[0]), int(stop[1])),
            radius=5,
            color=(0, 0, 255),
            thickness=-1,
        )
        img = cv2.circle(
            img,
            (int(stop[0]), int(start[1])),
            radius=5,
            color=(255, 0, 0),
            thickness=-1,
        )
        img = cv2.circle(
            img,
            (int(start[0]), int(stop[1])),
            radius=5,
            color=(255, 0, 0),
            thickness=-1,
        )
        img = cv2.putText(img, CLASSES[cls_idx], (int((start[0]+50)), start[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
    return img

if __name__ == "__main__":
    # Test the annotator
    img_path = "sample_data/data/val2025/mohan.jpeg"
    annotations = []
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    ia = BboxAnnotator(annotations, 0, img_path)
    cv2.setMouseCallback("Image", ia.mouse_callback)
    while True:
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        else:
            ia.key_pressed(k)
