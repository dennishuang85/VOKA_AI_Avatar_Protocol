import os
from PIL import Image
import matplotlib.pyplot as plt
import threading
import cv2, numpy, tqdm
from detect_face import create_detector
from tqdm.contrib.concurrent import process_map
from queue import Queue
import json

queue_lock = threading.Lock()

class MyThread(threading.Thread):
    def __init__(self, work_queue):
        threading.Thread.__init__(self)
        self.work_queue = work_queue
        self.detector = create_detector('yolov3')
    def run(self):
        while True:
            queue_lock.acquire()
            if self.work_queue.empty():
                queue_lock.release()
                break
            else:
                root, name = self.work_queue.get()
                if work_queue.qsize() % 100 == 0:
                    print(work_queue.qsize())
                queue_lock.release()
            _detect_face(root, name, self.detector)
            
def _detect_face(root, name, detector):
    try:
        img = Image.open(os.path.join(root, name))
        img = img.convert('RGB')
        cv_img = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
        
        img_b = numpy.zeros(cv_img.shape, cv_img.dtype)
        label_path = os.path.join(root, name)[:-4] + '.json'
        with open(label_path, 'r') as label_file:
            labels = json.load(label_file)
        
        for shape in labels["shapes"]:
            if shape["label"] == "face":
                print(shape["points"])
                cv2.fillPoly(img_b, [numpy.array(shape["points"], dtype=numpy.int32)], (255, 0, 0))
        for shape in labels["shapes"]:
            if shape["label"] == "eye":
                cv2.fillPoly(img_b, [numpy.array(shape["points"], dtype=numpy.int32)], (0, 255, 0))
        for shape in labels["shapes"]:
            if shape["label"] == "eyeball":
                cv2.fillPoly(img_b, [numpy.array(shape["points"], dtype=numpy.int32)], (0, 0, 255))
        img_b = Image.fromarray(cv2.cvtColor(img_b,cv2.COLOR_BGR2RGB))
        preds = detector(cv_img)
        
        for i, bbox in enumerate(preds):
            
            bbox = bbox['bbox']
            
            width, height = img.size

            center = (int((bbox[0] + bbox[2]) / 2), int((2 * bbox[1] + bbox[3]) / 3))
            length = min([center[0],
                        width - center[0],
                        center[1],
                        height - center[1],
                        int(center[0] - bbox[0]) * 3,
                        int(bbox[2] - center[0]) * 3,
                        int(center[1] - bbox[1]) * 6,
                        int(bbox[3] - center[1]) * 3
            ])
            box = (center[0] - length,
                center[1] - length,
                center[0] + length,
                center[1] + length,
                )
            new_img = img.crop(box).resize((512, 512))
            if length >= 80 and bbox[4] > 0.9:
                from random import randint
                if randint(0, 10) == 0:
                    group_id = "test"
                else:
                    group_id = "train"
                
                new_img.save("face_seg" + group_id + "A\\" 
                        + name.split('.')[0] + '_' + str(i) + '.png'
                        )
                new_img_b = img_b.crop(box).resize((512, 512))
                new_img_b.save("face_seg" + group_id + "B\\" 
                        + name.split('.')[0] + '_' + str(i) + '.png'
                        )
            
            
            
    except Exception as e:
        raise e

work_queue = Queue()

for root, dirs, files in os.walk("photo1", topdown=False):
    for name in files:
        if name.endswith(".png") or name.endswith(".jpg"):
            work_queue.put((root, name))

threads = []
for i in range(8):
    thread = MyThread(work_queue)
    thread.start()
    threads.append(thread)

for thread in threads:
    thread.join()