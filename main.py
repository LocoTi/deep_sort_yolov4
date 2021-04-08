import os
import argparse
import warnings
from collections import deque

from yolo.darknet_images import *

from tools import generate_detections
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="",
                        help="image source. It can be a single image, a"
                        "txt with paths to them, or a folder. Image valid"
                        " formats are jpg, jpeg or png."
                        "If no input is given, ")
    parser.add_argument("--output", type=str, default="output",
                        help="path to save annonation txt, or save detected video")
    parser.add_argument("--output_file", type=str, default="./res_data/detection.txt",
                        help="file to save result txt with MOT format")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="number of images to be processed at the same time")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--save_labels", action='store_true',
                        help="save detections bbox for each image in yolo format")
    parser.add_argument("--config_file", default="./yolo/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./yolo/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")

    parser.add_argument("--model_file", type=str, default="model_data/mars-small128.pb",
                        help="the model file to extract feature used in generate_detections.py")
    return parser.parse_args()

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.input and not os.path.exists(args.input):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(150, 255, size=(100, 3), dtype="uint8")

def main():
    min_confidence = 20
    max_cosine_distance = 0.5
    min_detection_height = 0
    nms_max_overlap = 1.0
    max_cosine_distance = 0.2
    nn_budget = 100

    args = parser()
    check_arguments_errors(args)

    encoder = generate_detections.create_box_encoder(args.model_file, batch_size=args.batch_size)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []
    
    # yolov4
    # yolo(args=args)
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )

    images = load_images(args.input)

    writeVideo_flag = True
    video_path = "./output/output.mp4"

    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        # w = int(video_capture.get(3))
        # h = int(video_capture.get(4))
        width, height = get_image_size(network)
        first_image = cv2.imread(images[0])
        org_h, org_w = first_image.shape[:2]
        print(width, height, org_w, org_h)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(video_path, fourcc, 15, (width, height))
        list_file = open(args.output_file, 'w')

    counter = []
    
    fps = 0.0

    index = 0
    while True:
        # loop asking for new image paths if no list is given
        if args.input:
            if index >= len(images):
                break
            image_name = images[index]
        else:
            image_name = input("Enter Image Path: ")
        prev_time = time.time()
        # after darknet detection, bbox = (center_x, center_y, w, h)
        image, detections = image_detection(
            image_name, network, class_names, class_colors, args.thresh
            )
        # if args.save_labels:
        #     save_annotations(image_name, image, detections, class_names, output=args.output)
        
        boxs = []
        predicted_names = []
        for label, confidence, bbox in detections:
            # extract feature, we need bbox = (left, top, w, h)
            center_x, center_y, w, h = bbox
            xmin = int(round(center_x - (w / 2)))
            ymin = int(round(center_y - (h / 2)))
            boxs.append([xmin, ymin, w, h])
            # predict_name = class_names.index(label)
            # predicted_names.append(predict_name)
        features = encoder(image, boxs)

        # 为每个框创建检测器
        box_idx = 0
        dets = []
        for label, confidence, bbox in detections:
            if bbox[3] < min_detection_height:
                continue
            if bbox[2] > 0.8 * width:
                print(label, ", ", confidence, bbox)
            # create Detection, we need bbox = (left, top, w, h)
            center_x, center_y, w, h = bbox
            xmin = int(round(center_x - (w / 2)))
            ymin = int(round(center_y - (h / 2)))
            dets.append(Detection([xmin, ymin, w, h], confidence, features[box_idx]))
            box_idx += 1
        
        # 过滤置信度小于阈值的框
        detections = [d for d in dets if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        # 对框进行非极大值抑制
        detections = [detections[i] for i in indices]

        # Update tracker.
        # 卡尔曼滤波对tracker跟踪器进行状态预测
        # 第一帧没有tracker
        tracker.predict()
        # 对跟踪器进行更新
        # 对未匹配的detections进行初始化，添加track
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        boxes = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(image,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 1)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # 图像经过darknet检测后，被resize到(608, 608)
            # 此时存储结果的话，需要重新resize回原图(1920 1080)对应的位置
            bbox = track.to_tlwh()
            x, y, w, h = bbox
            a_x, a_y, a_w, a_h = x/width, y/height, w/width, h/height
            results.append([index+1, track.track_id, org_w*a_x, org_h*a_y, org_w*a_w, org_h*a_h])

            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 1)
            cv2.putText(image, str(track.track_id),(int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (color),1)
            # if len(class_names) > 0:
            #    cv2.putText(image, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (color),1)
            
            # pt1 = int(bbox[0]), int(bbox[1])
            # pt2 = int(bbox[2]), int(bbox[3])
            # cv2.rectangle(image, pt1, pt2, color, 1)
            # if track.track_id is not None:
            #     text_size = cv2.getTextSize(str(track.track_id), cv2.FONT_HERSHEY_PLAIN, 1, 1)

            #     center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
            #     pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
            #     cv2.rectangle(image, pt1, pt2, color, -1)
            #     cv2.putText(image, str(track.track_id), center, cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            i += 1

            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            pts[track.track_id].append(center)
            thickness = 1
            #center point
            cv2.circle(image,  (center), 1, color, thickness)

	        #draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(image,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),1)
                #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

        # count = len(set(counter))
        # cv2.putText(image, "Total Object Counter: "+str(count),(int(20), int(120)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        # cv2.putText(image, "Current Object Counter: "+str(i),(int(20), int(80)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        cv2.putText(image, "FPS: %f"%(fps),(int(20), int(40)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
        cv2.namedWindow("YOLO_Deep_SORT", 0)
        cv2.resizeWindow('YOLO_Deep_SORT', 1024, 768)
        cv2.imshow('YOLO_Deep_SORT', image)

        if writeVideo_flag:
            out.write(image)
        fps  = ( fps + (1./(time.time()-prev_time)) ) / 2
        #print(set(counter))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # fps = int(1./(time.time() - prev_time))
        print("FPS: {}".format(fps))
        index = index + 1
    
    if writeVideo_flag:
        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]),file=list_file)
    # os.system(r'python yolo/darknet_images.py ' + 
    #             '--weights "D:/work/code/python/yolov4/yolov4.weights" ' + 
    #             '--input test_data/dog.jpg')

if __name__ == "__main__":
    main()