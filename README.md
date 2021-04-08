## deep_sort_yolov4

项目使用darknet版本的yolov4作为detector，不需要将模型进行转换，通过yolo_v4作为前端检测器，后面使用deep_sort官方提供的[mars-small128.pb](https://drive.google.com/open?id=18fKzfqnqhqW3s9zwsCbnVJ5XF2JFeqMp)模型和`generate_detections.py`代码提取feature，再进行deep_sort算法的跟踪流程。

大致流程：

- yolo_v4根据图片检测物体（人、车辆或其他），获取detections信息包含bbox，confidence等
- 将bbox传入encoder获取feature特征
- 接下来就是deep_sort算法的步骤，开始跟踪

### 1. 配置yolo_v4：

按博客配置python版本yolo_v4，博文详细提供了如何编译darknet，并测试运行: https://blog.csdn.net/u010512638/article/details/107931119



### 2. deep_sort：

1）下载MOT16官方数据集[MOT16.zip](https://motchallenge.net/data/MOT16.zip)，解压。

3）下载yolo_v4官方权重文件[yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights) (Google-drive mirror [yolov4.weights](https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT) )

2）运行main.py

```
python main.py --weights "D:/work/code/python/yolov4/yolov4.weights" --input D:/work/data/AutoDrivingData/MOT16/train/MOT16-02/img1 --output_file ./res_data/MOT16-02.txt
```



### 3.  评估：

参考我的另一篇博文，详细介绍了如何使用python版本的[MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit/tree/master/MOT)工具对MOT结果进行评测：https://blog.csdn.net/u010512638/article/details/115526828

评估结果不理想，与deep_sort官方结果相差甚远，后期需要改进的地方很多。

![在这里插入图片描述](https://img-blog.csdnimg.cn/202104082231478.png)



### Reference：

yolo_v4：https://github.com/AlexeyAB/darknet

deep_sort：https://github.com/nwojke/deep_sort

deep_sort_yolov3：https://github.com/Qidian213/deep_sort_yolov3

MOTChallengeEvalKit：https://github.com/dendorferpatrick/MOTChallengeEvalKit/tree/master/MOT