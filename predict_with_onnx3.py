import onnxruntime as rt
from collections import Counter
import colorsys,time,onnxruntime,cv2
import numpy as np
import os
import pandas as pd

##############################################################################
#########YOLOV7-OBB prediction
#############################################################################
def resize_Image(image, size, letterbox_image):
    ih, iw = image.shape[:2]
    h, w = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
        new_image = 128 * np.ones((h, w, 3), dtype=np.uint8)
        new_image[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
    else:
        new_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    return new_image


def preprocess_input(image):
    image /= 255.0
    return image


class DecodeBox():
    def __init__(self, anchors, num_classes, input_shape, anchors_mask=[[6, 7, 8], [3, 4, 5], [0, 1, 2]]):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 6 + num_classes
        self.input_shape = input_shape
        # -----------------------------------------------------------#
        #   13x13:anchor [142, 110],[192, 243],[459, 401]
        #   26x26:  anchor [36, 75],[76, 55],[72, 146]
        #   52x52: anchor [12, 16],[19, 36],[40, 28]
        # -----------------------------------------------------------#
        self.anchors_mask = anchors_mask

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            # input:
            #   batch_size = 1
            #   batch_size, 3 * (5 + 1 + 80), 20, 20
            #   batch_size, 255, 40, 40
            #   batch_size, 255, 80, 80
            # -----------------------------------------------#
            batch_size = input.shape[0]
            input_height = input.shape[2]
            input_width = input.shape[3]


            #   stride_h = stride_w = 32、16、8
            # -----------------------------------------------#
            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
            # -------------------------------------------------#
            #  scaled_anchors size related to feature layer
            # -------------------------------------------------#
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              self.anchors[self.anchors_mask[i]]]


            #   batch_size, 3, 20, 20, 85
            #   batch_size, 3, 40, 40, 85
            #   batch_size, 3, 80, 80, 85
            # -----------------------------------------------#
            prediction = input.reshape(batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height,
                                       input_width)
            prediction = np.transpose(prediction, (0, 1, 3, 4, 2))
            # -----------------------------------------------#
            #   cx,cy
            # -----------------------------------------------#
            x = 1 / (1 + np.exp(-prediction[..., 0]))
            y = 1 / (1 + np.exp(-prediction[..., 1]))
            # -----------------------------------------------#
            #   w,h
            w = 1 / (1 + np.exp(-prediction[..., 2]))
            h = 1 / (1 + np.exp(-prediction[..., 3]))

            #   angle
            angle = 1 / (1 + np.exp(-prediction[..., 4]))

            conf = 1 / (1 + np.exp(-prediction[..., 5]))

            pred_cls = 1 / (1 + np.exp(-prediction[..., 6:]))


            #   prior center
            #   batch_size,3,20,20

            grid_x = np.linspace(0, input_width - 1, input_width)
            grid_x = np.tile(grid_x, (input_height, 1))
            grid_x = np.tile(grid_x, (batch_size * len(self.anchors_mask[i]), 1, 1)).reshape(x.shape)

            grid_y = np.linspace(0, input_height - 1, input_height)
            grid_y = np.tile(grid_y, (input_width, 1)).T
            grid_y = np.tile(grid_y, (batch_size * len(self.anchors_mask[i]), 1, 1)).reshape(y.shape)

            scaled_anchors = np.array(scaled_anchors)
            anchor_w = scaled_anchors[:, 0:1]
            anchor_h = scaled_anchors[:, 1:2]
            anchor_w = np.tile(anchor_w, (batch_size, 1)).reshape(1, -1, 1)
            anchor_w = np.tile(anchor_w, (1, 1, input_height * input_width)).reshape(w.shape)
            anchor_h = np.tile(anchor_h, (batch_size, 1)).reshape(1, -1, 1)
            anchor_h = np.tile(anchor_h, (1, 1, input_height * input_width)).reshape(h.shape)


            #   x 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 => response range
            #   y 0 ~ 1 => 0 ~ 2 => -0.5, 1.5 =>
            #   w 0 ~ 1 => 0 ~ 2 => 0 ~ 4 => wh: 0~4
            #   h 0 ~ 1 => 0 ~ 2 => 0 ~ 4 =>
            pred_boxes = np.zeros(prediction[..., :4].shape, dtype='float32')
            pred_boxes[..., 0] = x * 2. - 0.5 + grid_x
            pred_boxes[..., 1] = y * 2. - 0.5 + grid_y
            pred_boxes[..., 2] = (w * 2) ** 2 * anchor_w
            pred_boxes[..., 3] = (h * 2) ** 2 * anchor_h
            pred_theta = (angle - 0.5) * np.pi


            _scale = np.array([input_width, input_height, input_width, input_height]).astype('float32')
            output = np.concatenate(
                (pred_boxes.reshape(batch_size, -1, 4) / _scale, pred_theta.reshape(batch_size, -1, 1),
                 conf.reshape(batch_size, -1, 1), pred_cls.reshape(batch_size, -1, self.num_classes)), -1)


            outputs.append(output)
        return outputs

    def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5,
                            nms_thres=0.4):
        # ----------------------------------------------------------#
        #   prediction  [batch_size, num_anchors, 85]
        # ----------------------------------------------------------#

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            #   class_conf  [num_anchors, 1]
            #   class_pred  [num_anchors, 1]
            # ----------------------------------------------------------#
            class_conf = np.max(image_pred[:, 6:6 + num_classes], axis=1, keepdims=True)
            class_pred = np.argmax(image_pred[:, 6:6 + num_classes], axis=1)
            class_pred = np.expand_dims(class_pred, axis=1)


            conf_mask = (image_pred[:, 5] * class_conf[:, 0] >= conf_thres).squeeze()
            # ----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            # ----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.shape[0]:
                continue
            # -------------------------------------------------------------------------#
            #   detections  [num_anchors, 8]
            #   8：x, y, w, h, angle, obj_conf, class_conf, class_pred
            # -------------------------------------------------------------------------#
            detections = np.concatenate((image_pred[:, :6], class_conf, class_pred), 1)


            unique_labels = np.unique(detections[:, -1])

            for c in unique_labels:

                detections_class = detections[detections[:, -1] == c]


                bboxes = [[[bbox[0], bbox[1]], [bbox[2], bbox[3]], bbox[4] * 180 / np.pi] for bbox in
                          detections_class[:, :5]]
                scores = [float(score) for score in detections_class[:, 5] * detections_class[:, 6]]
                indices = cv2.dnn.NMSBoxesRotated(bboxes, scores, conf_thres, nms_thres)
                max_detections = detections_class[indices.flatten()]
                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else np.concatenate((output[i], max_detections))

            if output[i] is not None:
                output[i][:, :5] = self.yolo_correct_boxes(output[i], input_shape, image_shape, letterbox_image)
        return output

    def yolo_correct_boxes(self, output, input_shape, image_shape, letterbox_image):

        box_xy = output[..., 0:2]
        box_wh = output[..., 2:4]
        angle = output[..., 4:5]
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        if letterbox_image:

            new_shape = np.round(image_shape * np.min(input_shape / image_shape))
            offset = (input_shape - new_shape) / 2. / input_shape
            scale = input_shape / new_shape

            box_yx = (box_yx - offset) * scale
            box_hw *= scale

        box_xy = box_yx[:, ::-1]
        box_hw = box_wh[:, ::-1]

        rboxes = np.concatenate([box_xy, box_wh, angle], axis=-1)
        rboxes[:, [0, 2]] *= image_shape[1]
        rboxes[:, [1, 3]] *= image_shape[0]
        return rboxes


class YOLO(object):
    _defaults = {
        "model_path": 'models.onnx',
        "input_shape": [1280, 1280],
        "confidence": 0.25,
        "nms_iou": 0.25,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value


        self.class_names = ['Car']
        self.num_classes = 1
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.anchors = np.array([[12, 16], [19, 36], [40, 28],
                                 [36, 75], [76, 55], [72, 146],
                                 [142, 110], [192, 243], [459, 401]])
        self.num_anchors = 9
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]),
                                   self.anchors_mask)
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()


    def generate(self):

        self.net = onnxruntime.InferenceSession(self.model_path,
                                                providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                           'CPUExecutionProvider'])
        self.output_name = [i.name for i in self.net.get_outputs()]
        self.input_name = [i.name for i in self.net.get_inputs()]


    def detect_image(self, image):

        image_shape = np.array(np.shape(image)[0:2])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = resize_Image(image, (self.input_shape[1], self.input_shape[0]), True)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        outputs = self.net.run(self.output_name, {self.input_name[0]: image_data})
        outputs = self.bbox_util.decode_box(outputs)

        results = self.bbox_util.non_max_suppression(np.concatenate(outputs, axis=1), self.num_classes,
                                                     self.input_shape,
                                                     image_shape, True, conf_thres=self.confidence,
                                                     nms_thres=self.nms_iou)

        if results[0] is None:
            return image

        top_label = np.array(results[0][:, 7], dtype='int32')
        top_conf = results[0][:, 5] * results[0][:, 6]
        top_rboxes = results[0][:, :5]


        for i, c in list(enumerate(top_label)):
            #predicted_class = self.class_names[int(c)]
            rbox = top_rboxes[i]
            score = top_conf[i]
            rbox = ((rbox[0], rbox[1]), (rbox[2], rbox[3]), rbox[4] * 180 / np.pi)
            cx,cy=rbox[0]
            cv2.circle(image,(int(cx),int(cy)),1,(0, 255, 0),thickness=2)

            poly = cv2.boxPoints(rbox).astype(np.int32)
            x, y = np.min(poly[:, 0]), np.min(poly[:, 1]) - 20

        total_num=len(top_label)
        text = 'number:{}'.format(total_num)
        imgh, imgw = image.shape[0:2]
        newh = imgh + 350
        shape = (newh, imgw, 3)  # y, x, RGB
        new_img = np.full(shape, 255)
        new_img[0:imgh, 0:imgw, :] = image.copy()
        new_img = cv2.putText(new_img.astype(np.int32), text, (15, newh - 230), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 0),
                              10)
        new_img = cv2.putText(new_img.astype(np.int32), 'id:{}'.format(1), (imgw - 500, newh - 230),
                              cv2.FONT_HERSHEY_COMPLEX, 5, (0, 0, 0), 10)

        new_img=new_img[:, :, ::-1]
        return new_img, total_num

def bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=True, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    # Intersection area
    inter = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
            (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = np.maximum(b1_x2, b2_x2) - np.minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = np.maximum(b1_y2, b2_y2) - np.minimum(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / np.pi ** 2) * np.square(np.arctan(w2 / h2) - np.arctan(w1 / h1))
                alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

def find_most_common_elements(lst):
    counter = Counter(lst)
    most_common = counter.most_common(1)
    return most_common[0][0]

# 前处理
def resize_image(image, size, letterbox_image):

    ih, iw, _ = image.shape
    print(ih, iw)
    h, w = size
    # letterbox_image = False
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
        image_back[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
    else:
        image_back = image
    return image_back


def img2input(img):
    img = np.transpose(img, (2, 0, 1))
    img = img / 255
    return np.expand_dims(img, axis=0).astype(np.float32)


def std_output(pred):
    """
    将（1，84，8400）处理成（8400， 85）  85= box:4  conf:1 cls:80
    """
    pred = np.squeeze(pred)
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    return pred


def xywh2xyxy(*box):

    ret = [box[0] - box[2] // 2, box[1] - box[3] // 2, \
           box[0] + box[2] // 2, box[1] + box[3] // 2]
    return ret

def NMS(pred, conf_thres, iou_thres):

    #(num_anchors,box4+score1+num_classes) filter the confidence
    box = pred[pred[..., 4] > conf_thres]
    cls_conf = box[..., 5:]
    conf_list=np.max(cls_conf,axis=1).reshape((-1,1))
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    pre_class = find_most_common_elements(cls)
    sort_box = sorted(box, key=lambda x: -x[4])
    sort_box = np.array(sort_box)
    class_box=pre_class*np.ones((len(sort_box),1))
    sort_box_=sort_box[:,:4]
    #[x,y,w,h,conf,class]
    sort_box =np.concatenate([sort_box_,conf_list,class_box],axis=1)
    output_box = []
    # get the highest confidence box
    max_conf_box = sort_box[0]
    output_box.append(max_conf_box)
    sort_box = np.delete(sort_box, 0, 0)
    # NMS other bounding box except for max_conf_box
    while len(sort_box) > 0:
        #current highest confidence box
        max_conf_box = output_box[-1]
        del_index = []
        for j in range(len(sort_box)):
            current_box = sort_box[j]
            iou = bbox_iou(max_conf_box, current_box, CIoU=True)
            if iou > iou_thres[pre_class]:
                # iou exceed the threshold
                del_index.append(j)
        # delete the index
        sort_box = np.delete(sort_box, del_index, 0)
        if len(sort_box) > 0:
            output_box.append(sort_box[0])
            sort_box = np.delete(sort_box, 0, 0)
    return output_box




def nms(pred, conf_thres, iou_thres):

    #(num_anchors,box4+score1+num_classes)
    box = pred[pred[..., 4] > conf_thres]  # 置信度筛选
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))

    pre_class=find_most_common_elements(cls)
    total_cls = list(set(cls))  # 记录图像内共出现几种物体

    output_box = []
    # 每个预测类别分开考虑
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        temp = box[:, :6]
        for j in range(len(cls)):
            # 记录[x,y,w,h,conf(最大类别概率),class]值
            if cls[j] == clss:
                temp[j][5] = clss
                cls_box.append(temp[j][:6])
        #  cls_box 里面是[x,y,w,h,conf(最大类别概率),class]
        cls_box = np.array(cls_box)
        sort_cls_box = sorted(cls_box, key=lambda x: -x[4])  # 将cls_box按置信度从大到小排序
        # box_conf_sort = np.argsort(-box_conf)
        # 得到置信度最大的预测框
        max_conf_box = sort_cls_box[0]
        output_box.append(max_conf_box)
        sort_cls_box = np.delete(sort_cls_box, 0, 0)
        # 对除max_conf_box外其他的框进行非极大值抑制
        while len(sort_cls_box) > 0:
            # 得到当前最大的框
            max_conf_box = output_box[-1]
            del_index = []
            for j in range(len(sort_cls_box)):
                current_box = sort_cls_box[j]
                #iou = get_iou(max_conf_box, current_box)
                iou=bbox_iou(max_conf_box,current_box,CIoU=True)
                if iou > iou_thres[pre_class]:
                    # 筛选出与当前最大框Iou大于阈值的框的索引
                    del_index.append(j)
            # 删除这些索引
            sort_cls_box = np.delete(sort_cls_box, del_index, 0)
            if len(sort_cls_box) > 0:
                output_box.append(sort_cls_box[0])
                sort_cls_box = np.delete(sort_cls_box, 0, 0)
    return output_box


def cod_trf(result, pre, after):
    """
    因为预测框是在经过letterbox后的图像上做预测所以需要将预测框的坐标映射回原图像上
    Args:
        result:  [x,y,w,h,conf(最大类别概率),class]
        pre:    原尺寸图像
        after:  经过letterbox处理后的图像
    Returns: 坐标变换后的结果,
    """
    res = np.array(result)
    x, y, w, h, conf, cls = res.transpose((1, 0))
    x1, y1, x2, y2 = xywh2xyxy(x, y, w, h)  # 左上角点和右下角的点
    h_pre, w_pre, _ = pre.shape
    h_after, w_after, _ = after.shape
    scale = max(w_pre / w_after, h_pre / h_after)  # 缩放比例
    h_pre, w_pre = h_pre / scale, w_pre / scale  # 计算原图在等比例缩放后的尺寸
    x_move, y_move = abs(w_pre - w_after) // 2, abs(h_pre - h_after) // 2  # 计算平移的量
    ret_x1, ret_x2 = (x1 - x_move) * scale, (x2 - x_move) * scale
    ret_y1, ret_y2 = (y1 - y_move) * scale, (y2 - y_move) * scale
    ret = np.array([ret_x1, ret_y1, ret_x2, ret_y2, conf, cls]).transpose((1, 0))
    return ret


from PIL import Image, ImageDraw, ImageFont



def cv2_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    #fontStyle = ImageFont.truetype("STSONG.TTF", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def draw(res, image, show=False):
    """
    将预测框绘制在image上
    Args:
        res: 预测框数据
        image: 原图
        cls: 类别列表，类似["apple", "banana", "people"]  可以自己设计或者通过数据集的yaml文件获取
    Returns:
    """
    if res[0][5] == 0:
        class_area = 1000
    else:
        class_area = 500

    if res[0][5] == 3:
        class_ratio = 3
    else:
        class_ratio = 2

    print(class_area)
    for r in res:
        print(r[-2])
        #draw point
        x1, y1, x2, y2 = r[0], r[1], r[2], r[3]
        # print(x1, y1, x2, y2)
        w, h = x2 - x1, y2 - y1
        area = w * h
        as_ratio = max(w / h, h / w)
        if as_ratio > class_ratio:
            continue
        if area < class_area:
            continue

        # 面积过滤改为1000比较合适
        # if area > 1000:
        #     image = cv2.circle(image, center=(int((r[0] + r[2]) / 2), int((r[1] + r[3]) / 2)), radius=5,
        #                        color=(255, 0, 255), thickness=8)
        # else:

        image=cv2.circle(image,center=(int((r[0]+r[2])/2),int((r[1]+r[3])/2)),radius=1,color=(0, 0, 255),thickness=5)
        image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color = (255, 0, 255), thickness=2)

        # if r[-2] < 0.1:
        #     image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 0), thickness=2)


    text='number:{}'.format(len(res))
    imgh,imgw=image.shape[0:2]
    newh=imgh+350
    shape = (newh, imgw, 3)  # y, x, RGB
    # 直接建立全白圖片 100*100
    new_img = np.full(shape, 255)

    img_id = "1"
    img_code = "None"
    new_img[0:imgh, 0:imgw, :]=image.copy()

    if show:
        cv2.imshow("result", new_img)
        cv2.waitKey()
    return new_img,len(res)



def draw2(res, image, img_id, img_code, worker, radius = 5, color = (255, 255, 0)):
    """
    将预测框绘制在image上
    Args:
        res: 预测框数据
        image: 原图
        cls: 类别列表，类似["apple", "banana", "people"]  可以自己设计或者通过数据集的yaml文件获取
    Returns:
    """
    color_palette = np.random.uniform(0, 255, size=(1, 3))[0]
    for r in res:
        #draw point
        #image=cv2.circle(image, center=(int((r[0]+r[2])/2),int((r[1]+r[3])/2)), radius=radius, color=color,thickness=-1)
        image=cv2.rectangle(image,(int(r[0]),int(r[1])),(int(r[2]),int(r[3])),color=color_palette,thickness=4,lineType=1)
    imgh,imgw=image.shape[0:2]
    newh=imgh+350
    shape = (newh, imgw, 3)  # y, x, RGB
    # 直接建立全白圖片 100*100
    new_img = np.full(shape, 255)
    new_img[0:imgh, 0:imgw, :]=image.copy()


    return image, len(res)


def slice_img(img, out_folder = "scratch_file",
              sliceHeight=640, sliceWidth=640,
              overlap=0.1, pad=0,
              skip_highly_overlapped_tiles=False):

    if not os.path.exists(out_folder):
        # print("将创建输出文件夹,路径为:", out_folder)
        os.makedirs(out_folder)

    image = img
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    # 通过带重叠率的滑窗获得裁剪目标，其本质是窗口内值的复制
    n_ims = 0
    for y0 in range(0, image.shape[0], dy):
        for x0 in range(0, image.shape[1], dx):

            n_ims += 1
            # 当遇到后边界时，不希望裁剪出太小的部分，因此通过后边界向内裁的方式
            if y0 + sliceHeight > image.shape[0]:
                # 处于边界时，如果向内裁的部分比希望得到的尺寸的 0.6 倍还多，考虑放弃，这里选择的是 false，即不管重叠多少，都会裁剪
                if skip_highly_overlapped_tiles:
                    if (y0 + sliceHeight - image.shape[0]) > (0.6 * sliceHeight):
                        continue
                    else:
                        y = image.shape[0] - sliceHeight
                else:
                    y = image.shape[0] - sliceHeight
            else:
                y = y0
            if x0 + sliceWidth > image.shape[1]:
                if skip_highly_overlapped_tiles:
                    if (x0 + sliceWidth - image.shape[1]) > (0.6 * sliceWidth):
                        continue
                    else:
                        x = image.shape[1] - sliceWidth
                else:
                    x = image.shape[1] - sliceWidth
            else:
                x = x0

            # 这里获得了要裁剪的部分在整幅图像上的位置
            xmin, xmax, ymin, ymax = x, x + sliceWidth, y, y + sliceHeight
            window_c = image[ymin:ymax, xmin:xmax]

            # out_final = out_folder + "/" + str(n_ims) + ".jpg"
            out_final = os.path.join(out_folder, 'capture' + '__' + str(y) + '_' + str(x) + '_'
                                     + str(sliceHeight) + '_' + str(sliceWidth)
                                     + '_' + str(0) + '_' + str(img.shape[1]) + '_' + str(img.shape[0])
                                     + '.jpg')

            cv2.imwrite(out_final, window_c)



def predict_with_yolov8(model_name, img_path, save_path):
    t0 = time.time()
    file_list = os.listdir(img_path)
    img_list = [file for file in file_list if file.endswith('.jpg')]
    # print('the amount of images needed to be predict is:', len(img_list))
    for img_name in img_list:
        img_name_ = img_path + '/' + img_name
        img_title = img_name.split('.')[0]
        img = cv2.imread(img_name_)
        txt_name = save_path + '/' + str(img_title) + '.txt'
        txt_file = open(txt_name, 'w')
        std_w, std_h = 640, 640
        img_after = resize_image(img, (std_w, std_h), True)
        data = img2input(img_after)
        # model input
        sess = rt.InferenceSession(model_name)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        try:
            # output : 8400x85
            pred = sess.run([label_name], {input_name: data})[0]
            pred = std_output(pred)
            pred_boxes = pred[:, 0:4]
            pred_scores = pred[:, 4]
            after_nms_indice = cv2.dnn.NMSBoxes(bboxes=pred_boxes, scores=pred_scores, score_threshold=0.5,
                                                nms_threshold=0.3)
            after_pred = pred[after_nms_indice]
            box_result = after_pred[:, 0:4]
            class_probility = after_pred[:, 5::]
            conf_list = np.max(class_probility, axis=1).reshape((-1, 1))
            pre_class = 0
            class_box = pre_class * np.ones((len(box_result), 1))
            results = np.concatenate([box_result, conf_list, class_box], axis=1)
            for result in results:
                xc, yc, w, h, score, _ = result
                xc, yc, w, h = xc / std_w, yc / std_h, w / std_w, h / std_h
                area = w * h
                as_ratio=max(w/h,h/w)
                #print(as_ratio)
                if area < 0.001:
                    continue
                if as_ratio>1.5:
                    continue
                output_str = '0' + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(w) + ' ' + str(h) + ' ' + str(score) + '\n'
                txt_file.write(output_str)
                # txt_file.close()
        except:
            continue
    t1 = time.time()
    print('total prediction time is :', t1 - t0)

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        os.rmdir(folder_path)
        print(f'Folder "{folder_path}" has been deleted.')

def convert_reverse(size, box):
    x, y, w, h = box
    dw = 1./size[0]
    dh = 1./size[1]

    w0 = w/dw
    h0 = h/dh
    xmid = x/dw
    ymid = y/dh

    x0, x1 = xmid - w0/2., xmid + w0/2.
    y0, y1 = ymid - h0/2., ymid + h0/2.

    return [x0, x1, y0, y1]


def get_global_coords(row,
                      edge_buffer_test=0,
                      max_edge_aspect_ratio=2.5,
                      test_box_rescale_frac=1.0,
                      max_bbox_size_pix=100):


    #从row即传过来的df，获取参数信息
    xmin0, xmax0 = row['Xmin'], row['Xmax']
    ymin0, ymax0 = row['Ymin'], row['Ymax']
    upper, left = row['Upper'], row['Left']
    sliceHeight, sliceWidth = row['Height'], row['Width']
    vis_w, vis_h = row['Im_Width'], row['Im_Height']
    pad = row['Pad']
    dx = xmax0 - xmin0
    dy = ymax0 - ymin0

    #如果框的大小过大，视为检测错误，直接删除
    if (dx > max_bbox_size_pix) \
            or (dy > max_bbox_size_pix):
        return [], []

    #如果框在边缘区域，根据设定值进行处理
    if edge_buffer_test > 0:
        #边缘多少个像素内，不允许出现框
        if ((float(xmin0) < edge_buffer_test) or
            (float(xmax0) > (sliceWidth - edge_buffer_test)) or
            (float(ymin0) < edge_buffer_test) or
                (float(ymax0) > (sliceHeight - edge_buffer_test))):
            # print ("离边缘太近，跳过", row, "...")
            return [], []

        elif ((float(xmin0) < edge_buffer_test) or
                (float(xmax0) > (sliceWidth - edge_buffer_test)) or
                (float(ymin0) < edge_buffer_test) or
                (float(ymax0) > (sliceHeight - edge_buffer_test))):
            #计算纵横比
            if (1.*dx/dy > max_edge_aspect_ratio) \
                    or (1.*dy/dx > max_edge_aspect_ratio):
                # print ("离边缘太近，纵横比高，跳过", row, "...")
                return [], []

    #跳过高纵横比,瘦长的检测框不要
    if (1.*dx/dy > max_edge_aspect_ratio) \
            or (1.*dy/dx > max_edge_aspect_ratio):
        return [], []

    #转换到全局坐标，其实就是相对子图的绝对像素加上了图名中的子图位置像素，pad指的是滑窗时填充的像素个数
    xmin = max(0, int(round(float(xmin0)))+left - pad)
    xmax = min(vis_w - 1, int(round(float(xmax0)))+left - pad)
    ymin = max(0, int(round(float(ymin0)))+upper - pad)
    ymax = min(vis_h - 1, int(round(float(ymax0)))+upper - pad)

    #框缩放与否
    if test_box_rescale_frac != 1.0:
        dl = test_box_rescale_frac
        xmid, ymid = np.mean([xmin, xmax]), np.mean([ymin, ymax])
        dx = dl*(xmax - xmin) / 2
        dy = dl*(ymax - ymin) / 2
        x0 = xmid - dx
        x1 = xmid + dx
        y0 = ymid - dy
        y1 = ymid + dy
        xmin, xmax, ymin, ymax = x0, x1, y0, y1

    #设置边界、点坐标
    bounds = [xmin, xmax, ymin, ymax]
    coords = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]

    #检查是否没有出错的部分，如果有，会退出
    if np.min(bounds) < 0:
        print(" 预测框出现负值:", bounds)
        print(" 出错数据为:", row)
        print(" 返回中")
        return
    if (xmax > vis_w) or (ymax > vis_h):
        print(" 预测框大于原图尺寸:", bounds)
        print(" 出错数据为:", row)
        print(" 返回中")
        return

    return bounds, coords



def augment_data(df,
                 slice_sep='__',
                 max_box_size=300,
                 edge_buffer_test=0,
                 max_edge_aspect_ratio=5,
                 test_box_rescale_frac=1.0):

    #df指datafream
    t0 = time.time()
    print("运行augment_data函数，此函数用于将子图坐标转换到全图坐标")
    print("数据帧的初始长度为:", len(df))

    im_roots, im_locs = [], []
    for j, im_name in enumerate(df['im_name'].values):

        root_tmp = im_name.split(slice_sep)[0]
        coo_tmp = im_name.split(slice_sep)[-1]

        im_locs.append(coo_tmp)

        if '.' not in root_tmp:
            im_roots.append(root_tmp + '.' + im_name)
        else:
            im_roots.append(root_tmp)

    df['Image_Root'] = im_roots  # df里又放了部分数据
    df['Slice_XY'] = im_locs
    # 由图片名称获取子图的相关讯息，并写入df中
    df['Upper'] = [float(sl.split('_')[0]) for sl in df['Slice_XY'].values]
    df['Left'] = [float(sl.split('_')[1]) for sl in df['Slice_XY'].values]
    df['Height'] = [float(sl.split('_')[2]) for sl in df['Slice_XY'].values]
    df['Width'] = [float(sl.split('_')[3]) for sl in df['Slice_XY'].values]
    df['Pad'] = [float(sl.split('_')[4].split('.')[0])
                 for sl in df['Slice_XY'].values]
    df['Im_Width'] = [float(sl.split('_')[5].split('.')[0])
                      for sl in df['Slice_XY'].values]
    df['Im_Height'] = [float(sl.split('_')[6].split('.')[0])
                       for sl in df['Slice_XY'].values]

    print("图名信息已导入")

    x0l, x1l, y0l, y1l = [], [], [], []
    bad_idxs = []
    for index, row in df.iterrows():
        bounds, coords = get_global_coords(
            row,
            edge_buffer_test=edge_buffer_test,
            max_edge_aspect_ratio=max_edge_aspect_ratio,
            max_bbox_size_pix=max_box_size,
            test_box_rescale_frac=test_box_rescale_frac)
        if len(bounds) == 0 and len(coords) == 0:
            bad_idxs.append(index)
            [xmin, xmax, ymin, ymax] = 0, 0, 0, 0
        else:
            [xmin, xmax, ymin, ymax] = bounds
        x0l.append(xmin)
        x1l.append(xmax)
        y0l.append(ymin)
        y1l.append(ymax)
    df['Xmin_Glob'] = x0l
    df['Xmax_Glob'] = x1l
    df['Ymin_Glob'] = y0l
    df['Ymax_Glob'] = y1l

    #删除部分不好的索引
    if len(bad_idxs) > 0:
        print("不满足边界、纵横比等要求的预测框数量:", len(bad_idxs))
        df = df.drop(index=bad_idxs)

    print("子图坐标转换至全图坐标用时“:", time.time() - t0, "秒")
    print("剩余全图预测框个数为:", len(df))
    return df



def execute(img, labels_dir='scratch_file',
            subgraph_size=640,
            classes=['millet'],
            detect_thresh=0.2,      #置信度低于这个值会被去掉
            max_edge_aspect_ratio=5,
            edge_buffer_test=0,
            ):

    data_list = []

    #遍历labes文件夹下的所有标签，把所有的内容取出来，方便后续处理
    for txt_name in sorted(z for z in os.listdir(labels_dir) if z.endswith('.txt')):
        txt_path = os.path.join(labels_dir, txt_name)
        prefix_name =txt_name.split('.txt')[0]
        #以csv格式读取当前txt文件，读取对象为txt_path
        data = pd.read_csv(txt_path, header=None, index_col=None, sep=' ',
                           names=['cat_int', 'x_frac', 'y_frac', 'w_frac', 'h_frac', 'prob'])

        #如果读的txt文件里有东西，注意：此时读到的是单个txt文件，也即一个子图对应的标签文件
        if len(data) > 0:
            #新建用于存储中间数据和键值的列表
            out_data = []
            out_cols = ['im_name', 'prob', 'Xmin', 'Ymin', 'Xmax', 'Ymax', 'cat_int', 'category']

            #下面对读到的单个标签文件中的每一条进行处理
            for dt in data.values:
                #获得类别,必须是classes中给出的
                cat_int = int(dt[0])
                if classes:
                    cat_str = classes[cat_int]
                else:
                    cat_str = ''
                #获得置信度
                prob = float(dt[5])
                #获得位置分数
                box =dt[1:5]
                #转换至子图像素坐标
                pix_box = convert_reverse((subgraph_size, subgraph_size), box)
                [x0, x1, y0, y1] =pix_box

                #将txt名称和上述获得的类别、置信度、子图像素坐标值写入out_data中
                out_data.append([prefix_name, prob, x0, y0, x1, y1, cat_int, cat_str])
            #将从当前txt文件读到的数据按指定顺序写入data_list
            data_part = pd.DataFrame(out_data, columns=out_cols)
            data_list.append(data_part)

    #检查data_list里是否有元素
    if len(data_list) == 0:
        print("没有要处理的输出文件，即将返回")
        return
    else:
        data_raw = pd.concat(data_list)
        data_raw.index = range(len(data_raw))
        print("提取成功!")

    #拿到没有附加切片坐标的图像名称，例如xinmi_0_0,会变成xinmi
    #采用的方式是通过_进行划分，并取第一个_之前的值
    im_name_root_list = [z.split('_')[0] for z in data_raw['im_name'].values]
    data_raw['im_name_root'] = im_name_root_list

    #直接过滤置信度直接低于阈值的对象
    data_raw = data_raw[data_raw['prob'] >= detect_thresh]
    data_raw['image_path'] = "capture.jpg"

    df_tiled_aug = augment_data(data_raw,                                       #传入数据帧
                    slice_sep='__',                                             #图名分割标志
                    edge_buffer_test=edge_buffer_test,                          #如果边界框与边的距离在此范围内，放弃
                    max_edge_aspect_ratio=max_edge_aspect_ratio,                #窗口边缘附近框的边界框的最大纵横比
                    test_box_rescale_frac=1.0,                                  #框缩放系数
                    max_box_size=300)                                           #允许的最大预测框像素值

    global_coo = []
    scores = []
    for i, dta in enumerate(df_tiled_aug.values):
        score = dta[1]
        width = dta[-3] - dta[-4]
        height = dta[-1] - dta[-2]
        x_center = dta[-4] + width/2
        y_center = dta[-2] + height/2
        coo = [x_center, y_center, width, height]
        scores.append(score)
        global_coo.append(coo)

    coo_before_nms = np.array(global_coo)
    # opencv的nms，目前阈值0.3
    after_nms_indice = cv2.dnn.NMSBoxes(bboxes=coo_before_nms, scores=scores, score_threshold=0.3,
                                        nms_threshold=0.3)

    print("nms后的数量", len(after_nms_indice))
    filtered_data = df_tiled_aug.iloc[after_nms_indice]

    columns_to_extract = [-4, -2, -3, -1]

    res = filtered_data.iloc[:, columns_to_extract].values
    img_result, num = draw2(res=res, image=img, img_id=1, img_code="dwd", worker=None)
    #delete_folder("scratch_file")
    return img_result, num


################################################################################################
#下面是调用各个模型的函数
################################################################################################


def predict_single_img(img, predict_index):
    std_h, std_w =1280, 1280 # 标准输入尺寸
    img = img
    if img.size == 0:
        print("图像传输错误")

    if predict_index == 1:
        yolov7_oob = YOLO()
        t1=time.time()
        image,total_num = yolov7_oob.detect_image(img)
        t2=time.time()
        print('predict time:',t2-t1)
    elif predict_index == 0:
        # preprocess
        t1=time.time()
        img_after = resize_image(img, (std_w, std_h), True)  # （960， 1280， 3）
        # 添加了一个mask进行测试



        data = img2input(img_after)
        t2 = time.time()
        print('preprocess time:',t2-t1)
        # model input
        sess = rt.InferenceSession("vision6.onnx")
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        #output : 8400x85
        pred = sess.run([label_name], {input_name: data})[0]
        pred = std_output(pred)
        t3=time.time()
        print('inference time:',t3-t2)
        # confidence filter+nms
        pred_boxes=pred[:, 0:4]
        pred_scores=pred[:, 4]
        after_nms_indice=cv2.dnn.NMSBoxes(bboxes=pred_boxes, scores=pred_scores, score_threshold=0.5, nms_threshold=0.2)
        after_pred = pred[after_nms_indice]
        box_result=after_pred[:,0:4]
        class_probility=after_pred[:,5::]
        conf_list = np.max(class_probility, axis=1).reshape((-1, 1))
        cls=list(np.argmax(class_probility, axis=1))
        pre_class = find_most_common_elements(cls)
        class_box = pre_class * np.ones((len(box_result), 1))
        print(pre_class)
        result=np.concatenate([box_result,conf_list,class_box],axis=1)
        print(result.shape)
        result = cod_trf(result, img, img_after)
        image,total_num = draw(result, img)
        t4=time.time()
        print('NMS time:', t4-t3)
    elif predict_index == 2:
        t1=time.time()
        slice_img(img, out_folder="scratch_file")
        predict_with_yolov8(model_name="/home/kingargroo/seed/ablation1/v8n.onnx", img_path="scratch_file", save_path="scratch_file")
        image, total_num = execute(img, labels_dir='scratch_file',
                subgraph_size=640)
        t2=time.time()
        print("11:",t2-t1)
    image = image.astype(np.uint8)
    return image, total_num






if __name__ == '__main__':

    combox_index = 2
    img_name_list=os.listdir('/home/kingargroo/seed/ablation1/test_result/millet')
    total=0
    num_list=[]

    img_path='/home/kingargroo/seed/test/out2paper/m2.jpg'
    img = cv2.imread(img_path, cv2.NORMAL_CLONE)
    result_img, total_num = predict_single_img(img=img, predict_index=combox_index)
    print(total_num)
    num_list.append(total_num)
    total+=total_num

    #result_img = cv2.resize(result_img, (640, 480))
    cv2.imwrite('resultmillet.jpg',result_img)
