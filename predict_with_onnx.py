import onnxruntime as rt
from collections import Counter
import colorsys,time,onnxruntime,cv2
import numpy as np


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
       
        self.anchors_mask = anchors_mask

    def decode_box(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
         
            batch_size = input.shape[0]
            input_height = input.shape[2]
            input_width = input.shape[3]

            stride_h = self.input_shape[0] / input_height
            stride_w = self.input_shape[1] / input_width
        
            scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in
                              self.anchors[self.anchors_mask[i]]]

        
            prediction = input.reshape(batch_size, len(self.anchors_mask[i]), self.bbox_attrs, input_height,
                                       input_width)
            prediction = np.transpose(prediction, (0, 1, 3, 4, 2))
          
            x = 1 / (1 + np.exp(-prediction[..., 0]))
            y = 1 / (1 + np.exp(-prediction[..., 1]))
          
            w = 1 / (1 + np.exp(-prediction[..., 2]))
            h = 1 / (1 + np.exp(-prediction[..., 3]))
          
            angle = 1 / (1 + np.exp(-prediction[..., 4]))
        
            conf = 1 / (1 + np.exp(-prediction[..., 5]))
     
            pred_cls = 1 / (1 + np.exp(-prediction[..., 6:]))

        
            # (input_width,) --> (input_height, input_width)
            grid_x = np.linspace(0, input_width - 1, input_width)
            grid_x = np.tile(grid_x, (input_height, 1))
            # (input_height, input_width)--> (bs*3,input_height, input_width)
            # -->(bs,3,input_height, input_width)->(bs,3,input_height, input_width,1)
            grid_x = np.tile(grid_x, (batch_size * len(self.anchors_mask[i]), 1, 1)).reshape(x.shape)

            # (input_height,) -->(input_width,input_height)->(input_height, input_width)
            grid_y = np.linspace(0, input_height - 1, input_height)
            grid_y = np.tile(grid_y, (input_width, 1)).T
            grid_y = np.tile(grid_y, (batch_size * len(self.anchors_mask[i]), 1, 1)).reshape(y.shape)

            # (3,2)
            scaled_anchors = np.array(scaled_anchors)
            # (3,1)
            anchor_w = scaled_anchors[:, 0:1]
            anchor_h = scaled_anchors[:, 1:2]
            # (3,1) --> (3*batch_size,1) --> (1,3*batch_size,1)
            anchor_w = np.tile(anchor_w, (batch_size, 1)).reshape(1, -1, 1)
    # (1,3*batch_size,1) --> (1,3*batch_size,input_height * input_width)--> (1,3*batch_size,input_height , input_width)
            anchor_w = np.tile(anchor_w, (1, 1, input_height * input_width)).reshape(w.shape)
            anchor_h = np.tile(anchor_h, (batch_size, 1)).reshape(1, -1, 1)
            anchor_h = np.tile(anchor_h, (1, 1, input_height * input_width)).reshape(h.shape)

   
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
          
            class_conf = np.max(image_pred[:, 6:6 + num_classes], axis=1, keepdims=True)
            class_pred = np.argmax(image_pred[:, 6:6 + num_classes], axis=1)
            class_pred = np.expand_dims(class_pred, axis=1)

         
            conf_mask = (image_pred[:, 5] * class_conf[:, 0] >= conf_thres).squeeze()
           
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.shape[0]:
                continue
     
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
        "model_path": '/home/kingargroo/yolov7-obb/model_data/models.onnx',
        "input_shape": [1280, 1280],
        "confidence": 0.4,
        "nms_iou": 0.25,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化YOLO
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value

            # ---------------------------------------------------#
        #   获得种类和先验框的数量
        # ---------------------------------------------------#
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

    # ---------------------------------------------------#
    #   生成模型
    # ---------------------------------------------------#
    def generate(self):
        # ---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        # ---------------------------------------------------#
        self.net = onnxruntime.InferenceSession(self.model_path,
                                                providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                           'CPUExecutionProvider'])
        self.output_name = [i.name for i in self.net.get_outputs()]
        self.input_name = [i.name for i in self.net.get_inputs()]

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image):
    
        image_shape = np.array(np.shape(image)[0:2])
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = resize_Image(image, (self.input_shape[1], self.input_shape[0]), True)
        # ---------------------------------------------------------#
    
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        # ---------------------------------------------------------#
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

        count_more=0
        color_palette = np.random.uniform(0, 255, size=(1, 3))[0]
        for i, c in list(enumerate(top_label)):
            #predicted_class = self.class_names[int(c)]
            rbox = top_rboxes[i]
            score = top_conf[i]
            as_ratio = max(rbox[2] / rbox[3], rbox[3] / rbox[2])
            print(as_ratio)
            rbox = ((rbox[0], rbox[1]), (rbox[2], rbox[3]), rbox[4] * 180 / np.pi)
            cx,cy=rbox[0]
            #cv2.circle(image,(int(cx),int(cy)),1,(0, 255, 0),thickness=2)

            poly = cv2.boxPoints(rbox).astype(np.int32)
            area=cv2.contourArea(poly)
            # if as_ratio<2.2:
            #     continue
            #     count_more += 1
            # if area< 2000:
            #     count_more+=1
            #     continue
            x, y = np.min(poly[:, 0]), np.min(poly[:, 1]) - 20

            cv2.polylines(image, [poly.reshape((-1, 1, 2))], True, color_palette, thickness=2)
            label = ' {:.2f}'.format(score)
            #cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=1)


        total_num=len(top_label)-count_more
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

        new_img=new_img[:,:,::-1]
        print(total_num)
        return image[:,:,::-1],total_num

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
    """
        resize
    Args:
        size:target size
        letterbox_image: bool letterbox or not
    Returns:指定尺寸的图像
    """
    ih, iw, _ = image.shape
    h, w = size
    # letterbox_image = False
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # print(image.shape)
        # 生成画布
        image_back = np.ones((h, w, 3), dtype=np.uint8) * 128
        # 将image放在画布中心区域-letterbox
        image_back[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
    else:
        image_back = image
        # cv2.imshow("img", image_back)
        # cv2.waitKey()
    return image_back


def img2input(img):
    img = np.transpose(img, (2, 0, 1))
    img = img / 255
    return np.expand_dims(img, axis=0).astype(np.float32)


def std_output(pred):
    """
    （1，84，8400）-->（8400， 85）  85= box:4  conf:1 cls:80
    """
    pred = np.squeeze(pred)
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    return pred


def xywh2xyxy(*box):
    """
    
    Args:
        box:
    Returns: x1y1x2y2
    """
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
    """
    nms
    Args:
        pred: 模型输出特征图
        conf_thres: 置信度阈值
        iou_thres: iou阈值,列表，記錄對應class 的nms threshold ：{'corn':0.3,'cucumber':0.5,'wheat':0.7},即 [0.3,0.5,0.7]
    Returns: 
    """
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


def draw(res, image,pre_class,show=False):
    """
    draw bounding box
    Args:
        res: predicted result
        image: original
        cls: ["apple", "banana", "people"]
    Returns:
    """
    total_num=0
    color_palette = np.random.uniform(0, 255, size=(1, 3))[0]
    for r in res:

        x1,y1,x2,y2=r[0],r[1],r[2],r[3]
        w,h=x2-x1,y2-y1
        area=w*h
        as_ratio=max(w/h,h/w)
        print(area)
        # if pre_class!=3 and as_ratio>1.5:
        #     continue
        if area<1000 :
            continue

        #draw point
        #image=cv2.circle(image,center=(int((r[0]+r[2])/2),int((r[1]+r[3])/2)),radius=1,color=(0, 0, 255),thickness=5)
        image =cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),color=color_palette,thickness=4,lineType=1)
        total_num+=1
    text='number:{}'.format(total_num)
    imgh,imgw=image.shape[0:2]
    newh=imgh+350
    shape = (newh, imgw, 3)  # y, x, RGB
    # 直接建立全白圖片 100*100
    new_img = np.full(shape, 255)
    new_img[0:imgh,0:imgw,:]=image.copy()
    new_img=cv2.putText(new_img.astype(np.int32),text,(15,newh-230),cv2.FONT_HERSHEY_COMPLEX,5,(0, 0, 0), 10)
    new_img=cv2.putText(new_img.astype(np.int32),'id:{}'.format(1),(imgw-500,newh-230),cv2.FONT_HERSHEY_COMPLEX,5,(0, 0, 0), 10)
    if show:
        cv2.imshow("result", new_img)
        cv2.waitKey()
    return image,total_num


def predict_single_img(input_path,img_path,onnx_model_path,out_path,predict_class,class_list=['corn','sorghum','soybean','wheat']):
    std_h, std_w =1280,1280 
    class_list = class_list
    input_path = input_path
    img_path = img_path
    img = cv2.imread(input_path + img_path)
    if img.size == 0:
        print("error path")

    if predict_class=='rice':
        yolov7_oob = YOLO()
        t1=time.time()
        image,total_num = yolov7_oob.detect_image(img)
        t2=time.time()
        print("rice time:",t2-t1)
    else:

        # preprocess
        t1=time.time()
        img_after = resize_image(img, (std_w, std_h), True)  # （960， 1280， 3）
        data = img2input(img_after)
        t2=time.time()
        print('preprocess time:',t2-t1)
        # model input
        sess = rt.InferenceSession(onnx_model_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        #output : 8400x85
        pred = sess.run([label_name], {input_name: data})[0]
        pred = std_output(pred)
        t3=time.time()
        print('inference time:',t3-t2)
        # confidence filter+nms
        pred_boxes=pred[:,0:4]
        pred_scores=pred[:,4]
        after_nms_indice=cv2.dnn.NMSBoxes(bboxes=pred_boxes,scores=pred_scores,score_threshold=0.025,nms_threshold=0.4)
        after_pred=pred[after_nms_indice]
        box_result=after_pred[:,0:4]
        class_probility=after_pred[:,5::]
        conf_list = np.max(class_probility, axis=1).reshape((-1, 1))
        cls=list(np.argmax(class_probility, axis=1))
        pre_class = find_most_common_elements(cls)
        class_box = pre_class * np.ones((len(box_result), 1))
        result=np.concatenate([box_result,conf_list,class_box],axis=1)
        #print(result.shape)
        result = cod_trf(result, img, img_after)
        image,total_num = draw(result, img, pre_class)
        t4=time.time()
        print('NMS time:',t4-t3)
        print(t4-t1)
    out_path = out_path
    cv2.imwrite(out_path + img_path, image)
    return total_num






if __name__ == '__main__':

    total_num=predict_single_img(input_path="/home/kingargroo/seed/ablation1/new2/",img_path='WIN_20240403_16_23_25_Pro.jpg',onnx_model_path="/home/kingargroo/Documents/new_best.onnx",out_path="/home/kingargroo/seed/test/predict",predict_class='wheat')
    print(total_num)