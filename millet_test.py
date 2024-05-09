import os
import time
import skimage.io
import pandas as pd
from PIL import Image
import colorsys,time,onnxruntime,cv2
import onnxruntime as rt
from predict_with_onnx import resize_image,img2input,std_output


#裁剪路径下的tif文件为png图片，并保存到指定文件夹下，如果输出文件夹不存在，则创建，如果存在，则输出提示信息
#tif_path为输入路径，out_path为输出目录，out_name为输出文件夹


def slice_tif_(im_path, out_dir, out_name,
              sliceHeight=640, sliceWidth=640,
              overlap=0.1, out_ext='.png',pad = 0,
              overwrite=True,
              skip_highly_overlapped_tiles=False,enhance_contrast=True):

    #计时开始
    t0 = time.time()

    #读取文件
    image = skimage.io.imread(im_path)
    print("输入图片尺寸:", image.shape)

    win_h, win_w = image.shape[:2]
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)

    #通过带重叠率的滑窗获得裁剪目标，其本质是窗口内值的复制
    n_ims = 0
    for y0 in range(0, image.shape[0], dy):
        for x0 in range(0, image.shape[1], dx):

            n_ims += 1
            if (n_ims % 100) == 0:
                print(n_ims)

            # 当遇到后边界时，不希望裁剪出太小的部分，因此通过后边界向内裁的方式
            if y0+sliceHeight > image.shape[0]:
                # 处于边界时，如果向内裁的部分比希望得到的尺寸的0.6倍还多，考虑放弃，这里选择的是false，即不管重叠多少，都会裁剪
                if skip_highly_overlapped_tiles:
                    if (y0+sliceHeight - image.shape[0]) > (0.6*sliceHeight):
                        continue
                    else:
                        y = image.shape[0] - sliceHeight
                else:
                    y = image.shape[0] - sliceHeight
            else:
                y = y0
            if x0+sliceWidth > image.shape[1]:
                if skip_highly_overlapped_tiles:
                    if (x0+sliceWidth - image.shape[1]) > (0.6*sliceWidth):
                        continue
                    else:
                        x = image.shape[1] - sliceWidth
                else:
                    x = image.shape[1] - sliceWidth
            else:
                x = x0

            # 这里获得了要裁剪的部分在整幅图像上的位置
            xmin, xmax, ymin, ymax = x, x+sliceWidth, y, y+sliceHeight
            window_c = image[ymin:ymax, xmin:xmax]



            #window_c= window_c.astype(np.uint8)

            #拼接出输出路径来,out_dir是传入的完整路径，精确到图片文件夹，out_name是tif图片的前缀
            out_final = os.path.join(out_dir, out_name + '__' + str(y) + '_' + str(x) + '_'
                                     + str(sliceHeight) + '_' + str(sliceWidth)
                                     + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h)
                                     + out_ext)


            #输出图片
            if not os.path.exists(out_final):
                skimage.io.imsave(out_final, window_c, check_contrast=False)
            elif overwrite:
                 skimage.io.imsave(out_final, window_c, check_contrast=False)
            else:
                print("文件已存在，overwrite设置为False，不进行覆盖")

    print("图片裁剪数量:", n_ims,
          "裁剪图片高为:{}，宽为:{}".format(sliceHeight, sliceWidth))
    print("裁剪用时:", im_path, time.time()-t0, '秒')
    return







def slice_to_png(tif_path, out_path,
                     Height = 640, Width = 640,
                     overlap = 0.1, out_ext = '.png',
                     overwrite = False,in_ext='tif',
                     skip_highly_overlapped_tiles=False):

    #计时
    t0 = time.time()

    #获取输入路径中的全部tif文件
    im_list = [z for z in os.listdir(tif_path) if z.endswith(in_ext)]

    #拼完整输出路径出来
    out_folder = out_path
    #不存在则创建，存在则给出提示信息
    if not os.path.exists(out_folder):
        print("将创建输出文件夹,路径为:", out_folder)
        os.makedirs(out_folder)

    #裁剪前的一些准备
    for i, im_name in enumerate(im_list):
        im_path = os.path.join(tif_path, im_name)
        im_tmp = skimage.io.imread(im_path)
        h, w = im_tmp.shape[:2]
        print("第{}张图片:{};".format(i+1, im_name), "高，宽:", h, w)

        out_name = im_name.split('.')[0]
        slice_tif_(im_path=im_path, out_dir=out_folder, out_name=out_name,
                  sliceHeight=Height, sliceWidth=Width,
                  overlap=overlap, out_ext=out_ext,
                  overwrite=overwrite,
                  skip_highly_overlapped_tiles=skip_highly_overlapped_tiles)
    out_list = []
    for f in sorted([z for z in os.listdir(out_folder) if z.endswith(out_ext)]):
        out_list.append(os.path.join(out_folder, f))
        df_tmp = pd.DataFrame({'image': out_list})



    print("进程结束，所用时间为:", time.time()-t0, "秒")




def non_max_suppression(boxes, probs=[],overlapThresh=0.8):

    len_init = len(boxes)
    print("NMS is doing, initial bounding box number：{}".format(len_init))
    #如果输入的坐标组没有对象，直接返回空列表
    if len(boxes) == 0:
        return [], [], []

    #boxs转array
    boxes = np.asarray([b[:4] for b in boxes])
    #将boxes里的数据类型转换为float类型
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    #初始化所选索引的列表
    pick = []
    #获取边界框的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    #计算框面积
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    #如果置信度为空，根据边界框的右下角y坐标对框进行排序
    if len(probs) == 0:
        idxs = np.argsort(y2)
    #否则，按最高prob（降序）对框进行排序
    else:
        # idxs = np.argsort(probs)[::-1]
        idxs = np.argsort(probs)

    #保持循环，同时某些索引仍保留在索引中
    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]

        pick.append(i)

        #将拿到的对象和其他目标做坐标方面的对比，x1[i]比x2[i]小，y1[i]比y2[i]小
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

            #删掉
        idxs = np.delete(
                idxs,
                np.concatenate(([last], np.where(overlap > overlapThresh)[0])))


    print("after NMS,bounding box number is :", len(pick))
    return pick

def predict_with_yolov8(model_name,img_path,save_path):

    t0 = time.time()
    img_list=os.listdir(img_path)
    print('the amount of images needed to be predict is:',len(img_list))
    for img_name in img_list:
        img_name_=img_path+'/'+img_name
        img_title = img_name.split('.')[0]
        img=cv2.imread(img_name_)

        txt_name=save_path+str(img_title)+'.txt'
        txt_file = open(txt_name, 'a+')

        std_w, std_h=640,640
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
            after_nms_indice = cv2.dnn.NMSBoxes(bboxes=pred_boxes, scores=pred_scores, score_threshold=0.05,
                                                nms_threshold=0.3)
            after_pred = pred[after_nms_indice]
            box_result = after_pred[:, 0:4]
            class_probility = after_pred[:, 5::]
            conf_list = np.max(class_probility, axis=1).reshape((-1, 1))
            pre_class = 0
            class_box = pre_class * np.ones((len(box_result), 1))
            results = np.concatenate([box_result, conf_list, class_box], axis=1)
            for result in results:
                xc,yc,w,h,score,_=result
                xc, yc, w, h=xc/std_w,yc/std_h,w/std_w,h/std_h
                area=w * h
                #print(area)
                if area<0.00075:
                    continue
                output_str='0'+' '+str(xc)+' '+str(yc)+' '+str(w)+' '+str(h)+' '+str(score)+'\n'
                txt_file.write(output_str)
        except:
            continue

    t1=time.time()
    print('total prediction time is :',t1-t0)


import time
import skimage.io

#用于将子图中的相对坐标，转换为相对子图的绝对坐标
def convert_reverse(size, box):
    #这里的x、y是中心点坐标，w、h是宽高，预测框的四点值为中心点坐标偏移一般宽高以后得到
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

#########################################################################################################



#####################################################################################
#用于将子图绝对坐标转换到全图绝对坐标，并去除部分不满足需求的目标框
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

#准备数据，以将子图坐标转换到全图坐标


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

        if(j % 10000) == 0:
            print("运行至第{}条".format(j))

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

#########################################################################################################





#####################################################################################

#重叠框的非极大值抑制
def py_cpu_softnms(dets, sc, Nt=0.3, sigma=0.5, thresh=0.05, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1)

    # the order of boxes coordinate is [y1,x1,y2,x2]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = sc
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1

        #
        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # IoU calculate
        xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        yy2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, 4][scores > thresh]
    keep = inds.astype(int)

    return keep


def non_max_suppression(boxes, probs=[], overlapThresh=0.4):

    print("正在进行重叠框的非极大值抑制")
    len_init = len(boxes)
    print('input box number:',len_init)
    #如果输入的坐标组没有对象，直接返回空列表
    if len(boxes) == 0:
        return [], [], []

    #boxs转array
    boxes = np.asarray([b[:4] for b in boxes])
    #将boxes里的数据类型转换为float类型
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    #初始化所选索引的列表
    pick = []

    #获取边界框的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    #计算框面积
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    #如果置信度为空，根据边界框的右下角y坐标对框进行排序
    if len(probs) == 0:
        idxs = np.argsort(y2)
    #否则，按最高prob（降序）对框进行排序
    else:
        # idxs = np.argsort(probs)[::-1]
        idxs = np.argsort(probs)

    #保持循环，同时某些索引仍保留在索引中
    while len(idxs) > 0:

        last = len(idxs) - 1
        i = idxs[last]

        pick.append(i)

        #将拿到的对象和其他目标做坐标方面的对比，x1[i]比x2[i]小，y1[i]比y2[i]小
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]



            #删掉
        idxs = np.delete(
                idxs,
                np.concatenate(([last], np.where(overlap > overlapThresh)[0])))


    print("重叠框NMS后，剩余预测框数量为:", len(pick))

    return pick



#移除低于检测阈值的元素，并应用非最大值抑制
def refine_data(df, groupby='Image_Path',
              groupby_cat='Category',
              cats_to_ignore=[],
              use_weighted_nms=True,
              nms_overlap_thresh=0.5, plot_thresh=0.5):

    print("运行refine_data函数，此函数用于过滤掉不需要的预测框")
    t0 = time.time()

    # 按图像分组和绘图，以下函数功能为以groupby所指的字符串进行分组
    group = df.groupby(groupby)
    count = 0
    df_idxs_tot = []


    # 分组以后 同一路径的比如说  /.../xinmi.jpg的会放在一个组里
    # 等效于只对属于一张图的子图做处理
    for i, g in enumerate(group):

        # 所属图片路径，精准到图片层次
        img_loc_string = g[0]
        # 其他全部信息
        data_all_classes = g[1]

        #这里的if语句只是起到了划分分组处理和不分组处理的作用
        #说白了就是将分出来的其他信息再按groupby_cat分一次组
        #分到了group2里面
        if len(groupby_cat) > 0:
            group2 = data_all_classes.groupby(groupby_cat)
            #class_str即类别
            for j, g2 in enumerate(group2):
                class_str = g2[0]

                #判断一下当前类别需不需要被忽略，如果需要跳过后续操作
                if (len(cats_to_ignore) > 0) and (class_str in cats_to_ignore):
                    print("忽略类别:", class_str)
                    continue

                #取出j类对象除类别以外的其他信息
                data = g2[1]
                #取出索引
                df_idxs = data.index.values
                #取出置信度分数
                scores = data['prob'].values

                #取出坐标信息
                xmins = data['Xmin_Glob'].values
                ymins = data['Ymin_Glob'].values
                xmaxs = data['Xmax_Glob'].values
                ymaxs = data['Ymax_Glob'].values

                #只保留置信度大于要求的对象
                high_prob_idxs = np.where(scores >= plot_thresh)
                scores = scores[high_prob_idxs]

                #取出留下的对象的坐标信息，此时是全图像素坐标
                xmins = xmins[high_prob_idxs]
                xmaxs = xmaxs[high_prob_idxs]
                ymins = ymins[high_prob_idxs]
                ymaxs = ymaxs[high_prob_idxs]
                #取出这些对象对应的索引
                df_idxs = df_idxs[high_prob_idxs]
                #坐标整合到boxes中
                boxes = np.stack((ymins, xmins, ymaxs, xmaxs), axis=1)


                # 重叠框的过滤，首先判断是不是要进行过滤
                if nms_overlap_thresh > 0:
                    #和上面的boxes是一个东西，这里是为了方便处理里面的对象新创新了一个
                    boxes_nms_input = np.stack(
                        (xmins, ymins, xmaxs, ymaxs), axis=1)
                    #判断是否只在满足上面置信度要求的目标对象上做重叠狂过滤，默认为True
                    if use_weighted_nms:
                        probs = scores
                    else:
                        probs = []

                    #拿到重叠框过滤后的索引
                    good_idxs = non_max_suppression(
                        boxes_nms_input, probs=probs,
                        overlapThresh=nms_overlap_thresh)

                    #good_idxs=py_cpu_softnms(boxes_nms_input,probs)


                    if len(boxes) == 0:
                        print("未检测到预测框")
                    #获得重叠框过滤后留下的预测框的索引
                    df_idxs = df_idxs[good_idxs]

                #当前类别下筛选后留下的预测框索引入列表，并保留该类被留下的预测框的个数，计入count
                df_idxs_tot.extend(df_idxs)
                count += len(df_idxs)

        #如果不进行分组检测？
        else:
            data = data_all_classes.copy()
            #过滤掉不要的类别
            if len(cats_to_ignore) > 0:
                data = data[~data['Category'].isin(cats_to_ignore)]
            #获得保留的预测框的索引、置信度、坐标
            df_idxs = data.index.values
            scores = data['prob'].values


            xmins = data['Xmin_Glob'].values
            ymins = data['Ymin_Glob'].values
            xmaxs = data['Xmax_Glob'].values
            ymaxs = data['Ymax_Glob'].values

            #过滤掉低置信度的对象
            high_prob_idxs = np.where(scores >= plot_thresh)
            #拿到符合要求的预测框的信息
            scores = scores[high_prob_idxs]
            xmins = xmins[high_prob_idxs]
            xmaxs = xmaxs[high_prob_idxs]
            ymins = ymins[high_prob_idxs]
            ymaxs = ymaxs[high_prob_idxs]
            df_idxs = df_idxs[high_prob_idxs]

            #对上述对象进行重叠框的非极大值抑制
            if nms_overlap_thresh > 0:
                boxes_nms_input = np.stack(
                    (xmins, ymins, xmaxs, ymaxs), axis=1)
                if use_weighted_nms:
                    probs = scores
                else:
                    probs = []
                good_idxs = non_max_suppression(
                    boxes_nms_input, probs=probs,
                    overlapThresh=nms_overlap_thresh)

                df_idxs = df_idxs[good_idxs]

            df_idxs_tot.extend(df_idxs)
            count += len(df_idxs)

    df_idxs_tot_final = np.unique(df_idxs_tot)

    # 创建数据帧
    df_out = df.loc[df_idxs_tot_final]

    t1 = time.time()
    print("数据帧，NMS处理前", len(df), "处理后:", len(df_out))
    print("refine_data()运行时长:", t1 - t0, "秒")
    return df_out  # refine_dic





def plot_detections(im, boxes,
                    scores=[],
                    classes=[],
                    outfile='',
                    plot_thresh=0.2,
                    color_dict={},
                    plot_line_thickness=1, show_labels=True,
                    compression_level=9,
                    skip_empty=False,
                    test_box_rescale_frac=1,
                    label_txt=None,
                    draw_rect=True, draw_circle=False,reduce=True):



    #预测框、字体等参数的设置
    font_size = 1.5
    font_width = 2
    display_str_height = 9
    font = cv2.FONT_HERSHEY_SIMPLEX

    output = im
    nboxes = 0


    alpha = 1
    for box, score, classy in zip(boxes, scores, classes):

        if score >= plot_thresh:
            nboxes += 1
            [xmin, ymin, xmax, ymax] = box
            left, right, top, bottom = xmin, xmax, ymin, ymax

            # 获得标签和对应的颜色
            classy_str = str(classy) + ': ' + \
                         str(int(100 * float(score))) + '%'
            color = color_dict[classy]


            #绘制矩形框
            if draw_rect:
                if reduce==True:
                    cv2.rectangle(output, (int(left/4), int(bottom/4)), (int(right/4),
                                                                     int(top/4)), color,
                                  plot_line_thickness)


                else:

                    cv2.rectangle(output, (int(left), int(bottom)), (int(right),
                                                                 int(top)), color,
                              plot_line_thickness)

            #绘制圆形框
            if draw_circle:  # 改一下，半径指定为定值，即下面的r
                d = max(abs(left - right), abs(top - bottom))  # linetype是调整线条粗细的，负值为实心
                # r = int(d/2.0)
                r = 2
                cx, cy = int((left + right) / 2.0), int((top + bottom) / 2.0)
                cv2.circle(output, (cx, cy), r, color, plot_line_thickness, lineType=-1)

            #是否显示标签(即预测框上面的类别)
            if show_labels:
                # 获取位置
                display_str = classy_str
                total_display_str_height = (1 + 2 * 0.05) * display_str_height
                if top > total_display_str_height:
                    text_bottom = top
                else:
                    text_bottom = bottom + total_display_str_height
                #反转列表并从下至上打印。
                (text_width, text_height), _ = cv2.getTextSize(display_str,
                                                               fontFace=font,
                                                               fontScale=font_size,
                                                               thickness=font_width)
                margin = np.ceil(0.1 * text_height)

                #获取矩形坐标和文本坐标，
                rect_top_left = (int(left - (plot_line_thickness - 1) * margin),
                                 int(text_bottom - text_height - (plot_line_thickness + 3) * margin))
                rect_bottom_right = (int(left + text_width + margin),
                                     int(text_bottom - (plot_line_thickness * margin)))
                text_loc = (int(left + margin),
                            int(text_bottom - (plot_line_thickness + 2) * margin))

                #需要显示标签时，以下语句可以让标签稍微下移一些
                if (alpha > 0.75) and (plot_line_thickness > 1):
                    rect_top_left = (rect_top_left[0], int(
                        rect_top_left[1] + margin))
                    rect_bottom_right = (rect_bottom_right[0], int(
                        rect_bottom_right[1] + margin))
                    text_loc = (text_loc[0], int(text_loc[1] + margin))

                if draw_rect:
                    cv2.rectangle(output, rect_top_left, rect_bottom_right,
                                  color, -1)
                cv2.putText(output, display_str, text_loc,
                            font, font_size, (0, 0, 0), font_width, cv2.LINE_AA)


    #调整输出图像大小
    if test_box_rescale_frac != 1:
        height, width = output.shape[:2]
        output = cv2.resize(output, (width / test_box_rescale_frac, height / test_box_rescale_frac),
                            interpolation=cv2.INTER_CUBIC)

    #如果需要，给图像添加txt文本，在text_loc_label位置写
    if label_txt:
        text_loc_label = (10, 20)
        cv2.putText(output, label_txt, text_loc_label,
                    font, 2 * font_size, (0, 0, 0), font_width,
                    cv2.LINE_AA)

    #保存压缩后的图片，compression_level为缩放系数
    if skip_empty and nboxes == 0:
        return
    else:
        print("结果图绘制完成，保存在:", outfile)
        cv2.imwrite(outfile, output, [
            cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return





#####################################关键函数在这里#####################################
#该函数用于将子图的标签文件拼贴并过滤，以生成相对于原图的总标签文件，最后再进行相关的绘图任务
#需要注意的是:本函数会将labels下所有txt文件写入同一个文件，在绘图时需要通过图名进行区分，例如xinmi的最后会倍用于拼xinmi的图片

def execute(labels_dir='',
            original='',
            out_results='out_results',
            csv_name='params.csv',
            subgraph_size=640,
            im_ext='.tif',
            ignore_names=[],
            classes=[],
            detect_thresh=0.2,      #置信度低于这个值会被去掉
            overlap_thresh=0.35,     #两个框重叠高于这个值会被去掉，越高去的越少
            max_edge_aspect_ratio=5,
            edge_buffer_test=0,
            groupby='image_path',
            groupby_cat='category',
            allow_nested_detections=True,

            ):


    #要改一些生成全局框的相关参数，找augment_data函数

    t0 = time.time()

    #拼出结果图片存放路径、拼出结果文件存放地址
    plot_path = out_results
    if os.path.exists(plot_path):
        print("输出路径已存在，将返回")
    else:
        print("输出路径不存在，将创建")
        os.makedirs(plot_path)
    csv_path = os.path.join(out_results, csv_name)
    subgraph_height = subgraph_size
    subgraph_width = subgraph_size

    data_list = []

    #遍历labes文件夹下的所有标签，把所有的内容取出来，方便后续处理
    for txt_name in sorted(z for z in os.listdir(labels_dir) if z.endswith('.txt')):
        txt_path = os.path.join(labels_dir, txt_name)
        prefix_name =txt_name.split('.txt')[0]

        #以csv格式读取当前txt文件，读取对象为txt_path
        data = pd.read_csv(txt_path, header=None, index_col=None, sep=' ',
                           names=['cat_int', 'x_frac', 'y_frac', 'w_frac', 'h_frac', 'prob'])

        #如果独到的txt文件里有东西，注意：此时读到的是单个txt文件，也即一个子图对应的标签文件
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
                pix_box = convert_reverse((subgraph_height, subgraph_width), box)
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

    #通过ignore_names忽略某些类对象,一般用不到这里
    if ignore_names:
        data_raw = data_raw[~data_raw['category'].isin(ignore_names)]

    #通过整合后的全部标签文件，拿到原始图片名称
    im_path_list = [os.path.join(original, im_name + im_ext) for
                    im_name in data_raw['im_name_root'].values]
    data_raw['image_path'] = im_path_list

    df_tiled_aug = augment_data(data_raw,                                       #传入数据帧
                    slice_sep='__',                                             #图名分割标志
                    edge_buffer_test=edge_buffer_test,                          #如果边界框与边的距离在此范围内，放弃
                    max_edge_aspect_ratio=max_edge_aspect_ratio,                #窗口边缘附近框的边界框的最大纵横比
                    test_box_rescale_frac=1.0,                                  #框缩放系数
                    max_box_size=300)                                           #允许的最大预测框像素值

    if allow_nested_detections:
        groupby_cat_refine = groupby_cat
    else:
        groupby_cat_refine = ''
    df_refine = refine_data(df_tiled_aug,
                          groupby=groupby,
                          groupby_cat=groupby_cat_refine,
                          nms_overlap_thresh=overlap_thresh,
                          plot_thresh=detect_thresh,
                          cats_to_ignore=ignore_names)


    #这里导出csv文件了，里面装着包含全局坐标在内的参数
    df_refine.to_csv(csv_path)




    print("进程结束，总耗时:", time.time()-t0, "秒")
    return

import pandas as pd
import numpy as np
import os
import cv2


def plot_detections(im, boxes,
                    scores=[],
                    classes=[],
                    outfile='',
                    plot_thresh=0.2,
                    color_dict={},
                    plot_line_thickness=1, show_labels=True,
                    compression_level=9,
                    skip_empty=False,
                    test_box_rescale_frac=1,
                    label_txt=None,
                    draw_rect=True, draw_circle=False,reduce=True):



    #预测框、字体等参数的设置
    font_size = 1.5
    font_width = 2
    display_str_height = 9
    font = cv2.FONT_HERSHEY_SIMPLEX

    output = im
    nboxes = 0


    alpha = 1
    for box, score, classy in zip(boxes, scores, classes):

        if score >= plot_thresh:
            nboxes += 1
            [xmin, ymin, xmax, ymax] = box
            left, right, top, bottom = xmin, xmax, ymin, ymax

            # 获得标签和对应的颜色
            classy_str = str(classy) + ': ' + \
                         str(int(100 * float(score))) + '%'
            color = color_dict[classy]


            #绘制矩形框
            if draw_rect:
                if reduce==True:
                    cv2.rectangle(output, (int(left/4), int(bottom/4)), (int(right/4),
                                                                     int(top/4)), color,
                                  plot_line_thickness)


                else:

                    cv2.rectangle(output, (int(left), int(bottom)), (int(right),
                                                                 int(top)), color,
                              plot_line_thickness)

            #绘制圆形框
            if draw_circle:  # 改一下，半径指定为定值，即下面的r
                d = max(abs(left - right), abs(top - bottom))  # linetype是调整线条粗细的，负值为实心
                # r = int(d/2.0)
                r = 2
                cx, cy = int((left + right) / 2.0), int((top + bottom) / 2.0)
                cv2.circle(output, (cx, cy), r, color, plot_line_thickness, lineType=-1)

            #是否显示标签(即预测框上面的类别)
            if show_labels:
                # 获取位置
                display_str = classy_str
                total_display_str_height = (1 + 2 * 0.05) * display_str_height
                if top > total_display_str_height:
                    text_bottom = top
                else:
                    text_bottom = bottom + total_display_str_height
                #反转列表并从下至上打印。
                (text_width, text_height), _ = cv2.getTextSize(display_str,
                                                               fontFace=font,
                                                               fontScale=font_size,
                                                               thickness=font_width)
                margin = np.ceil(0.1 * text_height)

                #获取矩形坐标和文本坐标，
                rect_top_left = (int(left - (plot_line_thickness - 1) * margin),
                                 int(text_bottom - text_height - (plot_line_thickness + 3) * margin))
                rect_bottom_right = (int(left + text_width + margin),
                                     int(text_bottom - (plot_line_thickness * margin)))
                text_loc = (int(left + margin),
                            int(text_bottom - (plot_line_thickness + 2) * margin))

                #需要显示标签时，以下语句可以让标签稍微下移一些
                if (alpha > 0.75) and (plot_line_thickness > 1):
                    rect_top_left = (rect_top_left[0], int(
                        rect_top_left[1] + margin))
                    rect_bottom_right = (rect_bottom_right[0], int(
                        rect_bottom_right[1] + margin))
                    text_loc = (text_loc[0], int(text_loc[1] + margin))

                if draw_rect:
                    cv2.rectangle(output, rect_top_left, rect_bottom_right,
                                  color, -1)
                cv2.putText(output, display_str, text_loc,
                            font, font_size, (0, 0, 0), font_width, cv2.LINE_AA)


    #调整输出图像大小
    if test_box_rescale_frac != 1:
        height, width = output.shape[:2]
        output = cv2.resize(output, (width / test_box_rescale_frac, height / test_box_rescale_frac),
                            interpolation=cv2.INTER_CUBIC)

    #如果需要，给图像添加txt文本，在text_loc_label位置写
    if label_txt:
        text_loc_label = (10, 20)
        cv2.putText(output, label_txt, text_loc_label,
                    font, 2 * font_size, (0, 0, 0), font_width,
                    cv2.LINE_AA)

    #保存压缩后的图片，compression_level为缩放系数
    if skip_empty and nboxes == 0:
        return
    else:
        print("结果图绘制完成，保存在:", outfile)
        cv2.imwrite(outfile, output, [
            cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return


def DrawAuxiliaryImages(df_refine_path,original_path,out_results_path,detect_thresh=0.2,im_ext='.png',reduce=True):
    """
    :param df_refine_path:   a csv file that store some key parameters for detection
    :param original_path:  original big image path
    :param out_results_path: save the drawed plot to this path (save the final result)
    :param detect_thresh:  lower this detection confidence score will be neglected
    :param im_ext: big image format
    :param reduce:  Four-dimensional downsampling to read in the image or not
    """
    df_refine=pd.read_csv(df_refine_path)
    colors = 40*[(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 140, 255),
                  (0, 255, 125), (125, 125, 125)]
    color_dict = {}
    for i, c in enumerate(sorted(np.unique(df_refine['category'].values))):
        color_dict[c] = colors[i]
    original=original_path
    plot_path = out_results_path
    # 获得输入图片的路径、根据标签文件生成的路径
    im_names_tiled = sorted([z.split(im_ext)[0] for z in os.listdir(original) if z.endswith(im_ext)])
    im_names_set = set(df_refine['im_name_root'].values)
    print(im_names_set)
    print(im_names_tiled)
    if len(im_names_set) == 0:
        print("路径中不存在标签对应的图片", original, "其后缀为:", im_ext)
        print("返回中")


    names = ['X', 'Y']
    for i, im_name in enumerate(im_names_tiled):

        im_path = os.path.join(original, im_name + im_ext)
        outfile_plot_image = os.path.join(plot_path, im_name + '.jpg')
        outfile_point = os.path.join(plot_path, im_name + '_coordinate' + '.csv')
        outfile_point_img = os.path.join(plot_path, im_name + '_point' + '.png')

        print("绘制图像:", im_name)

        if reduce == True:
            im_cv2 = cv2.imread(im_path, cv2.IMREAD_REDUCED_COLOR_4)
        else:
            im_cv2 = cv2.imread(im_path)

        x_pos = []
        y_pos = []
        X_Y = []

        if im_name not in im_names_set:
            boxes, probs, classes, box_names = [], [], [], []
        else:
            df_filt = df_refine[df_refine['im_name_root'] == im_name]
            boxes = df_filt[['Xmin_Glob', 'Ymin_Glob', 'Xmax_Glob', 'Ymax_Glob']].values
            probs = df_filt['prob']
            classes = df_filt['category']
            # 黑底图，后续用于显示植株点
            point_img = np.zeros((im_cv2.shape[0], im_cv2.shape[1]), dtype=np.uint8)
            print(point_img.shape)
            for box in boxes:
                [xmin, ymin, xmax, ymax] = box
                left, right, top, bottom = xmin, xmax, ymin, ymax
                # 计算中心点坐标
                cx, cy = int((left + right) / 2.0), int((top + bottom) / 2.0)

                if reduce==True:
                    cx, cy = int(cx / 4), int(cy / 4)
                point_img[cy, cx] = 255
                x_pos.append(cx)
                y_pos.append(cy)
                X_Y.append((cx, cy))

            out_paint = pd.DataFrame(columns=names, data=X_Y)
            out_paint.to_csv(outfile_point)
            print(point_img.shape)
            cv2.imwrite(outfile_point_img, point_img)

            plot_detections(im_cv2, boxes,
                            scores=probs,
                            classes=classes,
                            outfile=outfile_plot_image,
                            plot_thresh=detect_thresh,
                            color_dict=color_dict,
                            plot_line_thickness=1,
                            show_labels=False,
                            test_box_rescale_frac=1,
                            label_txt=None,
                            draw_rect=True, draw_circle=False,reduce=reduce)




if __name__ =='__main__':
    ##First step : slice the big image into small pieces
    slice_to_png(tif_path='/home/kingargroo/seed/ablation1/test_result/1', out_path='/home/kingargroo/seed/ablation1/test_result/2', in_ext='.jpg',
                 Height=640, Width=640)

    # # Second step: give the prediction using YOLOV8 or other detection model
    predict_with_yolov8(model_name='/home/kingargroo/corn/runs/detect/train22/weights/best.pt', img_path='/home/kingargroo/seed/test/out', save_path='/home/kingargroo/seed/test/out2/')

    # #Thith step: post-process according to the predicted results ,including NMS
    execute(labels_dir='/home/kingargroo/seed/test/out2/', original='/home/kingargroo/seed/test/millet', classes=['millet'],
            out_results='/home/kingargroo/seed/test/out3/', im_ext='.jpg',
            detect_thresh=0.1,
            overlap_thresh=0.3)

    # Forth step : visualize the result, draw the white dot in the black background and draw the detection bounding box in big image
    df_refine_path = '/home/kingargroo/seed/test/out3' + '/' + 'params.csv'
    DrawAuxiliaryImages(df_refine_path=df_refine_path, original_path='/home/kingargroo/seed/test/millet/',
                        out_results_path='/home/kingargroo/seed/test/out3/', reduce=False,
                        detect_thresh=0.1, im_ext='.jpg')