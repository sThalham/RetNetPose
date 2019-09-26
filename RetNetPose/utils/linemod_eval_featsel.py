"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#from pycocotools.cocoeval import COCOeval

import keras
import numpy as np
import json
import pyquaternion
import math
import transforms3d as tf3d
import geometry
import os
import copy
import cv2
import open3d
from ..utils import ply_loader
from .pose_error import reproj, add, adi, re, te, vsd
import time

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


# LineMOD
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899


threeD_boxes = np.ndarray((15, 8, 3), dtype=np.float32)
threeD_boxes[0, :, :] = np.array([[-0.0088, -0.0277, -0.01747],  # ape [76, 78, 92]
                                     [-0.0175, 0.012, -0.0168],
                                     [0.005, 0.0188, -0.0161],
                                     [0.0256, 0.0035, -0.0302],
                                     [0.0285, -0.0245, -0.0061],
                                     [0.0097, 0.0209, 0.0184],
                                     [0.0171, -0.0005, 0.0278],
                                     [-0.0363, -0.0096, -0.0159]])
threeD_boxes[1, :, :] = np.array([[0.1055, 0.0002, -0.086],  # benchvise [216, 122, 219]
                                     [-0.0471, -0.0305, -0.0908],
                                     [-0.1045, -0.0012, -0.0205],
                                     [-0.0481, 0.028, -0.0928],
                                     [-0.0191, 0.0, 0.0017],
                                     [0.0482, 0.0353, -0.0177],
                                     [0.0474, -0.0339, -0.0173],
                                     [0.0613, 0.0014, 0.1084]])
threeD_boxes[2, :, :] = np.array([[0.083, 0.0825, 0.037],  # bowl [166, 165, 74]
                                     [0.083, 0.0825, -0.037],
                                     [0.083, -0.0825, -0.037],
                                     [0.083, -0.0825, 0.037],
                                     [-0.083, 0.0825, 0.037],
                                     [-0.083, 0.0825, -0.037],
                                     [-0.083, -0.0825, -0.037],
                                     [-0.083, -0.0825, 0.037]])
threeD_boxes[3, :, :] = np.array([[0.0681, 0.0168, -0.01],  # camera [137, 143, 100]
                                     [-0.011, 0.0612, -0.0094],
                                     [-0.0641, 0.019, 0.0246],
                                     [-0.0661, -0.0226, -0.0216],
                                     [-0.0619, -0.0657, 0.0181],
                                     [-0.0018, -0.0459, 0.0264],
                                     [0.0104, 0.0174, 0.0411],
                                     [-0.048, 0.018, 0.0479]])
threeD_boxes[4, :, :] = np.array([[-0.0072, -0.083, 0.0162],  # can [101, 182, 194]
                                     [0.0482, 0.0065, 0.0203],
                                     [0.0047, 0.0681, -0.0662],
                                     [-0.0473, 0.0095, 0.0197],
                                     [0.0074, 0.0778, 0.0087],
                                     [0.0009, 0.0107, 0.0968],
                                     [-0.0039, -0.0417, -0.0649],
                                     [0.0, 0.0, 0.0]])
threeD_boxes[5, :, :] = np.array([[0.0019, -0.0564, 0.0248],  # cat [67, 128, 117]
                                     [-0.0222, 0.0627, 0.0263],
                                     [0.0128, 0.0121, 0.0585],
                                     [0.0317, 0.0315, -0.0002],
                                     [-0.0197, 0.0318, -0.0475],
                                     [0.022, 0.0322, -0.0484],
                                     [-0.0256, -0.0588, -0.0439],
                                     [0.0204, -0.0623, -0.0431]])
threeD_boxes[6, :, :] = np.array([[0.059, 0.046, 0.0475],  # mug [118, 92, 95]
                                     [0.059, 0.046, -0.0475],
                                     [0.059, -0.046, -0.0475],
                                     [0.059, -0.046, 0.0475],
                                    [-0.059, 0.046, 0.0475],
                                    [-0.059, 0.046, -0.0475],
                                     [-0.059, -0.046, -0.0475],
                                     [-0.059, -0.046, 0.0475]])
threeD_boxes[7, :, :] = np.array([[-0.1146, -0.0005, 0.04],  # drill [118, 92, 95]
                                     [-0.0599, 0.0007, -0.0885],
                                     [0.0302, 0.0017, -0.0842],
                                     [0.1145, -0.0027, 0.0889],
                                    [0.018, 0.0004, 0.0033],
                                    [-0.0354, -0.0026, 0.0895],
                                     [0.0049, -0.031, 0.035],
                                     [0.0037, 0.0214, 0.0346]])
threeD_boxes[8, :, :] = np.array([[0.0439, 0.0009, 0.0227],  # duck [104, 77, 86]
                                     [-0.0155, -0.0012, 0.0364],
                                     [-0.0425, -0.0014, 0.0012],
                                     [-0.0077, -0.0363, -0.0207],
                                     [0.025, 0.0011, 0.0346],
                                     [0.0514, 0.0021, -0.0264],
                                     [-0.0078, 0.037, -0.0222],
                                     [-0.0087, 0.0001, -0.0009]])
threeD_boxes[9, :, :] = np.array([[0.0018, -0.0513, 0.0172],  # eggbox [150, 107, 69]
                                     [0.049, -0.0011, 0.0331],
                                     [-0.0668, 0.0465, 0.0052],
                                     [-0.0694, -0.039, -0.0023],
                                     [0.0692, -0.0419, -0.0007],
                                     [0.0685, 0.0416, 0.0054],
                                     [0.001, 0.0528, 0.0052],
                                     [-0.0466, 0.002, 0.0331]])
threeD_boxes[10, :, :] = np.array([[0.0004, 0.0025, 0.0863],  # glue [37, 78, 173]
                                     [-0.0012, 0.0132, 0.053],
                                     [0.0133, -0.021, 0.0197],
                                     [-0.0146, -0.0209, 0.0201],
                                     [0.0114, -0.0365, -0.0509],
                                     [0.0155, 0.0347, -0.0549],
                                     [-0.0151, 0.0354, -0.0544],
                                     [-0.0146, -0.0351, -0.0546]])
threeD_boxes[11, :, :] = np.array([[-0.0376, 0.0457, -0.0349],  # holepuncher [101, 108, 91]
                                     [-0.0344, -0.0465, -0.0342],
                                     [0.024, -0.0405, -0.0257],
                                     [0.024, 0.0444, -0.0264],
                                     [0.0388, 0.048, 0.037],
                                     [0.0376, -0.0437, 0.0376],
                                     [-0.0019, 0.001, 0.0262],
                                     [-0.0477, 0.0024, -0.0005]])
threeD_boxes[12, :, :] = np.array([[-0.1127, -0.0005, 0.0691],  # drill [230, 76, 208]
                                     [-0.1291, -0.039, -0.0372],
                                     [-0.1275, 0.0434, -0.0373],
                                     [-0.0052, -0.0023, 0.0104],
                                     [0.0777, -0.0021, 0.042],
                                     [0.1269, -0.0021, -0.0554],
                                     [0.0533, -0.0425, -0.0443],
                                     [0.0453, 0.0417, -0.0419]])
threeD_boxes[13, :, :] = np.array([[-0.0694, 0.0013, 0.0809],  # iron [258, 118, 141]
                                     [0.0098, 0.0318, 0.0053],
                                     [0.0099, -0.0244, 0.0032],
                                     [0.0639, -0.0482, 0.0774],
                                     [0.0647, 0.0465, 0.0789],
                                     [0.0143, 0.0512, -0.0846],
                                     [0.0154, -0.0413, -0.089],
                                     [-0.0356, 0.0046, -0.0876]])
threeD_boxes[14, :, :] = np.array([[0.0272, 0.0281, -0.0477],  # phone [94, 147, 185]
                                     [0.0243, -0.0393, -0.0492],
                                     [0.0023, -0.037, 0.0394],
                                     [-0.0393, 0.0177, -0.0239],
                                     [-0.0374, 0.0540, -0.0816],
                                     [-0.0444, -0.047, -0.0839],
                                     [-0.0233, -0.0676, -0.0173],
                                     [-0.0236, 0.0656, -0.0139]])

#model_radii = np.array([0.041, 0.0928, 0.0675, 0.0633, 0.0795, 0.052, 0.0508, 0.0853, 0.0445, 0.0543, 0.048, 0.05, 0.0862, 0.0888, 0.071])
#model_radii = np.array([0.0515, 0.143454, 0.0675, 0.0865, 0.101, 0.0775, 0.0508, 0.131, 0.545, 0.88182, 0.088, 0.081, 0.1515765, 0.1425775, 0.1065])
model_dia = np.array([0.10209865663, 0.24750624233, 0.16735486092, 0.17249224865, 0.20140358597, 0.15454551808, 0.12426430816, 0.26147178102, 0.10899920102, 0.16462758848, 0.17588933422, 0.14554287471, 0.27807811733, 0.28260129399, 0.21235825148])


def get_evaluation(pcd_temp_, pcd_scene_, inlier_thres, tf, final_th=0, n_iter=5):#queue
    tf_pcd =np.eye(4)

    reg_p2p = open3d.registration_icp(pcd_temp_, pcd_scene_ , inlier_thres, np.eye(4),
              open3d.TransformationEstimationPointToPoint(),
              open3d.ICPConvergenceCriteria(max_iteration=1)) #5?
    tf = np.matmul(reg_p2p.transformation, tf)
    tf_pcd = np.matmul(reg_p2p.transformation,tf_pcd)
    pcd_temp_.transform(reg_p2p.transformation)

    for i in range(4):
        inlier_thres = reg_p2p.inlier_rmse*3
        if inlier_thres == 0:
            continue

        reg_p2p = open3d.registration_icp(pcd_temp_,pcd_scene_ , inlier_thres, np.eye(4),
                  open3d.TransformationEstimationPointToPlane(),
                  open3d.ICPConvergenceCriteria(max_iteration=1)) #5?
        tf = np.matmul(reg_p2p.transformation, tf)
        tf_pcd = np.matmul(reg_p2p.transformation, tf_pcd)
        pcd_temp_.transform(reg_p2p.transformation)
    inlier_rmse = reg_p2p.inlier_rmse

    ##Calculate fitness with depth_inlier_th
    if(final_th>0):
        inlier_thres = final_th #depth_inlier_th*2 #reg_p2p.inlier_rmse*3
        reg_p2p = registration_icp(pcd_temp_,pcd_scene_, inlier_thres, np.eye(4),
                  TransformationEstimationPointToPlane(),
                  ICPConvergenceCriteria(max_iteration = 1)) #5?

    if( np.abs(np.linalg.det(tf[:3,:3])-1)>0.001):
        tf[:3,0]=tf[:3,0]/np.linalg.norm(tf[:3,0])
        tf[:3,1]=tf[:3,1]/np.linalg.norm(tf[:3,1])
        tf[:3,2]=tf[:3,2]/np.linalg.norm(tf[:3,2])
    if( np.linalg.det(tf) < 0) :
        tf[:3,2]=-tf[:3,2]

    return tf,inlier_rmse,tf_pcd,reg_p2p.fitness


def toPix_array(translation):

    xpix = ((translation[:, 0] * fxkin) / translation[:, 2]) + cxkin
    ypix = ((translation[:, 1] * fykin) / translation[:, 2]) + cykin
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


def load_pcd(cat):
    # load meshes
    mesh_path = "/RetNetPose/models_ply/"
    #mesh_path = "/home/stefan/data/val_linemod_cc_rgb/models_ply/"
    ply_path = mesh_path + 'obj_' + cat + '.ply'
    model_vsd = ply_loader.load_ply(ply_path)
    pcd_model = open3d.PointCloud()
    pcd_model.points = open3d.Vector3dVector(model_vsd['pts'])
    open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    # open3d.draw_geometries([pcd_model])
    model_vsd_mm = copy.deepcopy(model_vsd)
    model_vsd_mm['pts'] = model_vsd_mm['pts'] * 1000.0
    pcd_model = open3d.read_point_cloud(ply_path)

    return pcd_model, model_vsd, model_vsd_mm


def create_point_cloud(depth, ds):

    rows, cols = depth.shape

    depRe = depth.reshape(rows * cols)
    zP = np.multiply(depRe, ds)

    x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1), indexing='xy')
    yP = y.reshape(rows * cols) - cykin
    xP = x.reshape(rows * cols) - cxkin
    yP = np.multiply(yP, zP)
    xP = np.multiply(xP, zP)
    yP = np.divide(yP, fykin)
    xP = np.divide(xP, fxkin)

    cloud_final = np.transpose(np.array((xP, yP, zP)))

    return cloud_final


def boxoverlap(a, b):
    a = np.array([a[0], a[1], a[0] + a[2], a[1] + a[3]])
    b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])

    x1 = np.amax(np.array([a[0], b[0]]))
    y1 = np.amax(np.array([a[1], b[1]]))
    x2 = np.amin(np.array([a[2], b[2]]))
    y2 = np.amin(np.array([a[3], b[3]]))

    wid = x2-x1+1
    hei = y2-y1+1
    inter = wid * hei
    aarea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    # intersection over union overlap
    ovlap = inter / (aarea + barea - inter)
    # set invalid entries to 0 overlap
    maskwid = wid <= 0
    maskhei = hei <= 0
    np.where(ovlap, maskwid, 0)
    np.where(ovlap, maskhei, 0)

    return ovlap


def evaluate_linemod_featsel(generator, model, threshold=0.05):
    threshold = 0.5
    """ Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        generator : The generator for generating the evaluation data.
        model     : The model to evaluate.
        threshold : The score threshold to use.
    """
    # start collecting results
    results = []
    image_ids = []
    image_indices = []
    idx = 0

    tp = np.zeros((16), dtype=np.uint32)
    fp = np.zeros((16), dtype=np.uint32)
    fn = np.zeros((16), dtype=np.uint32)

    # interlude
    tp55 = np.zeros((16), dtype=np.uint32)
    fp55 = np.zeros((16), dtype=np.uint32)
    fn55 = np.zeros((16), dtype=np.uint32)

    tp6 = np.zeros((16), dtype=np.uint32)
    fp6 = np.zeros((16), dtype=np.uint32)
    fn6 = np.zeros((16), dtype=np.uint32)

    tp65 = np.zeros((16), dtype=np.uint32)
    fp65 = np.zeros((16), dtype=np.uint32)
    fn65 = np.zeros((16), dtype=np.uint32)

    tp7 = np.zeros((16), dtype=np.uint32)
    fp7 = np.zeros((16), dtype=np.uint32)
    fn7 = np.zeros((16), dtype=np.uint32)

    tp75 = np.zeros((16), dtype=np.uint32)
    fp75 = np.zeros((16), dtype=np.uint32)
    fn75 = np.zeros((16), dtype=np.uint32)

    tp8 = np.zeros((16), dtype=np.uint32)
    fp8 = np.zeros((16), dtype=np.uint32)
    fn8 = np.zeros((16), dtype=np.uint32)

    tp85 = np.zeros((16), dtype=np.uint32)
    fp85 = np.zeros((16), dtype=np.uint32)
    fn85 = np.zeros((16), dtype=np.uint32)

    tp9 = np.zeros((16), dtype=np.uint32)
    fp9 = np.zeros((16), dtype=np.uint32)
    fn9 = np.zeros((16), dtype=np.uint32)

    tp925 = np.zeros((16), dtype=np.uint32)
    fp925 = np.zeros((16), dtype=np.uint32)
    fn925 = np.zeros((16), dtype=np.uint32)

    tp95 = np.zeros((16), dtype=np.uint32)
    fp95 = np.zeros((16), dtype=np.uint32)
    fn95 = np.zeros((16), dtype=np.uint32)

    tp975 = np.zeros((16), dtype=np.uint32)
    fp975 = np.zeros((16), dtype=np.uint32)
    fn975 = np.zeros((16), dtype=np.uint32)
    # interlude end

    tp_add = np.zeros((16), dtype=np.uint32)
    fp_add = np.zeros((16), dtype=np.uint32)
    fn_add = np.zeros((16), dtype=np.uint32)

    rotD = np.zeros((16), dtype=np.uint32)
    less5 = np.zeros((16), dtype=np.uint32)
    rep_e = np.zeros((16), dtype=np.uint32)
    rep_less5 = np.zeros((16), dtype=np.uint32)
    add_e = np.zeros((16), dtype=np.uint32)
    add_less_d = np.zeros((16), dtype=np.uint32)
    vsd_e = np.zeros((16), dtype=np.uint32)
    vsd_less_t = np.zeros((16), dtype=np.uint32)

    model_pre = []

    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):
        image_raw = generator.load_image(index)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        anno = generator.load_annotations(index)
        if len(anno['labels']) > 1:
            t_cat = 2
            obj_name = '02'
            ent = np.where(anno['labels'] == 1.0)
            t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[ent][0]
            t_tra = anno['poses'][ent][0][:3]
            t_rot = anno['poses'][ent][0][3:]

        else:
            t_cat = int(anno['labels']) + 1
            obj_name = str(t_cat)
            if len(obj_name) < 2:
                obj_name = '0' + obj_name
            t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[0]
            t_tra = anno['poses'][0][:3]
            t_rot = anno['poses'][0][3:]

        #if t_cat != 2:
        #    continue

        if t_cat == 3 or t_cat == 7:
            print(t_cat, ' ====> skip')
            continue

        # run network
        boxes, boxes3D, scores, labels, feat_vises = model.predict_on_batch(np.expand_dims(image, axis=0))

        # correct boxes for image scale
        boxes /= scale

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # target annotation

        if obj_name != model_pre:
            point_cloud, model_vsd, model_vsd_mm = load_pcd(obj_name)
            model_pre = obj_name

        rotD[t_cat] += 1
        rep_e[t_cat] += 1
        add_e[t_cat] += 1
        vsd_e[t_cat] += 1
        #t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[0]
        #t_tra = anno['poses'][0][:3]
        #t_rot = anno['poses'][0][3:]
        fn[t_cat] += 1
        #interlude
        fn55[t_cat] += 1
        fn6[t_cat] += 1
        fn65[t_cat] += 1
        fn7[t_cat] += 1
        fn75[t_cat] += 1
        fn8[t_cat] += 1
        fn85[t_cat] += 1
        fn9[t_cat] += 1
        fn925[t_cat] += 1
        fn95[t_cat] += 1
        fn975[t_cat] += 1

        # end interlude
        fn_add[t_cat] += 1
        fnit = True

        # compute predicted labels and scores
        for box, box3D, score, label, feat_vis in zip(boxes[0], boxes3D[0], scores[0], labels[0], feat_vises[0]):
            # scores are sorted, so we can break
            if score < threshold:
                continue

            if label < 0:
                continue


            cls = generator.label_to_inv_label(label)
            #cls = 1
            #control_points = box3D[(cls - 1), :]
            #control_points = box3D[0, :]
            control_points = box3D

            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.image_ids[index],
                'category_id' : generator.label_to_inv_label(label),
                'score'       : float(score),
                'bbox'        : box.tolist(),
                'pose'        : control_points.tolist()
            }

            # append detection to results
            results.append(image_result)

            #if cls > 5:
            #    cls = cls + 2
            #elif cls > 2:
            #    cls = cls + 1
            #else:
            #    pass

            if cls == t_cat:
                b1 = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                b2 = np.array([t_bbox[0], t_bbox[1], t_bbox[2], t_bbox[3]])
                IoU = boxoverlap(b1, b2)
                # occurences of 2 or more instances not possible in LINEMOD
                if IoU > 0.5:
                    if fnit is True:
                        # interlude
                        if IoU > 0.55:
                            tp55[t_cat] += 1
                            fn55[t_cat] -= 1
                        else:
                            fp55[t_cat] += 1
                        if IoU > 0.6:
                            tp6[t_cat] += 1
                            fn6[t_cat] -= 1
                        else:
                            fp6[t_cat] += 1
                        if IoU > 0.65:
                            tp65[t_cat] += 1
                            fn65[t_cat] -= 1
                        else:
                            fp65[t_cat] += 1
                        if IoU > 0.7:
                            tp7[t_cat] += 1
                            fn7[t_cat] -= 1
                        else:
                            fp7[t_cat] += 1
                        if IoU > 0.75:
                            tp75[t_cat] += 1
                            fn75[t_cat] -= 1
                        else:
                            fp75[t_cat] += 1
                        if IoU > 0.8:
                            tp8[t_cat] += 1
                            fn8[t_cat] -= 1
                        else:
                            fp8[t_cat] += 1
                        if IoU > 0.85:
                            tp85[t_cat] += 1
                            fn85[t_cat] -= 1
                        else:
                            fp85[t_cat] += 1
                        if IoU > 0.9:
                            tp9[t_cat] += 1
                            fn9[t_cat] -= 1
                        else:
                            fp9[t_cat] += 1
                        if IoU > 0.925:
                            tp925[t_cat] += 1
                            fn925[t_cat] -= 1
                        else:
                            fp925[t_cat] += 1
                        if IoU > 0.95:
                            tp95[t_cat] += 1
                            fn95[t_cat] -= 1
                        else:
                            fp95[t_cat] += 1
                        if IoU > 0.975:
                            tp975[t_cat] += 1
                            fn975[t_cat] -= 1
                        else:
                            fp975[t_cat] += 1

                        # interlude end

                        tp[t_cat] += 1
                        fn[t_cat] -= 1
                        fnit = False

                        obj_points = np.ascontiguousarray(threeD_boxes[cls-1, :, :], dtype=np.float32) #.reshape((8, 1, 3))
                        est_points = np.ascontiguousarray(control_points.T, dtype=np.float32).reshape((8, 1, 2))

                        K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

                        #retval, orvec, otvec = cv2.solvePnP(obj_points, est_points, K, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
                        retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                                           imagePoints=est_points, cameraMatrix=K,
                                                                           distCoeffs=None, rvec=None, tvec=None,
                                                                           useExtrinsicGuess=False, iterationsCount=100,
                                                                           reprojectionError=5.0, confidence=0.99,
                                                                           flags=cv2.SOLVEPNP_ITERATIVE)

                        R_est, _ = cv2.Rodrigues(orvec)
                        t_est = otvec

                        rot = tf3d.quaternions.mat2quat(R_est)
                        #pose = np.concatenate(
                        #            (np.array(t_est[:, 0], dtype=np.float32), np.array(rot, dtype=np.float32)), axis=0)


                        t_rot = tf3d.euler.euler2mat(t_rot[0], t_rot[1], t_rot[2])
                        R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                        t_gt = np.array(t_tra, dtype=np.float32) * 0.001

                        rd = re(R_gt, R_est)
                        xyz = te(t_gt, t_est.T)
                        #print(control_points)

                        #tDbox = R_gt.dot(obj_points.T).T
                        #tDbox = tDbox + np.repeat(t_gt[np.newaxis, :], 8, axis=0)
                        #box3D = toPix_array(tDbox)
                        #tDbox = np.reshape(box3D, (16))
                        #print(tDbox)

                        '''
                        colR = 242
                        colG = 119
                        colB = 25

                        colR1 = 242
                        colG1 = 119
                        colB1 = 25

                        colR2 = 242
                        colG2 = 119
                        colB2 = 25

                        colR3 = 65
                        colG3 = 102
                        colB3 = 245

                        colR4 = 65
                        colG4 = 102
                        colB4 = 245

                        colR5 = 65
                        colG5 = 102
                        colB5 = 245

                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (255, 255, 255), 5)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (255, 255, 255), 5)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (255, 255, 255),
                           5)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (255, 255, 255),
                           5)
                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (255, 255, 255),
                           5)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), (255, 255, 255),
                           5)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), (255, 255, 255),
                           5)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), (255, 255, 255),
                           5)
                        img = cv2.line(img, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()),
                           (255, 255, 255),
                           5)
                        img = cv2.line(img, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()),
                           (255, 255, 255),
                           5)
                        img = cv2.line(img, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()),
                           (255, 255, 255),
                           5)
                        img = cv2.line(img, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()),
                           (255, 255, 255),
                           5)

                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (colR, colG, colB), 4)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (colR, colG, colB), 4)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (colR1, colG1, colB1), 4)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (colR1, colG1, colB1), 4)
                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (colR2, colG2, colB2), 4)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), (colR2, colG2, colB2), 4)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), (colR5, colG5, colB5), 4)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), (colR5, colG5, colB5), 4)
                        img = cv2.line(img, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), (colR3, colG3, colB3),
                           4)
                        img = cv2.line(img, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), (colR3, colG3, colB3),
                           4)
                        img = cv2.line(img, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), (colR4, colG4, colB4),
                           4)
                        img = cv2.line(img, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), (colR4, colG4, colB4),
                           4)

                        font = cv2.FONT_HERSHEY_COMPLEX
                        bottomLeftCornerOfText = (int(bb[0]) + 5, int(bb[1]) + int(bb[3]) - 5)
                        fontScale = 0.5
                        fontColor = (25, 215, 250)
                        fontthickness = 2
                        lineType = 2

                        if detCats[i] == 1:
                            cate = 'Ape'
                        elif detCats[i] == 2:
                            cate = 'Benchvise'
                        elif detCats[i] == 3:
                            cate = 'Bowl'
                        elif detCats[i] == 4:
                            cate = 'Camera'
                        elif detCats[i] == 5:
                            cate = 'Can'
                        elif detCats[i] == 6:
                            cate = 'Cat'
                        elif detCats[i] == 7:
                            cate = 'Cup'
                        elif detCats[i] == 8:
                            cate = 'Driller'
                        elif detCats[i] == 9:
                            cate = 'Duck'
                        elif detCats[i] == 10:
                            cate = 'Eggbox'
                        elif detCats[i] == 11:
                            cate = 'Glue'
                        elif detCats[i] == 12:
                            cate = 'Holepuncher'
                        elif detCats[i] == 13:
                            cate = 'Iron'
                        elif detCats[i] == 14:
                            cate = 'Lamp'
                        elif detCats[i] == 15:
                            cate = 'Phone'
                        gtText = cate
                        # gtText = cate + " / " + str(detSco[i])

                        fontColor2 = (0, 0, 0)
                        fontthickness2 = 4
                        cv2.putText(img, gtText,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor2,
                            fontthickness2,
                            lineType)

                        cv2.putText(img, gtText,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            fontthickness,
                            lineType)

                        name = '/home/sthalham/visTests/detected.jpg'
                        img_con = np.concatenate((img, img_gt), axis=1)
                        cv2.imwrite(name, img_con)
                        name_est = '/home/sthalham/visTests/detected_est.jpg'
                        cv2.imwrite(name_est, img_con)

                        '''

                        #for i in range(0,8):
                        #    cv2.circle(image, (control_points[2*i], control_points[2*i+1]), 1, (0, 255, 0), thickness=3)
                        #    cv2.circle(image, (tDbox[2 * i], tDbox[2 * i + 1]), 1, (255, 0, 0), thickness=3)

                        #cv2.imwrite('/home/sthalham/inRetNetPose.jpg', image)

                        if not math.isnan(rd):
                            if rd < 5.0 and xyz < 0.05:
                                less5[t_cat] += 1

                        #err_vsd = vsd(R_est, t_est * 1000.0, R_gt, t_gt * 1000.0, model_vsd_mm, image_dep, K, 0.3, 20.0)
                        #if not math.isnan(err_vsd):
                        #    if err_vsd < 0.3:
                        #        vsd_less_t[t_cat] += 1

                        err_repr = reproj(K, R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                        if not math.isnan(err_repr):
                            if err_repr < 5.0:
                                rep_less5[t_cat] += 1

                        #if cls == 3 or cls == 7 or cls == 10 or cls == 11:
                        #    err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                        #else:
                        err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                        print(err_add)

                        if not math.isnan(err_add):
                            if err_add < (model_dia[cls - 1] * 0.1):
                                add_less_d[t_cat] += 1

                        if not math.isnan(err_add):
                            if err_add < (model_dia[cls - 1] * 0.15):
                                tp_add[t_cat] += 1
                                fn_add[t_cat] -= 1

                else:
                    fp[t_cat] += 1
                    fp_add[t_cat] += 1

                    fp55[t_cat] += 1
                    fp6[t_cat] += 1
                    fp65[t_cat] += 1
                    fp7[t_cat] += 1
                    fp75[t_cat] += 1
                    fp8[t_cat] += 1
                    fp85[t_cat] += 1
                    fp9[t_cat] += 1
                    fp925[t_cat] += 1
                    fp95[t_cat] += 1
                    fp975[t_cat] += 1

                print('Stop')

        # append image to list of processed images
        image_ids.append(generator.image_ids[index])
        image_indices.append(index)
        idx += 1

    print(len(image_ids))

    if not len(results):
        return

    # write output
    json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
    #json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

    detPre = [0.0] * 16
    detRec = [0.0] * 16
    detPre_add = [0.0] * 16
    detRec_add = [0.0] * 16
    F1_add = [0.0] * 16
    less_55 = [0.0] * 16
    less_repr_5 = [0.0] * 16
    less_add_d = [0.0] * 16
    less_vsd_t = [0.0] * 16

    np.set_printoptions(precision=2)
    print('')
    for ind in range(1, 16):
        if ind == 0:
            continue

        if tp[ind] == 0:
            detPre[ind] = 0.0
            detRec[ind] = 0.0
            detPre_add[ind] = 0.0
            detRec_add[ind] = 0.0
            less_55[ind] = 0.0
            less_repr_5[ind] = 0.0
            less_add_d[ind] = 0.0
            less_vsd_t[ind] = 0.0
        else:
            detRec[ind] = tp[ind] / (tp[ind] + fn[ind]) * 100.0
            detPre[ind] = tp[ind] / (tp[ind] + fp[ind]) * 100.0
            detRec_add[ind] = tp_add[ind] / (tp_add[ind] + fn_add[ind]) * 100.0
            detPre_add[ind] = tp_add[ind] / (tp_add[ind] + fp_add[ind]) * 100.0
            F1_add[ind] = 2 * ((detPre_add[ind] * detRec_add[ind])/(detPre_add[ind] + detRec_add[ind]))
            less_55[ind] = (less5[ind]) / (rotD[ind]) * 100.0
            less_repr_5[ind] = (rep_less5[ind]) / (rep_e[ind]) * 100.0
            less_add_d[ind] = (add_less_d[ind]) / (add_e[ind]) * 100.0
            less_vsd_t[ind] = (vsd_less_t[ind]) / (vsd_e[ind]) * 100.0

        print('cat ', ind, ' rec ', detPre[ind], ' pre ', detRec[ind], ' less5 ', less_55[ind], ' repr ',
                  less_repr_5[ind], ' add ', less_add_d[ind], ' vsd ', less_vsd_t[ind], ' F1 add 0.15d ', F1_add[ind])

    dataset_recall = sum(tp) / (sum(tp) + sum(fp)) * 100.0
    dataset_precision = sum(tp) / (sum(tp) + sum(fn)) * 100.0
    dataset_recall_add = sum(tp_add) / (sum(tp_add) + sum(fp_add)) * 100.0
    dataset_precision_add = sum(tp_add) / (sum(tp_add) + sum(fn_add)) * 100.0
    F1_add_all = 2 * ((dataset_precision_add * dataset_recall_add)/(dataset_precision_add + dataset_recall_add))
    less_55 = sum(less5) / sum(rotD) * 100.0
    less_repr_5 = sum(rep_less5) / sum(rep_e) * 100.0
    less_add_d = sum(add_less_d) / sum(add_e) * 100.0
    less_vsd_t = sum(vsd_less_t) / sum(vsd_e) * 100.0

    print('IoU 05: ', sum(tp) / (sum(tp) + sum(fp)) * 100.0, sum(tp) / (sum(tp) + sum(fn)) * 100.0)
    print('IoU 055: ', sum(tp55) / (sum(tp55) + sum(fp55)) * 100.0, sum(tp55) / (sum(tp55) + sum(fn55)) * 100.0)
    print('IoU 06: ', sum(tp6) / (sum(tp6) + sum(fp6)) * 100.0, sum(tp6) / (sum(tp6) + sum(fn6)) * 100.0)
    print('IoU 065: ', sum(tp65) / (sum(tp65) + sum(fp65)) * 100.0, sum(tp65) / (sum(tp65) + sum(fn65)) * 100.0)
    print('IoU 07: ', sum(tp7) / (sum(tp7) + sum(fp7)) * 100.0, sum(tp7) / (sum(tp7) + sum(fn7)) * 100.0)
    print('IoU 075: ', sum(tp75) / (sum(tp75) + sum(fp75)) * 100.0, sum(tp75) / (sum(tp75) + sum(fn75)) * 100.0)
    print('IoU 08: ', sum(tp8) / (sum(tp8) + sum(fp8)) * 100.0, sum(tp8) / (sum(tp8) + sum(fn8)) * 100.0)
    print('IoU 085: ', sum(tp85) / (sum(tp85) + sum(fp85)) * 100.0, sum(tp85) / (sum(tp85) + sum(fn85)) * 100.0)
    print('IoU 09: ', sum(tp9) / (sum(tp9) + sum(fp9)) * 100.0, sum(tp9) / (sum(tp9) + sum(fn9)) * 100.0)
    print('IoU 0975: ', sum(tp925) / (sum(tp925) + sum(fp925)) * 100.0, sum(tp925) / (sum(tp925) + sum(fn925)) * 100.0)
    print('IoU 095: ', sum(tp95) / (sum(tp95) + sum(fp95)) * 100.0, sum(tp95) / (sum(tp95) + sum(fn95)) * 100.0)
    print('IoU 0975: ', sum(tp975) / (sum(tp975) + sum(fp975)) * 100.0, sum(tp975) / (sum(tp975) + sum(fn975)) * 100.0)

    print('rec: ', dataset_recall)
    print('pre: ', dataset_precision)
    print('repr: ', less_repr_5)
    print('add: ', less_add_d)
    print('F1: ', F1_add_all)

    return dataset_recall, dataset_precision, less_55, less_vsd_t, less_repr_5, less_add_d, F1_add_all
