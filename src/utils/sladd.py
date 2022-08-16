import os
import dlib
import numpy as np
import math
import cv2
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import measure

# predictor_path = 'shape_predictor_81_face_landmarks.dat'

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(predictor_path)
# landmark_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 78, 74, 79, 73, 72, 80, 71, 70, 69, 68, 76, 75, 77]

def generate_random_mask(mask, res=256):
    randwl = np.random.randint(10, 60)
    randwr = np.random.randint(10, 60)
    randhu = np.random.randint(10, 60)
    randhd = np.random.randint(10, 60)
    newmask = np.zeros(mask.shape)
    mask = np.where(mask > 0.1, 1, 0)
    props = measure.regionprops(mask)
    if len(props) == 0:
        return newmask
    center_x, center_y = props[0].centroid
    center_x = int(round(center_x))
    center_y = int(round(center_y))
    newmask[max(center_x-randwl, 0):min(center_x+randwr, res-1), max(center_y-randhu, 0):min(center_x+randhd, res-1)]=1
    newmask *= mask
    return newmask

def random_deform(mask, nrows, ncols, mean=0, std=10):
    h, w = mask.shape[:2]
    rows = np.linspace(0, h-1, nrows).astype(np.int32)
    cols = np.linspace(0, w-1, ncols).astype(np.int32)
    rows += np.random.normal(mean, std, size=rows.shape).astype(np.int32)
    rows += np.random.normal(mean, std, size=cols.shape).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])

    trans = PiecewiseAffineTransform()
    trans.estimate(anchors, deformed.astype(np.int32))
    warped = warp(mask, trans)
    warped *= mask
    blured = cv2.GaussianBlur(warped, (5, 5), 3)
    return blured

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_five_key(landmarks_68):
    # get the five key points by using the landmarks
    leye_center = (landmarks_68[36] + landmarks_68[39])*0.5
    reye_center = (landmarks_68[42] + landmarks_68[45])*0.5
    nose = landmarks_68[33]
    lmouth = landmarks_68[48]
    rmouth = landmarks_68[54]
    leye_left = landmarks_68[36]
    leye_right = landmarks_68[39]
    reye_left = landmarks_68[42]
    reye_right = landmarks_68[45]
    out = [ tuple(x.astype('int32')) for x in [
        leye_center,reye_center,nose,lmouth,rmouth,leye_left,leye_right,reye_left,reye_right
    ]]
    return out

def remove_eyes(image, landmarks, opt):
    ##l: left eye; r: right eye, b: both eye
    if opt == 'l':
        (x1, y1), (x2, y2) = landmarks[5:7]
    elif opt == 'r':
        (x1, y1), (x2, y2) = landmarks[7:9]
    elif opt == 'b':
        (x1, y1), (x2, y2) = landmarks[:2]
    else:
        print('wrong region')
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    if opt != 'b':
        dilation *= 4
    line = binary_dilation(line, iterations=dilation)
    return line

def remove_nose(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    return line

def remove_mouth(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[3:5]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    return line


# def parse(img, reg, real_lmk, fakemask):
def parse(img, reg, real_lmk):
    five_key = get_five_key(real_lmk)
    if reg == 0:
        mask = remove_eyes(img, five_key, 'l')
    elif reg == 1:
        mask = remove_eyes(img, five_key, 'r')
    elif reg == 2:
        mask = remove_eyes(img, five_key, 'b')
    elif reg == 3:
        mask = remove_nose(img, five_key)
    elif reg == 4:
        mask = remove_mouth(img, five_key)
    elif reg == 5:
        mask = remove_nose(img, five_key) + remove_eyes(img, five_key, 'l')
    elif reg == 6:
        mask = remove_nose(img, five_key) + remove_eyes(img, five_key, 'r')
    elif reg == 7:
        mask = remove_nose(img, five_key) + remove_eyes(img, five_key, 'b')
    elif reg == 8:
        mask = remove_nose(img, five_key) + remove_mouth(img, five_key)
    elif reg == 9:
        mask = remove_eyes(img, five_key, 'b') + remove_nose(img, five_key) + remove_mouth(img, five_key)
    # else:
    #     mask = generate_random_mask(fakemask)
    mask = random_deform(mask, 5, 5)
    return mask*1.0

# def get_convex_hull(image, region_idx):
#     dets = detector(image, 0)
#     masks = []
#     if len(dets) == 0:
#         return 'NoFace'
#     for k, d in enumerate(dets):
#         shape = predictor(image, d)
#         landmarks = np.array([[p.x, p.y] for p in shape.parts()])
#         # pts = landmarks[landmark_list]
#         pts = landmarks[:68]
#         mask = parse(image, region_idx, pts)
#         src_mask = mask > 0  # (H, W)

#         tgt_mask = np.asarray(src_mask, dtype=np.uint8)
#         tgt_mask *= 255
            
#     return tgt_mask

def sladd_get_convexhull(image, landmarks, region_idx):
    pts = landmarks[:68]
    mask = parse(image, region_idx, pts)
    src_mask = mask > 0  # (H, W)

    tgt_mask = np.asarray(src_mask, dtype=np.uint8)
    tgt_mask *= 255
            
    return tgt_mask
# def vizMask(mask):
#     src_mask = mask > 0  # (H, W)

#     tgt_mask = np.asarray(src_mask, dtype=np.uint8)
#     tgt_mask *= 255
#     return tgt_mask

# if __name__ == '__main__':
#     data_dir = 'Temp'
#     output_path = 'Convex/'
#     filenames = os.listdir(data_dir)
#     for filename in filenames:
#         frame = cv2.imread(os.path.join(data_dir,filename))
#         mask = get_convex_hull(frame, 0)
#         cv2.imwrite(os.path.join(output_path,f'{idx}_{filename}'), mask)
