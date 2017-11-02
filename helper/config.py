import os.path

# gets home dir cross platform
ddir = os.path.join("/home/zengyu/Lab/pytorch/standard-panel-classification/data/")

# note: if you used our download scripts, this should be right
# path to VOCdevkit root dir
VOCroot = ddir

VOC_CLASSES = (  # always index 0
    # 'aeroplane', 'bicycle', 'bird', 'boat',
    # 'bottle', 'bus', 'car', 'cat', 'chair',
    # 'cow', 'diningtable', 'dog', 'horse',
    # 'motorbike', 'person', 'pottedplant',
    # 'sheep', 'sofa', 'train', 'tvmonitor')
    'sp_sfb', 'sp_dn', 'sp_cns', 'sp_qn', 'sp_xn', 'sp_gg',
    'nsp_sfb', 'nsp_dn', 'nsp_cns', 'nsp_qn', 'nsp_xn', 'nsp_gg'
)

# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

# SSD300 CONFIGS
# ---------------------------------------------------------------------------------
# newer version: use additional conv11_2 layer as last layer before multibox layers
v2 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim'      : 300,

    'steps'        : [8, 16, 32, 64, 100, 300],

    'min_sizes'    : [30, 60, 111, 162, 213, 264],

    'max_sizes'    : [60, 111, 162, 213, 264, 315],

    # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance'     : [0.1, 0.2],

    'clip'         : True,

    'name'         : 'v2',
}

# use average pooling layer as last layer before multibox layers
v1 = {
    'feature_maps' : [38, 19, 10, 5, 3, 1],

    'min_dim'      : 300,

    'steps'        : [8, 16, 32, 64, 100, 300],

    'min_sizes'    : [30, 60, 114, 168, 222, 276],

    'max_sizes'    : [-1, 114, 168, 222, 276, 330],

    # 'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'aspect_ratios': [[1, 1, 2, 1 / 2], [1, 1, 2, 1 / 2, 3, 1 / 3], [1, 1, 2, 1 / 2, 3, 1 / 3],
                      [1, 1, 2, 1 / 2, 3, 1 / 3], [1, 1, 2, 1 / 2, 3, 1 / 3], [1, 1, 2, 1 / 2, 3, 1 / 3]],

    'variance'     : [0.1, 0.2],

    'clip'         : True,

    'name'         : 'v1',
}
