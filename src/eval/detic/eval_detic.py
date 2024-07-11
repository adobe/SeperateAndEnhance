# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
import sys
import mss

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/')
from centernet.config import add_centernet_config
from detic.config import add_detic_config


# from detic.predictor import VisualizationDemo

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
import torch
from detic.modeling.utils import reset_cls_test

from large_scale import concepts

def get_clip_embeddings(vocabulary, prompt='a '):
    from detic.modeling.text.text_encoder import build_text_encoder
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

class VisualizationDemo(object):
    def __init__(self, cfg, custom_vocabulary, 
        instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get("__unused")
        self.metadata.thing_classes = custom_vocabulary.split(',')
        classifier = get_clip_embeddings(self.metadata.thing_classes)
        
        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.predictor = DefaultPredictor(cfg)

        reset_cls_test(self.predictor.model, classifier, num_classes)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

# ------------------------ 
# hyper parameters
# ------------------------
# test_path: path to the test images
# confidence_threshold: confidence threshold for detection
confidence_threshold = 0.3
test_path = './'


def setup_cfg():
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file('configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml')
    cfg.merge_from_list(['MODEL.WEIGHTS', 'models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'])
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
    cfg.freeze()
    return cfg


cfg = setup_cfg()


all_concepts = concepts['animal_train'] + concepts['animal_test'] \
            + concepts['object_train'] + concepts['object_test']
word_maps = {}
custom_vocabulary = ''
for i in range(len(all_concepts)):
    word_maps[all_concepts[i]] = i
    custom_vocabulary += all_concepts[i] + ','
custom_vocabulary = custom_vocabulary[:-1]

demo = VisualizationDemo(cfg, custom_vocabulary)   


prompts = os.listdir(test_path)
prompts.sort()

overall_success = 0.0
overall_count = 0.0
for prompt in prompts:
    words = prompt.split()
    idx1 = word_maps[words[4]]
    idx2 = word_maps[words[7]]
    
    success_case = 0.0
    for i in range(64):
        path = test_path + '{}/{}.png'.format(prompt,i)
        img = read_image(path, format="BGR")
        predictions, _ = demo.run_on_image(img)
        pred_classes = predictions['instances'].pred_classes
        if idx1 in pred_classes and idx2 in pred_classes:
            success_case += 1
    print(prompt,success_case / 64.0)
    overall_success += success_case
    overall_count += 64
print('overall success rate:',overall_success / overall_count)

