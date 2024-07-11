# Separate-and-Enhance: Compositional Finetuning for Text2Image Diffusion Models

This is the repository for [*Separate-and-Enhance: Compositional Finetuning for Text2Image Diffusion Models*](https://arxiv.org/abs/2312.06712), published at SIGGRAPH 2024.  



<img src='./teaser/teaser.gif'/>

[[Project Page](https://zpbao.github.io/projects/SepEn/)]
[[Paper](https://arxiv.org/abs/2312.06712)]

## Set up
Build conda environment by running: 

```
conda create -n sepen python=3.10  
conda activate sepen
pip install -r requirement.txt
```

## Training
#### Individual concepts 
see ```src/run_individual.sh``` for a sample training script.  
#### Individual concepts
see ```src/run_large.sh``` for a sample training script.  
#### Sample
see ```src/sample.py``` for refernce.

## Evaluation
#### FID
install [clean-fid](https://github.com/GaParmar/clean-fid) via ```pip install clean-fid```  then refer to ```src/eval/fid/eval_fid.py``` for FID evaluation.

#### BLIP score
We adopt the implementation from [A&E](https://github.com/yuval-alaluf/Attend-and-Excite). See ```src/eval/blip/eval_blip.py``` for BLIP similarity score evaluation. 

#### Detection score
Clone and build [Detic](https://github.com/facebookresearch/Detic) from their official repo. Then move the Python files under ```src/eval/detic``` to the cloned folder. See ```src/eval/detic/eval_detic.py``` for details.

## Large-scale concepts and prompts
The 220 concepts we used for the large-scale experiment is at ```src/concepts/large_scale.py```.  
The 200 evaluation prompts are at ```src/concepts/large_test.txt```. 


## Acknowledgment
Part of our codes is inspired by [Custom Diffusion](https://github.com/adobe-research/custom-diffusion) and [Attend and Excite](https://github.com/yuval-alaluf/Attend-and-Excite).

We leverage [Detic](https://github.com/facebookresearch/Detic) and [clean-fid](https://github.com/GaParmar/clean-fid) for our evaluation.

## Citation

```
@inproceedings{bao2024sepen,
    Author = {Bao, Zhipeng and Li, Yijun and Singh, Krishna Kumar and Wang, Yu-Xiong and Hebert, Martial},
    Title = {Separate-and-Enhance: Compositional Finetuning for Text2Image Diffusion Models},
    Booktitle = {SIGGRAPH},
    Year = {2024},
}
```


