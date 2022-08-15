

#   MICCAI2022-GOALS@Baidu
-   MICCAI2022 Challenge: Glaucoma Oct Analysis and Layer Segmentation (GOALS)
-   GOALS Challenge:https://aistudio.baidu.com/aistudio/competition/detail/230/0/introduction
-   Implementation by PaddlePaddle/PyTorch


##  Task1:Layer Segmentation
    Model:TCCT/PyTorch
-   Pre-Process:       
    "python t1_pre.py"
-   Training:       
    "python t1_train.py --gpu=0"
-   TTA:            
    "python t1_tta.py --root=xxx --gpu=0"
-   Post-Process:    
    "python t1_post.py"


##  Task2:Glaucoma Classification
    Model:ResNet/Paddle
-   Training:       
    "python t2_train.py --gpu=0"
-   Ensemble:            
    "python t2_ensemble.py --root=xxx --gpu=0"


##  Weights
-   百度网盘链接：https://pan.baidu.com/s/1p2ceNgFR6xKgDnwRpasnkw?pwd=life 
提取码：life

-   ![img](weights.png)


##  GitHub 仓库
    https://github.com/tyb311/TCCT

##  Contact
-   tyb311@qq.com