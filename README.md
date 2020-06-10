# SphereFace


## 训练数据
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training. This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance improvement has been seen if the dataset has been filtered before training. Some more information about how this was done will come later.
The best performing model has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.

## 面部对齐
> cd facevec/align  
> python align_dataset_mtcnn.py ../../../CASIA-WebFace ../../data  
> python gen_filelist.py ../../data  
> python align_dataset_mtcnn.py ../../../lfw ../../test_data 

## 训练
> cd facevec
> python train.py sphere_cos_softmax ../../data/filelist.txt

## 测试
> cd facevec
> python evaluate.py sphere_cos_softmax ../../test_data/filelist.txt


