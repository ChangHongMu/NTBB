The source code of our manuscript submitted to : Journal of the Science of Food and Agriculture  
---------------------------------------------------------------------------------------------------  
Non-destructive Detection of Blueberry Skin Pigments and Fruit Intrinsic Qualities Based on Deep Learning  
==========================================================================================================  
****
Changhong Mu, Zebin Yuan, Xiuqin Ouyang, Pu Sun, Bo Wang *    
---------------------------------------------------------  
****
**Prerequisites:**  
1.Visual Studio 2015 x64 enviroument  
2.CUDA v8.0, cudnn v7.0  
3.Matlab 2016b  
4.GPU: NVIDIA GeForce GTX 1080 Ti  
****
**Default path:**  
1.The default installation path of CUDA is: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64",  
2.The default installation path of VisualStudio is: "C:\Program Files\Microsoft Visual Studio 14.0\VC\bin".  
3.The readers should change the CUDA path and VisualStudio path in '.\cuda\compile.m' file accordingly.  
4.The matlab path of '.\cuda*.cu' files should also be changed accordingly  
(e.g., the default matlab path is "C:\Program Files\MATLAB\R2016b\extern\include\mex.h").  
****
**Usage:**  
a. Please first compline Caffe of Faster Rcnn and Faste Rcnn for Matlab.  
b. Please download the Model of NTBB.  
d. Please put the model NTBB under folder \caffe\model.  
e. Run NTBBTest.m  
f. Results can be found in \Resutls\  
****
**Downloads:**  
All datasets of our method are availabled from  
baidu cloud: (comming soon).  
****  
**Notice:** 
a. Because we have changed the original code of Faste RCNN and Faster RCNN, please replace the relevant files for running.  
b. If you make changes to the accelerated code (CUDA), please recompile the file of CUDA/*.cu .  
c.Due to the current project in the patent application and the caffe installation is too complicated, now we provide a simple preview version, the simple version is written in pytorch

