**real-nerveseg2021**

_Authors:_ Lai Wing Sum, Lee Fung Lui, Man Yin Ting Eunice, Wong Chin Hung, Yeung Chi Kin, WONG, Ngai Nick Alex

**Introduction**

Complications that resulted in life-long irreversible nerve damages are seldom accompanied with certain types of surgery such as esophageal surgery, these complications often result in functional loss and lifelong dependency in medications. The chance of causing these complications is highly dependent on experiences of surgeons, this implied that there is room of improvement by the use of real-time intraoperative imaging techniques.

Myelin layers of nerve has a special optical property where reflectance is directly dependent on wavelength and amount of reflected light is highly correlated to the number of myelin layer of nerve. By carefully choosing suitable wavelength, the nerve will reflect significantly more light than the surrounding non-nerve tissue. This can be used to produce high contrast images that highlight nerve.

In this project, we developed a deep learning model that aims at real time identification of nerve during surgical procedures. The models consist of two convolutional neural network that classify images that contains nerve and segment nerve.

A powerful neural network that classifies objects with very high accuracy requires very large training dataset and training time. Existing powerful pre-trained classification neural network can be augmented for classifying medical images such as nerve by transfer learning.

**File structure of this deposit**

This deposit is divided into 5 subfolders that their contents are described in their names.
The classification training and evaluation scripts are contained in the subfolder &quot;classification&quot;. The segmentations are divided by their models into their own folders, &quot;UNET&quot;, &quot;DoubleUNET&quot;, &quot;deeplabV3+&quot;.
The root directory contained this readme file.

**Classification**

This folder contains the script for training transfer learnt classification neural network using the pre-trained ImageNet image classification.

Getting started
Dependencies
These codes are written in Python 3 and has been tested in environment with the following packages:
tensorflow 2.3.0
numpy 1.18
sklearn 0.23
These dependencies can be installed using pip install

**Data and models**

The classification neural networks were built on weights pre-trained with ImageNET available in Keras Application by transfer learning. ImageNET is an image database with number of images for each kind of objects. Further information can be found here:
[http://image-net.org/
](http://image-net.org/)The pre-trained classification models are:
Densenet169
Densenet201
Resnet50v2
Resnet101v2
mobilenetv2

**Code**

Sources
The majority of the training code comes from tensorflow&#39;s guide to transfer learning and fine-tuning, the guide can be found here:
[https://www.tensorflow.org/tutorials/images/transfer\_learning
](https://www.tensorflow.org/tutorials/images/transfer_learning)The majority of the original tensorflow guide code is comprehended and modified by Marco Lee.

**Preparation of dataset**
The images in the datasets are manually divided into &quot;training and validation&quot; and &quot;testing&quot; folder, each folder contained images from experiment on some particular dates. Each folder contains three sub-folder which corresponds to the three classes that the images are being classified into (nerve, opening wound, tendon). These images are in jpg format.
During the training process, the images in the &quot;training and validation&quot; dataset is read by tensorflow function and automatically shuffled and divided into training and validation datasets in 8:2 ratio.

The default location for the datasets are the subfolder &quot;/Classification/datasets/&quot;. The images has been compressed into a zip file for download, they should be decompressed before use.
091810021113.zip can be downloaded here:
[https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/EQYuqS-gtflNp3FM5xzjCJEBwiGCvN05ukLK72tGVwImUg?e=Wt5WuK]

101611271211.zip can be downloaded here:
[https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/ETmoTe6sgsRIjgCG3bw_OUYBJuKP819Pd_8mpSxq7esUoA?e=UH0WRF]

**Training**

The scripts for training neural network using pre-trained weighting of the following models are included:
 densenet169, densenet201, mobilenet v2, resnet50v2, renset101v2.
The file names of these Python scripts are named in the following patterns:
**/classification/train\_#ModelName#.py** , where #ModelName# are the names of the models.

**Defining variable parameters for the training scripts**
At the beginning of the code, the training parameters, dataset path and path for saving results and trained weighting are defined. The default parameters for resnet152v2 are listed below as example:

#define parameters
BATCH\_SIZE = 12
IMG\_SIZE = (224, 224)
dp\_rate = 0.3
base\_learning\_rate=0.001
initial\_epochs = 10
fine\_tune\_epochs = 15
fine\_tune\_at = 555
folder = &quot;resnet152v2&quot;
all\_dir

BATCH\_SIZE defines the batch size.
IMG\_SIZE defines the size of the image being resized into before trained.
Dp\_rate defines the dropout rate
folder specifies the name of the working directory and is defined to be the name of the models.

Variable parameters like the fine- tuning epoch and the layer from which the neural networks are being fine-tuned are defined at the beginning. The values of these parameters for the different models used in this study was defined there.

all\_dir specify the directory where the training and validation datasets are read.

**Image pre-processing and model compilation**
The datasets are loaded and resized into size equal to IMG\_SIZE (224x224) and preprocessed before being used for transfer learning. The pre-processing includes the built-in preprocessing function in tensorflow for the respective pre-trained models.

The data preprocessing is implemented as Keras layer. This layer is followed by:
 loading the pre-trained weightings with all base models being frozen,
global average layer,
Keras dropout layer with the defined dropout rate,
prediction dense layer with number of neurons the same as the number of classification class.
The layers above are chained together and compiled with Adam optimizer and Sparse Categorical Cross Entropy loss.
 The pre-trained imagenet classification weightings are loaded (download if not yet downloaded before) without including the top layer for image classification when the python scripts are executed.

**Transfer learning and fine tuning**
First, the loaded pre-trained weightings have the base models being frozen and not trained. The top classification output layer is trained with the pre-defined number of epochs (initial\_epochs) and pre-defined learning rate (base\_learning\_rate). The accuracy and loss are reported during the training process.

After the initial training, the layers from the base model from the layer number defined in &quot;fine\_tune\_at&quot; parameter onwards are unfrozen. The model is recompiled with the learning rate equal to 1/10 of base\_learning\_rate and pre-defined number of fine-tuning epochs (fine\_tune\_epochs)

**Results reporting**
After the initial training and fine tuning, accuracy and loss reported throughout the training process were plotted against epochs and saved in the working directory.

The newly trained weightings are saved in the subfolder (/classification/saved\_model/current) of the working directory.

As the training and validation datasets are obtained by randomly shuffling and dividing the &quot;training and validation&quot; dataset, the training and validation datasets are different every time the training script is executed, so the performance of the trained weighting on the training and validation datasets is evaluated immediately after the training and fine-tuning process. The evaluation process and results saving are the same as the script for evaluating test dataset. So, please refer to the paragraphs below for details.

Evaluation of the transfer learnt weightings

The script &quot; **/classification/evaluateVideo.ipynb**&quot; can be run to evaluate a testing dataset with the trained classification weighting.
The whole evaluation process is defined in function &quot;eva()&quot;.

Firstly, the testing datasets and saved models are loaded by tensorflow built-in function. The datasets are predicted into the three classes (nerve, opening wound, tendon) using the saved weighting in batches with batch size equal to &quot;BATCH\_SIZE&quot;. During the prediction process, for each batch of images, the last image of each batch is saved as sample images indicating the correctness of prediction and the true class label and predicted class label. The overall confusion matrix of all classes is generated. This confusion matrix is then broken down into confusion matrix for the respective classes. Subsequently, the True Positive, True Negative, False Positive, False Negative numbers are obtained for each class. These numbers are used to compute sensitivity, specificity, positive predictive value, negative predictive value, accuracy, F1 score (positive and negative) and Matthew&#39;s coefficient. ROC curves are also plotted and AUC are calculated.

The datasets should be read without shuffling to obtain the same set of output sample images when evaluating the same set of test dataset with different trained weightings for comparison purpose.

No sample images are saved in evaluating training and validation datasets.

The best weightings of each of the 5 models we studied that we obtained can be downloaded with the following links and decompressed into the folder &quot;/Classification/saved\_models/#modelname#/&quot; for loading for evaluation, where #modelname# is the name of the model.
denseNET169018.zip
[https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/EX37THnpcJ1FjCixPliY3IoB63B9ORR5j5kb9-IncaiqsA?e=CIapS8]
denseNET201020.zip
[https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/EdQEZ_DWgLNFtW3hC24_oAwBEYLAP2FehEYJmJPSAsvE5w?e=ef7LNd]
mobileNETv2020.zip
[https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/EWvFGn6DOjFKhOO00nJw33cBlZLXnzrsVcqJHtf4ES5aIw?e=MbQVzg]
resnet101v2017.zip
[https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/EcWLRuxQPLZPro_WXNmrZ6kBbwrWd3rWuckWvgftxlgRnA?e=7uXg4Q]
resnet50v2024.zip
[https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/EcnHlOWH2r1Fru52lfeDNkEBRXx9cPQNsW4NBsSnD9tVtg?e=1dfhq9]

**Saved result files of Classification**
 The accuracy and loss metrics during the training process are plotted in the two file &quot;acc\_loss.png&quot; and &quot;acc\_loss\_after\_fine\_tuning.png&quot;. Files with the following patterns are also saved in the directory &quot;/classfication/#modelname#/&quot; : #modelname#\_train/val\_AUC/CM/F1.txt.
 These files are results of evaluating the training and validation dataset immediately after training the weighting. AUC stands for area under curve figure, CM stands for confusion matrix and F1 is the files that contained the TPR, NPV, MCC, accuracy...etc figure.

Files with similar patterns are also saved in the process of evaluating the independent test dataset. These files are saved in the directory: &quot;/classification/VideoResults/&quot; with subfolder names that consist of two parts: model name and nos (a code number that was used to identify the weightings saved after different instance of training) . The files are in the following format: &quot;#model name+nos#\_AUC/CM/F1.txt&quot;.
Sample images are also generated with the following file naming pattern:
model name + nos+ Correct/Wrong+TruthLabel+PredictedClass+batch number+#n image of that batch.png. For example, the name &quot;denseNET201020Wrongnerveopening wound12931.png&quot; mean #020 weighting of denseNET201 wrong prediction with truth label of nerve predicted as opening wound and the image come from 129th batch of images and the #31 image in that batch counting from zero.

**Manual work after training**

After each instance of training of classification models, a folder with custom name should be manually created at the directory of &quot;/classification/#modelname#/&quot; and all metrics files generated in the training and evaluation of training and validation process should be manually moved to that folder. Afterwards, the folder &quot;/classification/#modelname#/saved\_model/ current &quot; where the trained weighting is saved should be renamed with custom name and moved to the directory &quot;/classification/saved\_models/#modelname#/&quot; manually.

**Segmentation**

Segmentation model data preparation

Since segmentation model is only trained by nerve images, all nerve images are chosen from the classification dataset manually. Then, import all the nerve images to an online annotation software called &quot;Supervisely&quot; as the location of nerve need to be annotated manually. The software can be found in the following link: [https://supervise.ly/](https://supervise.ly/). Follow the instructions of Supervisely and annotate all the nerve images.

The flow of annotation:

[https://drive.google.com/file/d/1AIUffwzkgLgl4T8wHCxGJRc9Y2S\_ClrP/view?usp=sharing](https://drive.google.com/file/d/1AIUffwzkgLgl4T8wHCxGJRc9Y2S_ClrP/view?usp=sharing)

Example of annotation:
![alt text](https://i.imgur.com/vf8bS8Q.png)
After annotation, download the datasets as follows:

![alt text](https://i.imgur.com/Pgurz5M.png)
Unzip the downloaded tar file, 4 folders can be found. They are &quot;ann&quot;, &quot;img&quot;, &quot;masks\_human&quot; and &quot;masks\_machine&quot; respectively. Only &quot;img&quot; and &quot;masks\_machine&quot; are used for the model training as &quot;img&quot; folder store all the raw images and &quot;masks\_machine&quot; folder store all the annotation masks.

After annotation, image is cropped to get rid of the overexposure part and then resized to 512x512 using python script &quot;cropLeftOnly-1080To512.py&quot; under your data folder. The figure below demonstrates before and after cropping of input image.

![](RackMultipart20210501-4-1h2cyox_html_954dc3cccfabd7bc.jpg) ![](RackMultipart20210501-4-1h2cyox_html_d473f29f2dd46f13.png)

_Before and after cropping of input image._

UNET

Getting started
Dependencies
These codes are written in Python 3 and has been tested in environment with the following packages:
tensorflow 2.3.0
numpy 1.18
skimage 0.18

These dependencies can be installed using pip install

**Code**
**Sources**
The majority of the training code comes from tensorflow&#39;s official guide for image segmentation, the sources can be found here:
[https://www.tensorflow.org/tutorials/images/segmentation
](https://www.tensorflow.org/tutorials/images/segmentation)the majority of the original code is comprehended and modified by Marco Lee.

**Structure of the code**
The code is divided into 2 main scripts and assistant scripts.

&quot;/UNET/ segmentation-modified-dataset-and-usePretrained-Share.ipynb&quot;
The main script that load dataset to train UNET neural network and to evaluate the training and validation dataset which is randomly allocated each time just before the training process with the trained UNET weighting.
&quot;/UNET/Evaluation.ipynb&quot;
The script that load the trained UNET weighting and to evaluate it with test dataset.
&quot;/UNET/512-resize.py&quot;
The script that resize convert all .jpg and .png files in the same directory.
&quot;/UNET/image-to-tfrecord-Marco\_dataset.ipynb&quot;
The script that convert all images into tfrecord format for training UNET.

The subfolder &quot;/UNET/TFrecord&quot; is the default directory where the tfrecord datasets are placed for training and testing. For details, please refer to the section &quot;Preparation of dataset&quot;.
Trainval.tfrecords can be downloaded here:
[https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/Ea05prqS4NdBjY4Z7o-z2tYBano3o99h9rwuoIRA8OG93A?e=CFaTZd]
test.tfrecords can be downloaded here:
[https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/Ee2QCANDCxNBqmjpI3e2F6oBlc4XfcA8cmpa_S9Ci9_EQA?e=MVJjNt]

**Preparation of dataset**
The annotated images should be in either &quot;.jpg&quot; or &quot;.png&quot; format. All images are converted into .tfrecord format before being loaded for training and inferencing.

The whole datasets used in this project are divided into training, validation and test datasets. The test dataset remains unchanged over the whole project. For each segmentation training instance, the training and validation datasets are different and is obtained each time when executing the training jupyter notebook by random shuffling and dividing a dataset &quot;trainval&quot; that contained the images other than the test dataset according to the ratio 7:1.5:1.5(train:val:test).

Important: To obtain the same &quot;test&quot; and &quot;trainval&quot; datasets across different segmentation models (UNET, Double UNET and DeeplabV3) that is used in this project, filelists that contained file names of the images for &quot;test&quot; and &quot;trainval&quot; datasets are generated by another code, please refer to the section below (DeepLabv3+ Segmentation Model – Data - ImageSets**)**.

After all the images containing nerve are annotated with &quot;supervisely&quot;, all images from different experiments are combined into a single folder in their respective &quot;img&quot; and &quot;masks\_machine&quot; folders. The python script &quot; **/UNET/512-resize.py**&quot; is then put into each of these folders and executed each of these python script in respective folders to resize all images into 512x512 size. Please be reminded that you should save another copy of images before carrying out this step.

The tfrecord format was used in the original code for storing the dataset. Tfrecord is a simple cross-platform format that allows efficient serialization of structured data. For further information about tfrecord, please refer to the following link:
[https://www.tensorflow.org/tutorials/load\_data/tfrecord
](https://www.tensorflow.org/tutorials/load_data/tfrecord)&quot; **/UNET/image-to-tfrecord-Marco\_dataset.ipynb**&quot; is executed to convert these resized images into .tfrecord format, which is uncompressed. The program will first read the txt filelists from the specified location and create a python dictionary for the filelist. According to which the python dictionary is then used to read images from the directories containing the dataset images, all files read according to the file list dictionary will be written into a .tfrecord file, each images being read is immediately renamed with a suffix of &quot;.done&quot;. This program should be run twice such that the &quot;trainval&quot; and &quot;test&quot; datasets images are read and seperately written into two separate .tfreocrd files. If the two txt filelists contained the file names of all images in the folder that contained all images to be used, all images files should have been renamed into .done file after the whole conversion procedure. The .done files are of no use afterward and can be deleted. Please be noted that the .tfrecord files are uncompressed and thus the file size are much larger than the original .jpg or .png images.
Please be reminded to change the variables that defines the filelist to be read and file path to which the tfrecord to be saved (&quot;WorkFile&quot; and &quot;Fname&quot;) .
Whether all images have been read and saved in tfrecord can be checked by whether file extensions of all images have been converted into &quot;.done&quot;.

DeeplabV3+ also share similar data structure and preparation of data in deeplabV3+ is also similar to UNET described here.

**Defining parameters**

The script &quot; **/UNET/ segmentation-modified-dataset-and-usePretrained-Share.ipynb**&quot; is the main script that is used to train the UNET segmentation weightings and evaluate the training and validation dataset.

At the beginning of the code, the various parameters are defined.

OrgSize
processedSize
raw\_image\_dataset
dp
EPOCHS
learning\_rate
Mname

Since UNET accept only image size of 224x224, the original size of the image and target size for resizing (orgSize, processedSize) are input for defining how the images are resized. raw\_image\_dataset defines the absolute datapath to which the tfrecord file are read.
Dp defines the dropout rate in the training process.
EPOCHS defines the number of epochs for training.
learning\_rate defines the learning rate when the model is compiled for training.
Mname defines the file name of the folder to save the results.
Later in the script, more parameters are defined:
BATCH\_SIZE
train\_size
BATCH\_SIZE defines the batch size for training and should be set to 1 if the dataset is very large in order to reduce memory usage during training.
train\_size defines the size of the training dataset. The whole trainval datasets are read and randomly shuffled, the number of image equal to train\_size are taken from the shuffled dataset as training dataset and the remaining are taken as validation dataset. Please be noted that the exact train size should be calculated and input as this variable manually according to the size of your trainval dataset and the train:validation ratio you defined.

**Image Augmentation**
There is 50% chance for an image from the training dataset to be flipped horizontally.

**Feature extraction layer.**
In UNET, the feature extraction layer can be extracted from some layers of classification neural networks such as mobilenetv2 and densenet201, however, the shape of these layers must be matched with UNET in order to be incorporated into UNET. In this project, mobilenetv2 has been used for feature extraction layer.
The pre trained ImageNET classification neural network weightings available in Keras application can be loaded, alternatively, other trained neural network weightings can also be loaded. Those feature extraction layer from various classification neural network that have been tried in this study with matched shape are listed. The suitable layers corresponding to the selected classification neural network should be selected by changing the annotation.

**Compiling the model**
Finally, the model is compiled with Adam optimizer, Sparse categorical cross entropy loss and defined learning rate. The compiled model is trained. The results (accuracy, loss and IOU) throughout the training process are plotted and saved into the folder: results/#Mname#/ . The model weighting is saved in the folder, /UNET/saved\_model/#Mname#/ . The saved model weighting is loaded.

**Evaluation of the training and validation dataset**
Since the training and validation dataset are generated every time just before training by dividing the randomly shuffled dataset and the information that define what the training and validation dataset generated contains are not saved, so training and validation dataset should be evaluated immediately after training the weighting.
In the remaining code of this script file, each images in the training and validation dataset are evaluated by the trained weighting. The IOU and dice coefficient for each images are calculated and saved temporarily in python list. The python list with the IOU and dice coefficient of the whole training and validation datasets are used to calculate the statistically result (mean, standard error of mean, 95% confidence interval) for the respective dataset in this instance. The statistical result is saved in the file &quot;saved\_model/#Mname#/TrainValResult.txt&quot;
The result illustration for each of the images in the two dataset are plotted but not saved automatically. Users can save some of the representative images manually.

The UNET mobileNETv2 feature extraction layers weightings that we trained can be downloaded and decompressed into &quot;/UNET/saved\_model/&quot; for loading for evaluation.
[https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/Efdx4LB-X45Dp3BJ9piLZmIBtb8Ih9kQk1d3W-TNEgJmMw?e=51qtRy]

**Evaluation of the test dataset**
Please run the &quot; **/UNET/Evaluation.ipynb**&quot; for evaluating an independent test dataset.
The evaluation process is similar to that in evaluating the training and validation dataset in the &quot;/UNET/segmentation-modified-dataset-and-usePretrained-Share.ipynb&quot;
Please refer to section &quot; **Evaluation of the training and validation dataset&quot; for details.**

Mname define the folder name containing the trained UNET weighting to be evaluated and HPath define parent directory of the folder defined in Mname. The weighting trained previously is read.

The variable dname define the file path of the test dataset tfrecord file without the &quot;.tfrecord&quot; file extension.
The results are saved in the file &quot;saved\_model/#Mname#/evaluation.txt&quot;

**DeepLabv3+ Segmentation Model**

**Introduction**

This is DeepLabv3+ with parameters tuned to suit our training of segmentation of nerve images.

![](RackMultipart20210501-4-1h2cyox_html_3250f920c94fe518.gif)

**Requirement**

Refer to [https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md)

![](RackMultipart20210501-4-1h2cyox_html_3250f920c94fe518.gif)

**Source**

This script was based on the code from [https://github.com/tensorflow/models/tree/master/research/deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab).

![](RackMultipart20210501-4-1h2cyox_html_3250f920c94fe518.gif)

**Data**

Stereoscopic images of the murine nerve transection process were used as demonstration. Sample data can be downloaded from [https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/EUMYzOcEnUlHh8JSxkzkDcIB0bcDzd2UNTuUaLf67RBKSQ?e=RE31Zm]

Please refer to the &#39;datasets\example\_data folder to see the data format.

**Under &#39;\deeplabV3+\datasets\example\_data\dataset&#39; is where you should place your data.**

Structure of dataset folder:

| Folder | Comment |
| --- | --- |
| ImageSets | Contains train.txt, trainval.txt, val.txt. Inside each text file, there are names of the images file, indicating which group of data does it belong to. To generate these txt files, place ALL images under a folder and run &#39;generateTxt.py&#39; under it. |
| SegmentationClass | The annotated masks of input images. |
| JPEGImages | Input images fed into DeepLabv3+. |

To use your customized data, follow the data format in example\_data.

After placing your data, change &#39;WORK\_DIR&#39; variable in &#39;/deeplabV3+/datasets/convert\_example.sh&#39; to your folder name. Then run &#39;convert\_example.sh&#39; to convert your data to tfrecord format.

![](RackMultipart20210501-4-1h2cyox_html_3250f920c94fe518.gif)

**Use**

Add your data&#39;s information in &#39;/deeplabV3+/ datasets/segmentation\_dataset.py&#39; to let the script recognize your data.

Run &#39;/deeplabV3+/vis-fyp.sh&#39; to see the visualized segmentation result of the model under &#39;your\_data\_folder/vis&#39;

Run &#39;/deeplabV3+/ train-fyp.sh&#39; to train with your own custom data. The ouput model is in &#39;your\_data\_folder/ exp/train\_on\_trainval\_set&#39;.

Run &#39;/deeplabV3+/ eval-fyp.sh&#39; to see the IOU result of your data.

Remember to change the &#39;PQR\_FOLDER&#39; variable in all these above mentioned with .sh scripts to your data folder name.

Please refer to [https://github.com/tensorflow/models/tree/master/research/deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab) for more instruction.

![](RackMultipart20210501-4-1h2cyox_html_3250f920c94fe518.gif)

**DoubleUNet segmentation model**

**Introduction**

This is DoubleUNet with parameters tuned to suit our training of segmentation of nerve images.

![](RackMultipart20210501-4-1h2cyox_html_3250f920c94fe518.gif)

**Requirement**

Refer to [https://github.com/DebeshJha/2020-CBMS-DoubleU-Net](https://github.com/DebeshJha/2020-CBMS-DoubleU-Net)

![](RackMultipart20210501-4-1h2cyox_html_3250f920c94fe518.gif)

**Source**

This script was based on the code from [https://github.com/DebeshJha/2020-CBMS-DoubleU-Net](https://github.com/DebeshJha/2020-CBMS-DoubleU-Net)

![](RackMultipart20210501-4-1h2cyox_html_3250f920c94fe518.gif)

**Data**

Stereoscopic images of the murine nerve transection process were used as demonstration. Please refer to the &#39;data&#39; folder to see the data format.

**Under &#39;\doubleunet\data\[dataset-group]\&#39; is where you should place your data. Data can be downloaded from** [https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/EUNP3qv0-DBLmBFVunvfoesB_Wj-eU84gB-ZbDgpM_Kf4g?e=9bOteK]

Structure of dataset folder:

| Folder | Comment |
| --- | --- |
| masks\_machine | The annotated masks of input images. |
| img | Input images fed into DoubleUNet. |

To use your customized data, follow the data format in the folder.

Remember to process your images with &#39;\doubleunet\from1to255.ipynb&#39; before actually running the model.

![](RackMultipart20210501-4-1h2cyox_html_3250f920c94fe518.gif)

**Use**

Change &#39;test\_path&#39; variable in \doubleunet\predict.ipynb and \doubleunet\train.ipynb to your data folder path.

Change &#39;model\_path&#39; variable in \doubleunet\train.ipynb to your desired pre-trained weight path. Our trained weight is [https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/EfdiIKHCYa9Bj6CnmmP8ltEBH62-0cM7gmeob31XwPLObA?e=IiGlMf] Change &#39;model&#39; variable in \doubleunet\predict.ipynb to your model file path.

Run \doubleunet\train.ipynb to train DoubleUNet with your dataset. The model file .h5 will be generated in root folder.

Run \doubleunet\predict.ipynb to predict masks of your dataset.

![](RackMultipart20210501-4-1h2cyox_html_3250f920c94fe518.gif)

**Real time nerve segmentation tool (Reduced framerate)**

**Introduction**

This script integrates classification model and segmentation model.

First images are passed into a classification model. If the predicted label by the classification model of that image has the index of 0, which represents &#39;Nerve&#39; class in our example data, the image will then be fed into the segmentation model to predict the mask of nerve.

**Requirement**

Environment can be set up by running pip install -r requirements.txt.

**Data**

Stereoscopic images of the murine nerve transection process were used as demonstration. &#39;/linked/example\_data&#39; folder has been created to demonstrate the architecture of data. It can be downloaded here [https://connectpolyu-my.sharepoint.com/:u:/g/personal/18041854r_connect_polyu_hk/EUoh39Kb77hHokzKq5CbY-IBFQ2hnrgSX2RnJOVR4baXBg?e=jqUb1f]

Structure of a data folder:

| Folder | Comment |
| --- | --- |
| /linked/example\_data/annotation | The annotated masks of input images. |
| /linked/example\_data/hdversion | The high quality version of input images (e.g. 512x512). |
| /linked/example\_data/input | Input images fed into the linked model (e.g. 224x224). |
| /linked/example\_data/order | To determine chronological order of input images. The content of images does not matter. Make sure the file name of the images are identical as those in example\_data/input folder. |
| /linked/example\_data/output | The output of the linked model according to chronological order. |

To use your customized data, follow the data format in /linked/example\_data.

**Trained model**

The variable &#39;model\_seg&#39; in /linked/linked.ipynb represents the segmentation model.

The variable &#39;model&#39; in /linked/linked.ipynb represents the classification model.

To use your own trained models, simply change these variables.

**Use**

After creating your own data folder according to the instructions of the &#39;Data&#39; section, change the variable &#39;folder\_name&#39; in linked.ipynb into your folder&#39;s name.

Run linked.ipynb and go to &#39;output&#39; subfolder to see model result.

**Example output**

1. Output image when the predicted label is &#39;Nerve&#39;:

![alt text](https://i.imgur.com/QycSTR3.png)

The green area represents the predicted mask. In the top-right corner, the predicted label by classification model, the process time for the image, and the IOU value for the segmentation prediction are shown.

1. Output image when the predicted label is not &#39;Nerve&#39;:

![alt text](https://i.imgur.com/smwnjrl.png)

As the classification model does not predict it as &#39;Nerve&#39;, it will not get fed into the segmentation model. No mask is generated.

**Images to video**

To generate video, place image2avi.py into your folder, then run it.
