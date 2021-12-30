# Pixel Difference Network

From paper: Pixel Difference Networks for Efficient Edge Detection  
   
Reference: Su, Z., Liu, W., Yu, Z., Hu, D., Liao, Q., Tian, Q., Pietikäinen, M. and Liu, L., 2021. Pixel Difference Networks for Efficient Edge Detection.

## Data Visualization and Image Processsing Final Project

Done by: 
Athira Shankar, Akarawint Chawalitanont , Ali Akouch

## Prerequisites:
pytorch 1.9 
cuda 10.2
Python 3.7+
numpy...

## This code line will be used for having cudatoolkit that will be used for this project.
```pip install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```

## The code was tested and runs using Windows.

## Installation
After downloading all needed libraries.... 

* Download HED-BSDS and PASCAL data using:

```wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz```

```wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz``` 

*  Extract HED-BSDS.tar.gz to /path/to/BSDS500/HED-BSDS

* Extract PASCAL.tar.gz to /path/to/BSDS500/PASCA


## For testing the edge detection 

* Create a folder /path/to/BSDS500/Custom_images

* Add your own images that you want to detect their edges inside the Custom_images file.

* For edge detection testing, add this code to the terminal:

>python main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir /path/to/savedir --datadir /path/to/custom_images --dataset Custom --evaluate /path/to/table5_pidinet/save_models/saved_model.pth --evaluate-converted

* Example:

>python main.py --model pidinet_converted --config carv4 --sa --dil -j 4 --gpu 0 --savedir "C:/Users/THINKPAD/PycharmProjects/pidinet/data/BSDS500" --datadir "C:/Users/THINKPAD/PycharmProjects/pidinet/data/BSDS500/custom_images" --dataset Custom –evaluate "C:/Users/THINKPAD/PycharmProjects/pidinet/trained_models/table5_pidinet.pth" --evaluate-converted



## The results will be added to a file called :
* eval_results
