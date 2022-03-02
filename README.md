# Flare Transformer
* An overview of the Flare Transformer.

![fig1](https://user-images.githubusercontent.com/75234574/154173007-d11c61d1-3541-4519-974b-a077fcceaa3b.png)
<!--  ![fig1](https://user-images.githubusercontent.com/75234574/148938753-87bcdde5-b7ad-4d6a-9783-7eaa15ca5e52.png) -->

## 1. Requirements
* Python 3.6.9
* Ubuntu 18.04

## 2. Installation
* In the following procedure, ```~/work``` is assumed to be used as a working directory.
```
$ cd ~/work
$ git clone URL
$ cd flare_transformer
$ pip install -U pip
$ pip install -r require.txt
```

## 3. Download data
* The required data files should be put under ```data/``` folder.
* Visit http://wdc.nict.go.jp/IONO/wdc/solarflare/index.html and download the database of physical features.
```
$ cd ~/work/flare_transformer
$ mv ~/data.zip data/
$ unzip data/data.zip
```

* Visit https://sdo.gsfc.nasa.gov/data/ and download hourly magnetograms.
```
$ mv ~/magnetogram_images.tar.gz data/
$ tar -zxvf data/magnetogram_images.tar.gz
```


## 4. Preprocess
* Preprocess the physical features and magnetogram images by the following procedure.
```
$ python src/preprocess.py
```
* The following data files should be created under ```data/```.
  *  data/data_20XX_magnetogram.npy
  *  data/data_20XX_feat.csv
  *  data/data_20XX_window.csv
  *  data/data_20XX_label.csv


## 5. Training
```
$ cd ~/work/flare_transformer
$ ./train.sh
```
*  Training example is shown in ```train.sh```. ```--params``` should be specified according to your settings.
*  ```--params``` takes a path of a JSON file. In the JSON file, values for parameters such as learning rate, batch size, etc. should be specified.  
*  A sample of a JSON file is shown in ```params/params2017.json```.

## 6. Results
* **Quantitative Results**
  * We report the model performance of the Flare Transfomer as follows: 

    |  | GMGS | <img src="https://latex.codecogs.com/svg.image?{\rm&space;TSS}_{\rm&space;\geq&space;M}" title="{\rm TSS}_{\rm \geq M}" /> | <img src="https://latex.codecogs.com/svg.image?{\rm&space;BSS}_{\rm&space;\geq&space;M}" title="{\rm BSS}_{\rm \geq M}" />  | 
    | --- | --- | --- | --- |
    | Flare Transformer | 0.503 ± 0.059 | 0.530 ± 0.112 | 0.082 ± 0.974 |

  * We also report the confusion matrix for the 2017 test set as follows:
 
   ![スクリーンショット (51)](https://user-images.githubusercontent.com/75234574/154173774-eea773a3-ff15-4582-9644-fcc738a7643a.png)
  


* **Qualitative Results**
  * The figure below shows line-of-sight magnetograms from September 3th, 2017 23:00 UT to September 5th, 2017, 23:00 UT. An X-class solar flare occurred at 12:02 on September 6, 2017, and the model was able to predict the correct maximum solar flare class.
  
    ![magnetogram_256](https://user-images.githubusercontent.com/75234574/148938052-5d2a017e-c8fd-4f4f-9c10-0226e447c939.gif)

<!-- * Memo
  * 2021/03/02 feat/09から分離
  * コードの整理が目的
  * pretrain.pyなどを削除 -->



<!-- ## 6. Additional physical features
a | b
--- | --- |
dt12Bmax | Time derivative of Bmax over 12 hr
dt12Area | Time derivative of Area over 12 hr
dt12USflux | Time derivative of USflux over 12 hr
dt02 Bmax | Time derivative of Bmax over 2 hr
dt02 Area | Time derivative of Area over 2 hr
dtUsflux02 | Time derivative of USflux over 2 hr
dt24VUSflu | Time derivative of USflux over 24 hr (measured by vector magnetogram)
dt24VArea | Time derivative of Area over 24 hr (measured by vector magnetogram)
dt24CHArea | Time derivative of CHArea over 24 hr
dt24 CHAll | Time derivative of CHAll over 24 hr
dt24 CHMax | Time derivative of CHMax over 24 hr
dt24TotNL | Time derivative of TotNL over 24 hr
dt24MaxNL | Time derivative of MaxNL over 24 hr
dt24 NumNL |Time derivative of NumNL over 24 hr
dt24TotUSJz | Time derivative of TotUSJz over 24 hr
dt24TotUSJh | Time derivative of TotUSJh over 24 hr
dt24TotFX | Time derivative of TotFX over 24 hr
dt24TotFY | Time derivative of TotFY over 24 hr
dt24TotFZ | Time derivative of TotFz over 24 hr
dt24SavNCPP | Time derivative of SavNCPP over 24 hr
dt24ABSnJzh | Time derivative of ABSnJzh over 24 hr
dt24TotBSQ | Time derivative of TotBSQ over 24 hr
dt24 Max grab | Time derivative of Max.∇Bz over 24 hr
dt24 MaxdxBz | Time derivative of Max. dBz/dx over 24 hr
dt24 MaxdyBz | Time derivative of Max. dBz/dy over 24 hr
dt24 MeanGBz | Time derivative of MeanGBz over 24 hr
dt24 MeanGBh | Time derivative of MeanGBh over 24 hr
dt24 MeanGBt | Time derivative of MeanGBt over 24 hr
dt24MeanGAM | Time derivative of MeanGAM over 24 hr
dt24MeanJzd | Time derivative of MeanJzd over 24 hr
dt24MeanJzh | Time derivative of MeanJzh over 24 hr -->

