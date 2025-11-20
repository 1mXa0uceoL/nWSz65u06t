# 17597


## 1. Create Environment


- Make Conda Environment
```
conda create -n Multinex python=3.9 -y
conda activate Multinex
```

- Install Dependencies
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm python_msssim ptflops
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```

&nbsp;

## 2. Prepare Dataset
Download the following datasets: 

LOL-v1 [Proton Drive](https://drive.proton.me/urls/FYX63A8P50#C5P6Yo6ypwwS)

LOL-v2 (Real and Synthetic) [Proton Drive](https://drive.proton.me/urls/C65K75JB6G#8ueDX6M8u9hI)


<details close>
<summary><b> Then organize these datasets as follows: </b></summary>

```
    |--data   
    |    |--LOLv1
    |    |    |--Train
    |    |    |    |--input
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--100.png
    |    |    |    |    |--101.png
    |    |    |    |     ...
    |    |    |--Test
    |    |    |    |--input
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |    |    |--target
    |    |    |    |    |--111.png
    |    |    |    |    |--146.png
    |    |    |    |     ...
    |    |--LOLv2
    |    |    |--Real_captured
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00001.png
    |    |    |    |    |    |--00002.png
    |    |    |    |    |     ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |    |    |--Normal
    |    |    |    |    |    |--00690.png
    |    |    |    |    |    |--00691.png
    |    |    |    |    |     ...
    |    |    |--Synthetic
    |    |    |    |--Train
    |    |    |    |    |--Low
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r000da54ft.png
    |    |    |    |    |   |--r02e1abe2t.png
    |    |    |    |    |    ...
    |    |    |    |--Test
    |    |    |    |    |--Low
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
    |    |    |    |    |--Normal
    |    |    |    |    |   |--r00816405t.png
    |    |    |    |    |   |--r02189767t.png
    |    |    |    |    |    ...
   
```

</details>


&nbsp;                    


## 3. Testing

Download our pre-trained models [Proton Drive](https://drive.proton.me/urls/V6BVHVC5J4#8GuaMudrA2Nd).

```shell
# activate the environment
conda activate Multinex

# LOL-v1
python Enhancement/test_from_dataset.py --opt Options/Multinex_LOL-v1_3-A-3_HybL.yaml --weights path/to/weights.pth --dataset LOL_v1 --self_ensemble

# LOL-v2-real
python Enhancement/test_from_dataset.py --opt Options/Multinex_LOL-v2-real_3-A-3_HybL.yaml --weights path/to/weights.pth --dataset LOL_v2_real --self_ensemble

# LOL-v2-synthetic
python Enhancement/test_from_dataset.py --opt Options/Multinex_LOL-v2-syn_3-A-3_HybL.yaml --weights path/to/weights.pth --dataset LOL_v2_syn --self_ensemble


```

`--self_ensemble` can be removed for faster inference.



- #### Parameter / FLOPs test.


```shell
python basicsr/complexity.py --opt Options/Multinex_LOL-v1_3-A-3_HybL.yaml --warmup 5 --runs 20 --device cuda --resolutions 256x256
```


&nbsp;


## 4. Training


```shell
# activate the enviroment
conda activate Multinex

# LOL-v1
python3 basicsr/train.py --opt Options/Multinex_LOL-v1_3-A-3_HybL.yaml 

# LOL-v2-real
python3 basicsr/train.py --opt Options/Multinex_LOL-v2-real_3-A-3_HybL.yaml  

# LOL-v2-synthetic
python3 basicsr/train.py --opt Options/Multinex_LOL-v2-syn_3-A-3_HybL.yaml  


```
