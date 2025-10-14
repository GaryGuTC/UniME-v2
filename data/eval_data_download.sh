#! Just reference, please download and correct them by yourself
sudo apt update
sudo apt install aria2

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

################################### MMEB evaluation set ##################################
# git clone https://huggingface.co/datasets/TIGER-Lab/MMEB-eval
bash hfd.sh TIGER-Lab/MMEB-eval --dataset --tool aria2c -x 10
cd MMEB-eval
unzip images.zip -d eval_images/

################################### Flickr test ##################################
git clone https://huggingface.co/datasets/royokong/flickr30k_test

###################################  COCO-test ##################################
git clone https://huggingface.co/datasets/royokong/coco_test

################################### Urban1K ##################################
git clone https://huggingface.co/datasets/BeichenZhang/Urban1k

################################### ShareGPT4V ##################################
git clone https://www.modelscope.cn/datasets/hjh0119/sharegpt4v-images.git # image
git clone https://huggingface.co/datasets/Lin-Chen/ShareGPT4V