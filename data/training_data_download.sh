#! Just reference, please download and correct them by yourself

sudo apt update
sudo apt install aria2

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install

################################### MMEB training set ##################################
bash hfd.sh TIGER-Lab/MMEB-train --dataset --tool aria2c -x 10
# git clone https://huggingface.co/datasets/TIGER-Lab/MMEB-train
cd MMEB-train
python unzip_file.py

## UniME-V2 data
### Qwen2.5 VL 7B score
# bash hfd.sh TIGER-Lab/MMEB-train --dataset --tool aria2c -x 10

### Intern VL 8B score
# bash hfd.sh TIGER-Lab/MMEB-train --dataset --tool aria2c -x 10

### Intern VL 14B score
# bash hfd.sh TIGER-Lab/MMEB-train --dataset --tool aria2c -x 10
