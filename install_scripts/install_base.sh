## Base install

touch ~/.no_auto_tmux # disable termux, optional

apt install -y wget curl inetutils-ping

# Check where we are - Chinese servers need mirrors due to the firewall https://www.cnblogs.com/fang-d/p/17832995.html
apt install -y jq
my_ip=$(curl ipinfo.io/ip)
country=$(curl ipinfo.io/$my_ip | jq -r '.country')

if [[ $country == "CN" ]]; then
# Download mamba from mirror, set mirror for pip
echo "Using mirrors for CN"
wget https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-`uname -m`.sh
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
git config --global http.version HTTP/1.1
else
wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Miniforge3-Linux-`uname -m`.sh
fi

# Then
chmod +x Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -f -p ~/miniforge3

~/miniforge3/bin/mamba init
# Would be better to log out and reconnect rather than re-run .bashrc
source ~/.bashrc

# For docker containers, add optional packages
if [[ `id -u` == 0 ]]; then
apt update
apt install build-essential file nano inetutils-ping psutils unzip screen less curl git -y
apt install ffmpeg libsm6 libxext6  -y
fi

## Datasets - VOC
mkdir datasets
# Download and extract VOC trainval data
curl -O http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar && tar -xf VOCtrainval_06-Nov-2007.tar
curl -O http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar && tar -xf VOCtrainval_11-May-2012.tar
mv VOCdevkit/VOC2007 datasets/VOC2007
mv VOCdevkit/VOC2012 datasets/VOC2012

# Download and extract test data
curl -O http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar && tar -xf VOCtest_06-Nov-2007.tar
cp -r VOCdevkit/VOC2007/* datasets/VOC2007/
rm -r VOCdevkit

# Download and extract few shot annotations
curl -O https://storage.googleapis.com/pascal_voc_few_shot_ann/voc_few_shot_ann_allseeds.zip
unzip -q voc_few_shot_ann_allseeds.zip
mv vocsplit datasets/

## Datasets - COCO

mkdir datasets/coco
mkdir datasets/coco/trainval2014
mkdir datasets/coco/val2014

# Few-shot data split
curl -O https://storage.googleapis.com/gfsod_exp_data_mirror/cocosplit.zip && unzip cocosplit.zip
mv cocosplit datasets/

# Only downloading seed 0 & 1 for now, rather than full COCO set
# curl -O https://storage.googleapis.com/gfsod_exp_data_mirror/coco_trainval2014_seed0.zip && unzip coco_trainval2014_seed0.zip
# curl -O https://storage.googleapis.com/gfsod_exp_data_mirror/coco_trainval2014_seed1.zip && unzip coco_trainval2014_seed1.zip
# mv trainval2014/ datasets/coco/
# rm coco_trainval2014_seed*.zip

# Alternatively, to download full COCO dataset:
wget http://images.cocodataset.org/zips/train2014.zip && unzip -q train2014.zip
mv train2014 datasets/coco/
mv datasets/coco/train2014 datasets/coco/trainval2014
rm train2014.zip

wget http://images.cocodataset.org/zips/val2014.zip && unzip -q val2014.zip
mv val2014/*  datasets/coco/trainval2014/
rmdir datasets/coco/val2014
# Test images are picked from val2014 but determined by split. To avoid unnecessary duplication, link it to trainval2014
ln -s ~/datasets/coco/trainval2014 -T ~/datasets/coco/val2014
rm val2014.zip
rmdir val2014

# Not downloading test images, as metasplit uses some of train2014+val2014 for training and some of val2014 for testing
#curl -O http://images.cocodataset.org/zips/test2017.zip && unzip test2017.zip
#mv test2017 datasets/coco/



