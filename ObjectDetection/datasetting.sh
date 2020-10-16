#mkdir -p ./data
#mkdir -p ./data/VOC

#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
#wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

#tar -xvf ./VOCtrainval_06-Nov-2007.tar
#tar -xvf ./VOCtrainval_11-May-2012.tar
#tar -xvf ./VOCtest_06-Nov-2007.tar

#rm ./VOCtrainval_06-Nov-2007.tar
#rm ./VOCtrainval_11-May-2012.tar
#rm ./VOCtest_06-Nov-2007.tar

#mv ./VOCdevkit ./data/VOC

#python ./datasetting_voc.py

#wget https://www.dropbox.com/s/emjqpg6wfhp1qfn/WeaponData.zip?dl=0

#mv ./WeaponData.zip?dl=0 ./WeaponData.zip

#unzip ./WeaponData.zip

#mv ./Weapon ./data

#rm ./WeaponData.zip

python ./datasetting_weapon.py