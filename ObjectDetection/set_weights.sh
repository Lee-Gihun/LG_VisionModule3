mkdir -p ./weights

wget https://www.dropbox.com/s/pkpnzr10kpifvv9/pretrained_SSD300.pth?dl=0

mv ./pretrained_SSD300.pth?dl=0 ./pretrained_SSD300.pth
mv ./pretrained_SSD300.pth ./weights

sleep 3

wget https://www.dropbox.com/s/p5btuqh8cn1wdpw/weapon_weights30.pth?dl=0

mv ./weapon_weights30.pth?dl=0 ./weapon_weights30.pth
mv ./weapon_weights30.pth ./weights