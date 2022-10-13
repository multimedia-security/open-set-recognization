#!/bin/bash
echo -e "start"
#python proser_proto3.py --alpha 1.5 --gpu 1 --lamda2 1 --lamda1 2
#python proser_proto3.py --alpha 1.5 --gpu 1 --lamda1 3 --lamda2 6 --decay 0.005
#python proser_proto3.py --alpha 1.5 --gpu 1 --lamda1 4 --lamda2 6 --decay 0.005
#python proser_proto3.py --alpha 1.5 --gpu 1 --lamda1 3 --lamda2 9 --decay 0.001
#python proser_proto3.py --alpha 1.5 --gpu 1 --lamda1 3 --lamda2 9 --decay 0.01
#python proser_proto3.py --alpha 3 --gpu 1 --lamda1 1 --lamda2 9 --decay 0.005
#python proser_proto3.py --alpha 3 --gpu 1 --lamda1 3 --lamda2 9 --decay 0.009
#python proser_proto3.py --alpha 4 --gpu 1 --lamda1 3 --lamda2 9 --decay 0.005
#python proser_proto3.py --alpha 1.5 --gpu 1 --lamda2 12
#python proser_proto3.py --alpha 1.5 --gpu 1 --lamda2 13
#python proser_proto3.py --alpha 1.5 --gpu 1 --lamda2 14
#python proser_proto3.py --alpha 1.5 --gpu 1 --lamda2 15
#python proser_proto3.py --alpha 1.5 --gpu 1 --lamda2 16
##python proser_proto3.py --alpha 0.5 --gpu 1 --seed 123
##python proser_proto3.py --alpha 1 --gpu 1 --seed 123
##python proser_proto3.py --alpha 2 --gpu 1 --seed 123

#python proser_proto3.py --novel 9 --gpu 1
python proser_proto3.py --novel 10 --gpu 1
python proser_proto3.py --novel 11 --gpu 1
python proser_proto3.py --novel 12 --gpu 1
python proser_proto3.py --novel 13 --gpu 1
python proser_proto3.py --novel 14 --gpu 1
python proser_proto3.py --novel 15 --gpu 1
python proser_proto3.py --novel 16 --gpu 1
python proser_proto3.py --novel 17 --gpu 1
python proser_proto3.py --novel 18 --gpu 1
python proser_proto3.py --novel 10 11 12 13 14 15 16 17 18 19 20 21 22 --gpu 1
#python proser_proto3.py --novel 14 --gpu 1
#python proser_proto3.py --novel 15 --gpu 1
#python proser_proto3.py --novel 16 --gpu 1
#python proser_proto3.py --novel 17 --gpu 1
#python proser_proto3.py --novel 10 11 12 13 14 15 16 17 --gpu 1
#python pretrain_proto.py

