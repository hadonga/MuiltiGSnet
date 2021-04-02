#!/bin/sh
#-*- coding:utf8 -*-
echo "start dataset generation!"
python gen_clip_sn.py; #windows下编写的脚本文件换行部分是 “/r/n” Linux是 “/n”
echo "start2"
python generate_n_lb.py;
