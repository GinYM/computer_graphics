import numpy as np
import sys,os,random,math
from scipy import misc
import matplotlib.pyplot as plt

caffe_root = './caffe_ext/'
sys.path.insert(0,caffe_root+'python')
model_file = './model/projection.caffemodel'
model_def_file = './model/projection.prototxt'
name_dir = './name_list.txt'
target_dir = './target_name.txt'
image_dir = '/media/gin/hacker/computer_graphics/TrainVal/VOCdevkit/VOC2011/JPEGImages/'

import caffe

name_list = []
target_list = []
target_fc6 = []
image_fc6 = []
result = []

caffe.set_mode_cpu()

net = caffe.Net(model_def_file,model_file,caffe.TEST)
net.blobs['data'].reshape(1,3,96,96)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})



def load_image():
    f = open(name_dir,'r')
    f1 = open(target_dir,'r')
    for name in f:
        name_list.append(name[:-1])
    f.close()
    for name in f1:
        target_list.append(name[:-1])
    f1.close()

def pre_process_image(name):
    im=caffe.io.load_image(image_dir+name)
    im = transformer.preprocess('data',im)
    chantokeep=random.randint(0,2);   #randomly delete some color
    for i in range(0,3):
        if i == chantokeep:
            im[:,:,i]-=np.mean(im[:,:,i])
        else:
            im[:,:,i]=np.random.uniform(0,1,(im.shape[0],im.shape[1]))- .5
    im=im / np.sqrt(np.mean(np.square(im))) * 50   #normalized, I don't know whether times 50 is necessary
    im = im.transpose(2,0,1)    #make sure that the image has the corresponding dimension
    im = im.reshape((1,3,96,96))
    net.blobs['data'].data[...] = im
    res = net.forward()
    out = net.blobs['fc6'].data
    return out.tolist()

def cal_target():
    for name in target_list:
        tmp = pre_process_image(name)  
        tmp.append(name)   #!!remember that I add file name to it!!!
        target_fc6.append(tmp)

def cal_image():
    for name in name_list:
        print "calculate "+name+"\n"
        if name in target_list:
            continue
        tmp = pre_process_image(name)
        tmp.append(name)    #!!remember that I add file name to it!!!
        image_fc6.append(tmp)
        
def cal_nc(tmp1,tmp2):   #calculate the normalized correlation
    r1 = 0
    r2 = 0
    r3 = 0
    for index in range(len(tmp1)):
        r1 += tmp1[index]*tmp2[index]
        r2 += tmp1[index]*tmp1[index]
        r3 += tmp2[index]*tmp2[index]
    res = r1/math.sqrt(float(r2*r3))
    return res

def cal_all_nc():
    tmp = []
    f = open('output.txt','w')
    for fc6_t in target_fc6:
        tmp = []
        for fc6_i in image_fc6:
            
            tmp_res = cal_nc(fc6_t[0],fc6_i[0])
           # tmp_res.append(fc6_i[-1])
            tmp.append([tmp_res,fc6_i[-1]])
        result.append(sorted(tmp,key = lambda fun: fun[0:-1],reverse=True))
        #print result
        #print fc6_t[-1]
        f.write(fc6_t[-1]+"\n")
        #print result
        for ele in result[0]:
            f.write(str(ele[0])+" "+ele[1]+"\n")
            

def main():
    load_image()
    print "calculate target fc6\n"
    cal_target()
    print "calculate image fc6\n"
    cal_image()
    print "calculate normalized correlation"
    cal_all_nc()
    print "image is loaded\n"

if __name__ == '__main__':
    main()




