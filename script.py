import numpy as np
import cv2
import os

def readimg(filename,dir):
    filename=dir+str(filename)
    img=cv2.imread(filename,0)
    li=[]
    for i in img:
        li.extend(i)
    for i in range(len(li)):
        if li[i]>(250):
            li[i]=1
        else:
            li[i]=-1
    return li

def readall(imgsname,dir):
    X=[]
    for j in imgsname:
        X.append(readimg(j,dir))
    return X
        
def ran(mean, sigma, col,row): 
    s = np.random.normal(mean, sigma, col*row)
    mat= np.reshape(s, (col, row))
    return mat

def softmax(Y):
    SY=np.exp(Y)/(np.exp(Y[0])+np.exp(Y[1]))
    return SY
    
def loss(trainlab,sofm):
    cross=-trainlab[0]*np.log(sofm[0])-trainlab[1]*np.log(sofm[1])
    return sum(cross)/len(cross)

def gradient_W(Xt,label,softm):
    grad_W=-((label-softm).dot(Xt))/len(label[0])
    return grad_W,(label-softm)
    
def gradient_B(label,softm):
    grad_B=-(np.sum(label,axis=1)-np.sum(softm,axis=1))/len(label[0])
    return grad_B    
   
   
A_directory=r"C:\\path\\to\\directoryA\\"
B_directory=r"C:\\path\\to\\directoryB\\"
A_imgs=os.listdir(A_directory)
B_imgs=os.listdir(B_directory)
Afile=readall(A_imgs,A_directory)
Bfile=readall(B_imgs,B_directory)

#Read images. We have 100 images for each category, first 90 images use for training, and 10 for testing
train=np.asarray(Afile[0:90]+Bfile[0:90]).transpose()
test=np.asarray(Afile[90:100]+Bfile[90:100]).transpose()
train_lab=lists = np.asarray([[0,1] for _ in range(90)]+[[1,0] for _ in range(90)]).transpose()
test_lab=lists = np.asarray([[0,1] for _ in range(10)]+[[1,0] for _ in range(10)]).transpose()

#Training
W=np.asarray(ran(0,1,len(train_lab),len(train)))
B=ran(0,1,len(train_lab),1)
alpha1 = 1.0 #learning rate
mu=0.9
lastW=np.zeros((2,784))
recordL=[]
generation=1000
for i in range(generation):
    #print "generation",i
    alpha=alpha1*(10000-i)/10000
    momentum=np.multiply(mu,lastW)   
    Y=W.dot(train)+B
    soft=softmax(Y)
    l=loss(train_lab,soft)
    gradw,subt=gradient_W(train.transpose(),train_lab,soft)
    bias=gradient_B(train_lab,soft)
    bias= np.reshape(bias, (2, 1))
    W = W-np.multiply(alpha,gradw)+momentum
    B = B-np.multiply(alpha,bias)
    recordL.append(l)
    lastW=gradw
print 'done'
        
#Testing
testY=W.dot(test)+B
outtest=softmax(testY) #test result
W0=  np.reshape(W[0], (28, 28))
W1=  np.reshape(W[1], (28, 28))
