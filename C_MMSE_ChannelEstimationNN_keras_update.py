# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:07:41 2020
@author: aguboshimec
"""

print ('*******MIMO Channel Estimation using Machine Learning-based Approach (MMSE)*******')
#Chanel Estimation/Prediction with Least Square (4-layer RNeural Network, etc)
import numpy as np
from numpy import mean #used for computing avg. mse
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.utils import plot_model
import sys 
accuracy = [] #store the prediction accuracy per loop (for diff. antenna array sizes)

nt = nr = 4  #number of tx_antennas #number of rx_antennas
dim = nr*nt
batch_size = dim
noOfNodes = 25
training = 6 #training sequence
layer1node = noOfNodes # number of nodes in first layer
layer2anode = noOfNodes # number of nodes in second layer (hidden)
layer2bnode = noOfNodes # number of nodes in third layer (hidden)
layer3node = dim # number of nodes in fourth layer

#epoch between (188 - 195) seems cool for 4by4 ant array?
epoch = 1850
ite = 1000 #This determines the 3rd dimension of the Tensor. With this, we can have: 40 by 4 by 4 Tensor (ie. if ite = 40)
idx = int(ite/2) # the loop index will be the number of the splitted parts of the 3D tensor/array.
Channel_MMSE_all = []
Channel_pred_all = []   
Channel_all = [] #this is the true channel coefficients for every corresponding least_sq estimation.
Noise = []
y_all_N = [] #stores the output of the model with varying noise power
Channel_MMSE_all_N = []  #stores the output of the MMSE Channel with varying noise power #I didnt use this again.

MSE_MMSE_all = [] # stores the MSE values for coeff. of LS solution
MSE_NN_all = [] # stores the MSE values for coeff. RNN estimation

#Channel model: y = Hx + n, #Assumption: Time-Invariant Channel, AWGN noise
# H is the channel matrix coeffs, x is the training data, n is the awgn noise, y is the output received
#Training samples or signals
x = np.random.randn(nt,training) #nt by x traning samples

# Generate or Adding AGWN noise: So, bascially, the noise level is what deteroritate the channel quality. the noise is the only cahning factor here.
noise = np.random.randn(ite,nr,1)

def Channel_dataset(): #used for testing data set
# Channel (idealized without nosie or true channel coefficients)
    for i in range (ite):
        Channel = np.random.randn(nr,nt)# same channel for varying noise and constant noise power. Recall: Its LTI
        y = np.add(np.dot(Channel,x),noise[i])
        
        #Minimum Mean Square estimation = H_mmse = Chanel_MMSE = (Rhh(Rhh + ((1/SNR)Identity_matrix)H_ls
        Channel_MMSE = np.dot((np.dot(y, (np.transpose(x)))),np.linalg.pinv(np.add(np.dot(x,(np.transpose(x))), (0.01*np.identity(nr)))))
        Channel_MMSE_all.append(np.reshape(Channel_MMSE, (dim, 1)))
        Channel_all.append(np.reshape(Channel, (dim, 1)))
        
Channel_dataset() # calls the function defined above.
#splits the tensor or array into two uniques parts (not vectors this time). Comparing the training loss & verification loss curves will help me know underfitting or overfitting
dataSetSplit = np.array_split(Channel_all, 2) 
Channel_v = np.reshape(dataSetSplit[1], (idx,dim))
Channel_t = np.reshape(dataSetSplit[0], (idx,dim))

dataSetSplit_MMSE = np.array_split(Channel_MMSE_all, 2) #splits the tensor or array into two uniques parts (not vectors this time)
Channel_MMSE_v = np.reshape(dataSetSplit_MMSE[1], (idx,dim))
Channel_MMSE_t = np.reshape(dataSetSplit_MMSE[0], (idx,dim))
   
#Building the network: Setting up layers, activation functions, optimizers, and other metrics.
model = Sequential()
model.add(Dense(layer1node, init = 'random_uniform',activation='relu', input_shape =(dim,)))#first layer #I used dense layering for now here
model.add(Dense(layer2anode , init = 'uniform', activation='relu'))# Hidden layer
model.add(Dense(layer2bnode, init = 'random_uniform', activation='relu'))#Hidden layer, 
model.add(Dense(layer3node, init = 'uniform', activation='linear',  input_shape = (dim,)))  #Output layer,
model.compile(optimizer = 'adam', loss = 'mse')

#train the model now:

NN_mf = model.fit(Channel_MMSE_t, Channel_t, validation_data = (Channel_MMSE_v, Channel_v), epochs=epoch, batch_size =  batch_size, verbose= 1)  
#Evaluting performance with varying mse vs snr: 
#Obtained a vector with varying noise power
start = 15
stop = 0.1
stepsize = ((stop - start)/(idx-1)) # i divided by 'cos I wanted to reduce the length of the vector. nothing really technical
noise_pw = np.arange(start, stop, stepsize) #Generates vector with elements used as varying noise power

SNR = np.reciprocal(noise_pw) # SNR is the reciprocal of noise power
print ('**'*8,'SNR is reciprocal of the noise power: see table below','**'*8)
noise_pw =  np.reshape(noise_pw, (-1))
SNR =  np.reshape(SNR, (-1))
print(np.c_[noise_pw, SNR])

noise_= np.random.randn(nr,1)
#Obtaining the overall noise vector with its varying power:
#To show the noise vectors multiplied by noise powers respectively/individually
for element in noise_pw:
    #print(i, end=', ')
    noise__ = [element]*noise_ # Generated Noise Vector (with varying noise level
    Noise.append(noise__)
    
#Generate new Test Data/Channel Coefficient. This will help give a proof of ability of model to generalize:
#transmit samples or signals, x_Test
Channel_test = np.random.randn(nr,nt)
# Recall: y = Hx + n
for k in range(len(Noise)):
    y_N = np.add(np.dot(Channel_test,x),Noise[k])
    y_all_N.append(y_N)

#Perform MMSquareError of the channel (with varying noise power). 
#MMSquareError estimation 
    Channel_MMSE_N = np.dot((np.dot(y_all_N[k], (np.transpose(x)))),np.linalg.pinv(np.add(np.dot(x,(np.transpose(x))), (3*np.identity(nr)))))
    Channel_MMSE_all_N.append((np.reshape(Channel_MMSE_N, (1,dim))))
    #predict the trained model
    Channel_pred = model.predict(Channel_MMSE_all_N[k], batch_size = idx)
    Channel_pred_all.append(Channel_pred)

for mse in range(idx-1):
    hNN_pred= np.reshape(Channel_pred_all[mse],(-1)) #reshapes or flattens the vector to allow being used for plotting
    hMMSE = np.reshape(Channel_MMSE_all_N[mse],(-1))
    h = np.reshape(Channel_test, (-1))
    MSE1 = np.mean((h - hNN_pred)**2) #to obtain the MSE = (mean(pow(hLS - h), 2))
    MSE2 = np.mean((h - hMMSE)**2) #to ocompute the MSE. Same as above. Considered the LS Coeff without varying noise power
    MSE_NN_all.append(MSE1)
    MSE_MMSE_all.append(MSE2)

c_idx = idx-2 #choose channel index to view
print("TrueChannel=%s, MMSEChannel=%s, PredictedMMSEChannel=%s" % (np.reshape(Channel_test,(nr,nt)), Channel_MMSE_all_N[c_idx], Channel_pred_all[c_idx]))
      
ite_t = 400 #the number of test channel for which i will compute the average mse
ccchannel = np.random.randn(ite_t,nr,nt)
y_all_nnn = [] 
Channel_MMSE_all_nn = []
Channel_pred_all_nn = []

#generate SNR with same length as the channels i have in order to correctly make a plot: #Evaluting performance with varying mse vs snr:  #Obtained a vector with varying noise power
start_ = 15
stop_ = 0.1
stepsize_ = ((stop_ - start_)/(ite_t)) # i divided by 'cos I wanted to reduce the length of the vector. nothing really technical
noise_pw_ = np.arange(start_, stop_, stepsize_) #Generates vector with elements used as varying noise power
SNR_ = np.reciprocal(noise_pw_) # SNR is the reciprocal of noise power
noise_pw_ =  np.reshape(noise_pw_, (-1))
SNR_ =  np.reshape(SNR_, (-1))
#print(np.c_[noise_pw_, SNR_])
Noise_ = []
for element in noise_pw_:
    #print(i, end=', ')
    noise__cc = [element]*noise_ # Generated Noise Vector (with varying noise level
    Noise_.append(noise__cc)

#this computes same noise, but different channel
for nn in range(len(Noise_)):
    for cc in range (len (ccchannel)):
        y_nn = np.add((np.dot(ccchannel[cc],x)),Noise_[nn])#for each iterate over all the channels
        y_all_nnn.append(y_nn)

#next compute the ls for the output
for lll in range (len(y_all_nnn)):
    Channel_MMSE_nn = np.dot((np.dot(y_all_nnn[lll], (np.transpose(x)))),np.linalg.pinv(np.add(np.dot(x,(np.transpose(x))), (3*np.identity(nr)))))  
    Channel_MMSE_all_nn.append(np.reshape(Channel_MMSE_nn, (1,dim)))
        #predict the trained model
    Channel_pred_nn = model.predict(Channel_MMSE_all_nn[lll], batch_size = idx)
    Channel_pred_all_nn.append(Channel_pred_nn)

#I basically, had to loop over all the channels with varying SNR individually.
Channel_MMSE_all_nnA = []
Channel_pred_all_nnA = []

Channel_MMSE_all_nn_sorted = [ Channel_MMSE_all_nn[i:i+ite_t] for i in range(0, len(Channel_MMSE_all_nn), ite_t) ] ##
[Channel_MMSE_all_nnA.extend(Channel_MMSE_all_nn_sorted[pp]) for pp in range(0,ite_t)]

Channel_NN_all_nn_sorted = [ Channel_pred_all_nn[i:i+ite_t] for i in range(0, len(Channel_pred_all_nn), ite_t) ] ##
[Channel_pred_all_nnA.extend(Channel_NN_all_nn_sorted[pp]) for pp in range(0,ite_t)]

avgMSE_MMSE_all = [] 
avgMSE_NN_all = []  

aa = 0
for eee in range(len(ccchannel)):
    hNN_pred_ = np.reshape(Channel_pred_all_nnA[eee+aa],(-1)) #reshapes or flattens the vector to allow being used for plotting
    hMMSE_ = np.reshape(Channel_MMSE_all_nnA[eee+aa],(-1))
    hc = np.reshape(ccchannel[eee], (-1))
    MSE1_ = np.mean((hc - hNN_pred_)**2) #to obtain the MSE = (mean(pow(hLS - h), 2))
    MSE2_ = np.mean((hc - hMMSE_)**2) #to ocompute the MSE. Same as above. Considered the LS Coeff without varying noise power
    avgMSE_MMSE_all.append(MSE2_)
    avgMSE_NN_all.append(MSE1_)
    aa = aa+len(ccchannel)
    
#Determine how close the predictions are to the real-values with some toleranace
CompareResult = np.isclose(Channel_MMSE_all_N[c_idx], Channel_pred_all[c_idx], rtol=0.2) #toleranace of +-0.2
print (CompareResult)
correct_pred = np.count_nonzero(CompareResult)
total_number = CompareResult.size
Accuracy = (correct_pred/total_number)*100
accuracy.append(Accuracy)
print ('Prediction Accuracy of', Accuracy,'%')

#Evaluate the performance of trained model using just one Channel matrix.  
plt.plot(noise_pw[::-1], MSE_MMSE_all)
plt.plot(noise_pw[::-1], MSE_NN_all)
plt.title('Graph of MSE with varying SNR (after training)')
plt.ylabel('Mean Square Error')
plt.xlabel('SNR')
plt.legend(['MMSE', 'NN_Pred'], loc='upper left')
plt.grid(b=None, which='major', axis='both')
plt.show()

#Evaluate the performance of Average MSE from the trained model.  
plt.plot(noise_pw_[::-1], avgMSE_MMSE_all)
plt.plot(noise_pw_[::-1], avgMSE_NN_all)
plt.title('Graph of Average MSE with varying SNR (after training)')
plt.ylabel('Avg. Mean Square Error')
plt.xlabel('SNR')
plt.legend(['MMSE', 'NN_Pred'], loc='upper left')
plt.grid(b=None, which='major', axis='both')
plt.show()

#Visualization after training and testing #To see the performance of the 3rd channel coefficient only
#a good overlap means good performance.
Channel_LS_test = np.reshape(Channel_MMSE_all_N[-1], (-1))
Channel_pred = np.reshape(Channel_pred_all[-1], (-1))
Channel_test = np.reshape(Channel_test, (-1))
plt.plot(Channel_LS_test, '*-')
plt.plot(Channel_pred, '.-')
plt.plot(Channel_test, ',-')
plt.ylabel('amplitude')
plt.xlabel('channel coefficient')
plt.title('Plot of Test Channel Data (MMSE) & its Predicted Channel')
plt.legend(['MMSEChannel', 'PredictedMMSEChannel', 'TrueChannel'], loc='upper left')
plt.grid(b=None, which='major', axis='both')
plt.show()

plt.plot(NN_mf.history['loss'])
plt.plot(NN_mf.history['val_loss'])
plt.title('Graph of Training Loss & its Validation Loss  - MMSE')
plt.ylabel('Loss')
plt.xlabel('No. of Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.grid(b=None, which='major', axis='both')
plt.show()

#More to visualization: show the sequential layers layers
plot_model(model, show_shapes=True, show_layer_names=True, to_file='NNmodel.png')
from IPython.display import Image
Image(retina=True, filename='NNmodel.png') #saves the picture inot the folder-.py collocation

#more to visualization of the model: #To obtain the weights and biases at each layer:
#Note: Layers apart from Layer 1 and Layer 3 are the hidden layers.
summary = model.summary()
TrainedWeight1 = model.layers[0].get_weights()[0]
TrainedBias1 = model.layers[0].get_weights()[1]
#print("trained weight of layer1 =", TrainedWeight1)
#print("trained bias of layer1 =", TrainedBias1)

TrainedWeight2a = model.layers[1].get_weights()[0]
TrainedBias2a = model.layers[1].get_weights()[1]
#print("trained weight of layer2 =", TrainedWeight2)
#print("trained bias of layer2 =", TrainedBias2)

TrainedWeight2b = model.layers[2].get_weights()[0]
TrainedBias2b = model.layers[2].get_weights()[1]
#print("trained weight of layer2 =", TrainedWeight2)
#print("trained bias of layer2 =", TrainedBias2)

TrainedWeight3 = model.layers[3].get_weights()[0]
TrainedBias3 = model.layers[3].get_weights()[1]
#print("trained weight of layer2 =", TrainedWeight3)
#print("trained bias of layer2 =", TrainedBias3)

#this will create the network topology or achitecture that i modelled. 
#so, in case you get a graphviz exectuabel error, use this llink (https://www.youtube.com/watch?v=q7PzqbKUm_4) to fix it. Cheers.
#from ann_visualizer.visualize import ann_viz;
#ann_viz(model, filename="RNNwithKeras", title="Neural Network Topology for Channel Estimation")

#Note: If at any point the MSE curve descreases, and then increases again, this could indicate overfitting? So, in this case, i reduce my epoch value
#or try to tweak other hyperparameters, number of nodes.

