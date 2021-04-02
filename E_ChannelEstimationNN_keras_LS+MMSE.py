# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:07:41 2020
@author: aguboshimec
"""

print ('*******MIMO Channel Estimation using Machine Learning-based Approach (LS & MMSE)*******')
#Chanel Estimation/Prediction with Least Square (4-layer RNeural Network, etc)
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import sys 
accuracy = [] #store the prediction accuracy per loop (for diff. antena array sizes)

#the next few lines with 'var1 = None' are basically to intialize the variables before I can declare them as global.
#with the global status, i can then reuse them for the MMSE evalation. It makes sense that way since for proper comparison, i have to use the same datasets.
layer1node = None
layer2anode = None
layer2bnode = None
layer3node = None
Chanel_v = None
h = None
batch_size = dim = None #if you want a different batchsize, then separate dim from batchszie, or just uncomment dim as in below
Chanel_t = None
Chanel_test = None
ite = None
ite_t = None
stepSize = None
idx = None
ccchannel = None
Chanel_MMSE_v = None
Chanel_MMSE_t = None
nt = nr = None
#dim = None # uncomment and detach from batchsize if both are not same value
Noise = None
noise_pw_ = None
noise_pw = None
y_all_N = None
y_all_nnn = None
x = None
summary = None


def ChanelEstimation(ant_array, pilot, noOfNodes):
    global noise_pw_, noise_pw, layer1node, nt, dim, layer2anode, layer2bnode, layer2bnode, y_all_nnn, ccchannel,stepSize, layer3node, ite_t, summary, ite, idx, Chanel_MMSE_v, Chanel_MMSE_t, Chanel_v, h, nr, Chanel_t, batch_size, Chanel_test, Noise, x, y_all_N
    nt = nr = ant_array  #number of tx_antenas #number of rx_antenas
    dim = nr*nt
    batch_size = dim
    training = pilot #training sequence
    layer1node = noOfNodes # number of nodes in first layer
    layer2anode = noOfNodes # number of nodes in second layer (hidden)
    layer2bnode = noOfNodes # number of nodes in third layer (hidden)
    layer3node = dim # number of nodes in fourth layer
    
    #epoch between (188 - 195) seems cool for 4by4 ant array?
    epoch = 195
    ite = 8000
    #This determines the 3rd dimension of the Tensor. With this, we can have: 40 by 4 by 4 Tensor (ie. if ite = 40)
    idx = int(ite/2) # the loop index will be the number of the splitted parts of the 3D tensor/array.
    Chanel_LS_all = []
    Chanel_pred_all = []   
    Chanel_all = [] #this is the true chanel coefficients for every corresponding least_sq estimation.
    Chanel_MMSE_all = []
    Noise = []
    y_all_N = [] #stores the output of the model with varying noise power
    Chanel_LS_all_N = []  #stores the output of the LS Chanel with varying noise power
     
    MSE_LS_all = [] # stores the MSE values for coeff. of LS solution
    MSE_NN_all = [] # stores the MSE values for coeff. Rn estimation
    
    #Chanel model: y = Hx + n, #Assumption: Time-Invariant Chanel, AWGN noise
    # H is the chanel matrix coeffs, x is the training data, n is the awgn noise, y is the output received
    #Training samples or signals
    x = np.random.randn(nt,training) #nt by x traning samples
    
    # Generate or Adding AGWN noise: So, bascially, the noise level is what deteroritate the chanel quality. the noise is the only cahning factor here.
    noise = np.random.randn(ite,nr,1)
    
    def Chanel_dataset(): #used for testing data set
    # Chanel (idealized without nosie or true chanel coefficients)
        for i in range (ite):
            Chanel = np.random.randn(nr,nt)# same chanel for varying noise and constant noise power. Recall: Its LTI
            #Perform MMSE Estimation. (Analytical Solution is: (inverse(Rhh + (transpose(x*(transpose(x)))))*hLS)
            y = np.add(np.dot(Chanel,x),noise[i])
            
            
            
            
            #Least Square Estimation
            Chanel_LS = np.dot(y,(np.linalg.pinv(x)))
            #Minimum Mean Square estimation = 
            Chanel_MMSE = np.dot((np.dot(y, (np.transpose(x)))),np.linalg.pinv(np.add(np.dot(x,(np.transpose(x))), (4*np.identity(nr)))))
            Chanel_MMSE_all.append(np.reshape(Chanel_MMSE, (dim, 1)))
            Chanel_LS_all.append ((np.reshape(Chanel_LS, (dim, 1))))
            Chanel_all.append(np.reshape(Chanel, (dim, 1)))
            
    Chanel_dataset() # calls the function defined above.
    #splits the tensor or array into two uniques parts (not vectors this time). Comparing the training loss & verification loss curves will help me know underfitting or overfitting
    dataSetSplit = np.array_split(Chanel_all, 2) 
    Chanel_v = np.reshape(dataSetSplit[1], (idx,dim))
    Chanel_t = np.reshape(dataSetSplit[0], (idx,dim))
    
    dataSetSplit_LS = np.array_split(Chanel_LS_all, 2) #splits the tensor or array into two uniques parts (not vectors this time)
    Chanel_LS_v = np.reshape(dataSetSplit_LS[1], (idx,dim))
    Chanel_LS_t = np.reshape(dataSetSplit_LS[0], (idx,dim))
    
    #i could not have splitted into 2 uneven parts 'cos the graph shows lines which has to be equal. #I decided to do the splitting here, and then call the variable from the MMSE script.
    dataSetSplit_MMSE = np.array_split(Chanel_MMSE_all, 2) #splits the tensor or array into two uniques parts (not vectors this time)
    Chanel_MMSE_v = np.reshape(dataSetSplit_MMSE[1], (idx,dim))
    Chanel_MMSE_t = np.reshape(dataSetSplit_MMSE[0], (idx,dim))
           
    #Building the network: Setting up layers, activation functions, optimizers, and other metrics.
    model = Sequential()
    model.add(Dense(layer1node, init = 'random_uniform',activation='relu', input_shape =(dim,)))#first layer #I used dense layering for now here
    model.add(Dense(layer2anode , init = 'uniform', activation='relu'))# Hidden layer
    model.add(Dense(layer2bnode, init = 'random_uniform', activation='relu'))#Hidden layer, 
    model.add(Dense(layer3node, init = 'uniform', activation='linear',  input_shape = (dim,)))  #Output layer,
    model.compile(optimizer = 'adam', loss = 'mse')
    
    #train the model now:
    n_mf = model.fit(Chanel_LS_t, Chanel_t, validation_data = (Chanel_LS_v, Chanel_v), epochs=epoch, batch_size =  batch_size, verbose= 1)
    
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
    
    #Generate new Test Data/Chanel Coefficient. This will help give a proof of ability of model to generalize:
    #transmit samples or signals, x_Test
    Chanel_test = np.random.randn(nr,nt)
    
    # Recall: y = Hx + n
    for k in range(len(Noise)):
        y_N = np.add(np.dot(Chanel_test,x),Noise[k])
        y_all_N.append(y_N)
    
    #Perform Least_Square of the chanel (with varying noise power). 
    #Least Square estimation = H_ls = Chanel_LS = (y*x_transpose(pinv(x*x_transpose))
        Chanel_LS_N = np.dot(y_all_N[k],(np.linalg.pinv(x)))
        
        Chanel_LS_all_N.append(np.reshape(Chanel_LS_N, (1,dim)))
        #predict the trained model
        Chanel_pred = model.predict(Chanel_LS_all_N[k], batch_size = idx)
        Chanel_pred_all.append(Chanel_pred)
    
    for mse in range(idx-1):
        hn_pred= np.reshape(Chanel_pred_all[mse],(-1)) #reshapes or flattens the vector to allow being used for plotting
        hLS = np.reshape(Chanel_LS_all_N[mse],(-1))
        h = np.reshape(Chanel_test, (-1))
        MSE1 = np.mean((h - hn_pred)**2) #to obtain the MSE = (mean(pow(hLS - h), 2))
        MSE2 = np.mean((h - hLS)**2) #to ocompute the MSE. Same as above. Considered the LS Coeff without varying noise power
        MSE_NN_all.append(MSE1)
        MSE_LS_all.append(MSE2)
    
    c_idx = idx-2 #choose chanel index to view
    print("TrueChanel=%s, LeastSqChanel=%s, PredictedLSChanel=%s" % (np.reshape(Chanel_test,(nr,nt)), Chanel_LS_all_N[c_idx], Chanel_pred_all[c_idx]))
                
    ite_t = 300 #the number of test channel for which i will compute the average mse
    ccchannel = np.random.randn(ite_t,nr,nt)
    y_all_nnn = [] 
    Channel_LS_all_nn = []
    Channel_pred_all_nn = []
    #generate SNR with same length as the channels i have in order to correctly make a plot: #Evaluting performance with varying mse vs snr:  #Obtained a vector with varying noise power
    start_ = 15
    stop_ = 0.01
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
            y_nn = np.add((np.dot(ccchannel[cc],x)),Noise_[nn]) #for each iterate over all the channels
            y_all_nnn.append(y_nn)
    
    #next compute the ls for the output: np.dot(y,(np.linalg.pinv(x)))
    for lll in range (len(y_all_nnn)):
        Channel_LS_nn = np.dot(y_all_nnn[lll],(np.linalg.pinv(x))) #computes the LS Channel realization for y_all_nnn 
        Channel_LS_all_nn.append(np.reshape(Channel_LS_nn, (1,dim)))
        #predict the trained model
        Channel_pred_nn = model.predict(Channel_LS_all_nn[lll], batch_size = idx) #predicts the LS Channel realizations also.
        Channel_pred_all_nn.append(Channel_pred_nn)
        
    #I basically, had to loop over all the channels with varying SNR individually.
    Channel_LS_all_nnA = []
    Channel_NN_all_nnA = []
    
    Channel_LS_all_nn_sorted = [ Channel_LS_all_nn[i:i+ite_t] for i in range(0, len(Channel_LS_all_nn), ite_t) ] ##
    [Channel_LS_all_nnA.extend(Channel_LS_all_nn_sorted[pp]) for pp in range(0,ite_t)]
    
    Channel_NN_all_nn_sorted = [ Channel_pred_all_nn[i:i+ite_t] for i in range(0, len(Channel_pred_all_nn), ite_t) ] ##
    [Channel_NN_all_nnA.extend(Channel_NN_all_nn_sorted[pp]) for pp in range(0,ite_t)]
    
    avgMSE_LS_all = [] 
    avgMSE_NN_all = []  
    
    aa = 0
    for eee in range(len(ccchannel)):
        hNN_predc = np.reshape(Channel_NN_all_nnA[eee+aa],(-1)) #reshapes or flattens the vector to allow being used for plotting
        hLSc = np.reshape(Channel_LS_all_nnA[eee+aa],(-1))
        hc = np.reshape(ccchannel[eee], (-1))
        MSE1_ = np.mean((hc - hNN_predc)**2) #to obtain the MSE = (mean(pow(hLS - h), 2))
        MSE2_ = np.mean((hc - hLSc)**2) #to ocompute the MSE. Same as above. Considered the LS Coeff without varying noise power
        avgMSE_LS_all.append(MSE2_)
        avgMSE_NN_all.append(MSE1_)
        aa = aa+len(ccchannel)
    
    #Determine how close the predictions are to the real-values with some toleranace
    CompareResult = np.isclose(Chanel_LS_all_N[c_idx], Chanel_pred_all[c_idx], rtol=0.2) #toleranace of +-0.2
    print (CompareResult)
    correct_pred = np.count_nonzero(CompareResult)
    total_number = CompareResult.size
    Accuracy = (correct_pred/total_number)*100 #multiplied by 100 to get the percetage value immediately
    accuracy.append(Accuracy)
    print ('Prediction Accuracy of', Accuracy,'%')
        
   #Evaluate the performance of trained model using just one Channel matrix.  
    plt.plot(noise_pw[::-1], MSE_LS_all)
    plt.plot(noise_pw[::-1], MSE_NN_all)
    plt.title('Graph of MSE with varying SNR (after training)')
    plt.ylabel('Mean Square Error')
    plt.xlabel('SNR')
    plt.legend(['LS', 'NN_Pred'], loc='upper left')
    plt.grid(b=None, which='major', axis='both')
    plt.show()
    
    #Evaluate the performance of Average MSE from the trained model.  
    plt.plot(noise_pw_[::-1], avgMSE_LS_all)
    plt.plot(noise_pw_[::-1], avgMSE_NN_all)
    plt.title('Graph of Average MSE with varying SNR (after training)')
    plt.ylabel('Avg. Mean Square Error')
    plt.xlabel('SNR')
    plt.legend(['LS', 'NN_Pred'], loc='upper left')
    plt.grid(b=None, which='major', axis='both')
    plt.show()
    
    #Visualization after training and testing #To see the performance of the 3rd chanel coefficient only
    #a good overlap means good performance.
    Chanel_LS_test = np.reshape(Chanel_LS_all_N[-1], (-1))
    Chanel_pred = np.reshape(Chanel_pred_all[-1], (-1))
    Chanel_test = np.reshape(Chanel_test, (-1))
    plt.plot(Chanel_LS_test, '*-')
    plt.plot(Chanel_pred, '.-')
    plt.plot(Chanel_test, ',-')
    plt.ylabel('amplitude')
    plt.xlabel('chanel coefficient')
    plt.title('Plot of Test Chanel Data (LS) & its Predicted Chanel')
    plt.legend(['LeastSqChanel', 'PredictedLSChanel', 'TrueChanel'], loc='upper left')
    plt.grid(b=None, which='major', axis='both')
    plt.show()
    
    
    plt.plot(n_mf.history['loss'])
    plt.plot(n_mf.history['val_loss'])
    plt.title('Graph of Training Loss & its Validation Loss  - LS')
    plt.ylabel('Loss')
    plt.xlabel('No. of Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.grid(b=None, which='major', axis='both')
    plt.show()
    
    #More to visualization: show the sequential layers layers
    plot_model(model, show_shapes=True, show_layer_names=True, to_file='Rnmodel.png')
    from IPython.display import Image
    Image(retina=True, filename='Rnmodel.png') #saves the picture inot the folder-.py collocation
    
    
    #more to visualization of the model: #To obtain the weights and biases at each layer:
    #Note: Layers apart from Layer 1 and Layer 3 are the hidden layers.
    summary = model.summary()
    TrainedWeight1 = model.layers[0].get_weights()[0]
    TrainedBias1 = model.layers[0].get_weights()[1]
    #print("trained weight of layer1 =", TrainedWeight1)
    #print("trained bias of layer1 =", TrainedBias1)
    
    TrainedWeight2a = model.layers[1].get_weights()[0]
    TrainedBias2a = model.layers[1].get_weights()[1]
    #print("trained weight of layer2 =", TrainedWeight2a)
    #print("trained bias of layer2 =", TrainedBias2)
    
    TrainedWeight2b = model.layers[2].get_weights()[0]
    TrainedBias2b = model.layers[2].get_weights()[1]
    #print("trained weight of layer2 =", TrainedWeight2a)
    #print("trained bias of layer2 =", TrainedBias2)
    
    TrainedWeight3 = model.layers[3].get_weights()[0]
    TrainedBias3 = model.layers[3].get_weights()[1]
    #print("trained weight of layer2 =", TrainedWeight3)
    #print("trained bias of layer2 =", TrainedBias3)
    
    #this will create the network topology or achitecture that i modelled. 
    #so, in case you get a graphviz exectuabel error, use this llink (https://www.youtube.com/watch?v=q7PzqbKUm_4) to fix it. Cheers.
    #from an_visualizer.visualize import an_viz;
    #an_viz(model, filename="RnwithKeras", title="Neural Network Topology for Chanel Estimation")

#Note: If at any point the MSE curve descreases, and then increases again, this could indicate overfitting? So, in this case, i reduce my epoch value
#or try to tweak other hyperparameters, number of nodes.
accuracy_MMSE = []
def ChanelEstimation_MMSE(ant_array, pilot, noOfNodes):
    epoch_MMSE = 1850  #since there is a possibility of quicker convergence for MMSE than for LS, I decided to use a different epoch value here.

    Chanel_MMSE_all_N = []  #stores the output of the MMSE Chanel with varying noise power #I didnt use this again.   
    Chanel_pred_MMSE_all = [] #output of the predicted MMSE Estimation
    MSE_n_all_mmse = [] #mean sqaure erorr of mmse estimation after training
    MSE_MMSE_all = [] #mean square error of mmse estmation before training
    
    
    #Building the network: Setting up layers, activation functions, optimizers, and other metrics.
    model = Sequential()
    model.add(Dense(layer1node, init = 'random_uniform',activation='relu', input_shape =(dim,)))#first layer #I used dense layering for now here
    model.add(Dense(layer2anode , init = 'uniform', activation='relu'))# Hidden layer
    model.add(Dense(layer2bnode, init = 'random_uniform', activation='relu'))#Hidden layer, 
    model.add(Dense(layer3node, init = 'uniform', activation='linear',  input_shape = (dim,)))  #Output layer,
    model.compile(optimizer = 'adam', loss = 'mse')
        
    #train the model now:   
    n_mf_MMSE = model.fit(Chanel_MMSE_t, Chanel_t, validation_data = (Chanel_MMSE_v, Chanel_v), epochs=epoch_MMSE, batch_size =  batch_size, verbose= 1)
    
    

    for k in range(len(Noise)):
        Chanel_MMSE_N = np.dot((np.dot(y_all_N[k], (np.transpose(x)))),np.linalg.pinv(np.add(np.dot(x,(np.transpose(x))), (5*np.identity(nr)))))
        Chanel_MMSE_all_N.append(np.reshape(Chanel_MMSE_N, (1,dim)))
        
        #predict the trained model from the MMSE data
        Chanel_pred_MMSE = model.predict(Chanel_MMSE_all_N[k], batch_size = idx)
        Chanel_pred_MMSE_all.append(Chanel_pred_MMSE)
    
    #evaluate the mean square error:
    for mse_ in range(idx-1):
            hn_pred_MMSE = np.reshape(Chanel_pred_MMSE_all[mse_], (-1)) #reshapes or flattens the vector to allow being used for plotting
            hMMSE = np.reshape(Chanel_MMSE_all_N[mse_],(-1))
            MSE1 = np.mean((h - hn_pred_MMSE)**2) #to obtain the MSE = (mean(pow(hLS - h), 2))
            MSE2 = np.mean((h - hMMSE)**2) #to ocompute the MSE. Same as above. Considered the LS Coeff without varying noise power
            MSE_n_all_mmse.append(MSE1)
            MSE_MMSE_all.append(MSE2)
    
    c_idx = idx-2 #choose chanel index to view
    print("TrueChanel=%s, MMSqEChanel=%s, PredictedMMSEChanel=%s" % (np.reshape(Chanel_test,(nr,nt)), Chanel_MMSE_all_N[c_idx], Chanel_pred_MMSE_all[c_idx]))

    Channel_MMSE_all_nn = []
    Channel_pred_all__nn = []
        
    #next compute the ls for the output
    for lll in range (len(y_all_nnn)):
        Channel_MMSE_nn = np.dot((np.dot(y_all_nnn[lll], (np.transpose(x)))),np.linalg.pinv(np.add(np.dot(x,(np.transpose(x))), (2*np.identity(nr)))))  
        Channel_MMSE_all_nn.append(np.reshape(Channel_MMSE_nn, (1,dim)))
            #predict the trained model
        Channel_pred__nn = model.predict(Channel_MMSE_all_nn[lll], batch_size = idx)
        Channel_pred_all__nn.append(Channel_pred__nn)
    
    
    
    #I basically, had to loop over all the channels with varying SNR individually.
    Channel_MMSE_all_nnA = []
    Channel_pred_all_nnA = []
    
    Channel_MMSE_all_nn_sorted = [ Channel_MMSE_all_nn[i:i+ite_t] for i in range(0, len(Channel_MMSE_all_nn), ite_t) ] ##
    [Channel_MMSE_all_nnA.extend(Channel_MMSE_all_nn_sorted[pp]) for pp in range(0,ite_t)]
    
    Channel_NN_all_nn_sorted = [ Channel_pred_all__nn[i:i+ite_t] for i in range(0, len(Channel_pred_all__nn), ite_t) ] ##
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
    CompareResult_MMSE = np.isclose(Chanel_MMSE_all_N[c_idx], Chanel_pred_MMSE_all[c_idx], rtol=0.2) #toleranace of +-0.2
    print (CompareResult_MMSE)
    correct_pred_mmse = np.count_nonzero(CompareResult_MMSE)
    total_number_mmse = CompareResult_MMSE.size
    Accuracy_MMSE = (correct_pred_mmse/total_number_mmse)*100
    print ('Prediction Accuracy of', Accuracy_MMSE,'%')
    accuracy_MMSE.append(Accuracy_MMSE)
      
    
    #more to visualization of the model: #To obtain the weights and biases at each layer:
    #Note: Layers apart from Layer 1 and Layer 3 are the hidden layers.
    summary = model.summary()
    TrainedWeight1_MMSE = model.layers[0].get_weights()[0]
    TrainedBias1_MMSE = model.layers[0].get_weights()[1]
    #print("trained weight of layer1 =", TrainedWeight1_MMSE)
    #print("trained bias of layer1 =", TrainedBias1_MMSE)
    
    TrainedWeight2a_MMSE = model.layers[1].get_weights()[0]
    TrainedBias2a_MMSE = model.layers[1].get_weights()[1]
    #print("trained weight of layer2 =", TrainedWeight2a_MMSE)
    #print("trained bias of layer2 =", TrainedBias2a_MMSE)
    
    TrainedWeight2b_MMSE = model.layers[2].get_weights()[0]
    TrainedBias2b_MMSE = model.layers[2].get_weights()[1]
    #print("trained weight of layer2 =", TrainedWeight2b_MMSE)
    #print("trained bias of layer2 =", TrainedBias2b_MMSE)
    
    TrainedWeight3_MMSE = model.layers[3].get_weights()[0]
    TrainedBias3MMSE = model.layers[3].get_weights()[1]
    #print("trained weight of layer2 =", TrainedWeight3_MMSE)
    #print("trained bias of layer2 =", TrainedBias3_MMSE)
    
    #Evaluate the performance of trained model. # Display the individal plot for the MMSE performance
    plt.plot(noise_pw[::-1], MSE_MMSE_all)
    plt.plot(noise_pw[::-1], MSE_n_all_mmse)
    plt.title('Graph of MSE with varying SNR (after training)')
    plt.ylabel('Mean Square Error')
    plt.xlabel('Signal to Noise Ratio')
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
    
    
    plt.plot(n_mf_MMSE.history['loss'])
    plt.plot(n_mf_MMSE.history['val_loss'])
    plt.title('Graph of Training Loss & its Validation Loss  - MMSE')
    plt.ylabel('Loss')
    plt.xlabel('No. of Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.grid(b=None, which='major', axis='both')
    plt.show()
    
    
    #Visualization after training and testing #To see the performance of the 3rd chanel coefficient only
    #a good overlap means good performance.
    Chanel_MMSE_test = np.reshape(Chanel_MMSE_all_N[-1], (-1))
    Chanel_pred_MMSE = np.reshape(Chanel_pred_MMSE_all[-1], (-1))
    plt.plot(Chanel_MMSE_test, '*-')
    plt.plot(Chanel_pred_MMSE, '.-')
    plt.plot(np.reshape(Chanel_test, (-1)), ',-')
    plt.ylabel('amplitude')
    plt.xlabel('chanel coefficient')
    plt.title('Plot of Test Chanel Data (MMSE) & its Predicted Chanel')
    plt.legend(['MMSEChanel', 'PredictedMMSEChanel', 'TrueChanel'], loc='upper left')
    plt.grid(b=None, which='major', axis='both')
    plt.show()

# Here, I effected the idea of evaluating several antena array numbers.
ant_array = [4, 6] #so i chose this values at random. Any number or value can work too.
pilot = [6, 9] # here also, i chose 150% of corresponding antena value.
noOfNodes = [25, 40] #i realized that i get better estimation using varying number of nodes for diff. antena array sizes

j = 0
if (ant_array[j] <  pilot[j]):
    print ("parameters are appropriate") #just for control measures. nothng really serious.
else:
    print("Error! ant_array must be at least less than pilot!") #the code halts if the vlaues do not conform to what is expected.
    sys.exit() 
j = j + 1 #I am iterating over all the elements in the lists.

for i in range (len(pilot)): #length of list - pilot and that of antena array are same.
        print ("*** Evaluates LS chanel estimation performance for", ant_array[i], "by", ant_array[i], "antena array ***") # I made in such a way that it print the antena array size under evaluation.
        ChanelEstimation(ant_array[i], pilot[i], noOfNodes[i]) # this is the main stuff that calls the function/
        print ("*** Evaluates MMSE chanel estimation performance for", ant_array[i], "by", ant_array[i], "antena array ***") # I made in such a way that it print the antena array size under evaluation.
        ChanelEstimation_MMSE(ant_array[i], pilot[i], noOfNodes[i]) # this is the main stuff that calls the function/

#the idea behind this visualization is this: (not so important). This helped me to know that a constant node size does not work for all antena array sizes.
#there is a possibility that the more the antena size, the less suitable is a general neural network topology, number of nodes, epoch value, etc could be for the antena array size.
#this is just a way of seeing if the prediction preformance improved or decline per antena array size.
x_axis = np.arange(len(ant_array))
y_axis = accuracy
plt.bar(x_axis, y_axis, align='center', alpha=0.5)
plt.xticks(x_axis, ant_array)
plt.ylabel('Percentage prediction')
plt.xlabel('Antena Array size')
plt.title('Visualization of LS prediction accuracy for different antena array sizes')
plt.show()

#this is just a way of seeing if the prediction preformance improved or decline per antena array size.  
x_axis_mmse = np.arange(len(ant_array))
y_axis_mmse = accuracy_MMSE
plt.bar(x_axis_mmse, y_axis_mmse, align='center', alpha=0.5)
plt.xticks(x_axis_mmse, ant_array)
plt.ylabel('Percentage prediction')
plt.xlabel('Antena Array size')
plt.title('Visualization of MMSE prediction accuracy for different antena array sizes')
plt.show()