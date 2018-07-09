
# coding: utf-8

# In[ ]:


#Preditic cat:1 vs non cat:0: using 1 layer neural network from scratch


# In[192]:


#importing numpy for mathematical calulation
import numpy as np
#importing matplotlib for visiuliztion
import matplotlib.pyplot as plt
#importing h5py for reading data
import h5py
get_ipython().magic(u'matplotlib inline')


# In[179]:


#loading TRAIN dataset
train_dataset = h5py.File('train_catvnoncat.h5', "r")


# In[180]:


#Dividing data into, x and y
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))


# In[181]:


#loading Test dataset
test_dataset = h5py.File('test_catvnoncat.h5', "r")


# In[182]:


#Dividing data into, x and y
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
classes = np.array(test_dataset["list_classes"][:]) # the list of classes


# In[183]:


train_set_x_orig.shape #209 training examples, with 64x64 images


# In[184]:


test_set_x_orig.shape  #50 test examples, 


# In[185]:


# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])


# In[186]:


m_train = train_set_x_orig.shape[0] #number of training examples
m_test = test_set_x_orig.shape[0] #number of test examples
px = train_set_x_orig.shape[1] #height of image=widhth


# In[187]:


#Reshaping training and test examples: Unrolling pixels: so each image  is represented by a column


# In[188]:


train = train_set_x_orig.reshape(px*px*3,m_train)
test = test_set_x_orig.reshape(px*px*3,m_test)


# In[189]:


train.shape


# In[190]:


test.shape


# In[191]:


#RGB value is specified for each pixel, so its range from 0 to 255
#so, for standardize our dataset, we divide each row of dataset by 255


# In[169]:


train_set = train/255
test_set =  test/255


# In[170]:


#Sigmoid activation function 1/(1+e^-z)
def sigmoid(z):
    return 1/(np.exp(-z))


# In[171]:


#Initializing Parameters w and b for forward propagation
def initialize(dim):
    w = np.zeros((dim,1))
    b =0
    return w , b
    


# In[172]:


#Forward Propagation: Calculating A and Cost then
#Backward Propagation: Calculating dw , db

def fpropagate(w,b,X,Y):
    #forward Propagation
    m=X.shape[1] #number of training examples=209
    A = sigmoid(np.dot(w.T,X)+b)
    cost = (np.sum(-Y*np.log(A)-(1-Y)*np.log(1-A)))/m
    #backward Propagation
    dw = np.dot(X,(A-Y).T)/m
    db = (np.sum(A-Y))/m
    return dw , db , cost


# In[173]:


#Updating/optimizing the parameters using gradient descent
def update(w,b,X,Y,num_iterations, learning_rate):
    costs =[] #for storing cost after every 100 iteration 
    for i in range(num_iterations):
        dw,db,cost= fpropagate(w,b,X,Y)
        w = w - learning_rate*dw
        b = b - learning_rate*db
        if i % 100==0:
            costs.append(cost) #appending cost value after every 100 iteration, to plot graph, to check gradient descent working correctly or not
            
    return w , b , dw, db, costs    
        
    


# In[174]:


#Prediction
def predict(w,b,X):
    m = X.shape[1]
    w = w.reshape(X.shape[0],1)
    
    #prediction the probalities of cat beign present in the picture, using trained w and b .
    A = sigmoid(np.dot(w.T,X+b))
    for i in range (A.shape[1]):
        A[A>0.5] = 1
        A[A<=0.5]= 0
    return A
    


# In[175]:


#merging all the function to create our model
def model(X_train, Y_train,X_test, Y_test,num_iteration=3000,learning_rate=0.005):
    w,b = initialize(X_train.shape[0])
    w,b,dw,db,costs = update(w,b,X_train,Y_train,num_iteration,learning_rate)
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    
    #Printing Accuracy\
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {
        "costs":costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train":Y_prediction_train,
        "w":w,
        "b":b,
        "learning_rate": learning_rate,
        "num_iteration":num_iteration
    }
    return d


# In[176]:


#Running Our model on our train_set and test_set
d = model(train_set,train_set_y,test_set,test_set_y,num_iteration=2000,learning_rate=0.005)


# In[177]:


# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()


# In[ ]:


#Result:::
#Test Accuracy: 66 %
#Accuracy can be further increased by: 
#1.tunning Hyperparameters
#2.Training More image examples
#3.Using two layer or Convolutional Neural Network

