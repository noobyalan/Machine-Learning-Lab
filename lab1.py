import numpy as np
# 1.use flip method to reverse the elements 
a = np.array([1,2,3,4])
a_reverse = np.flip(a)
# 2.create a 10*10 random matrix , find out the max value of each columns
b = np.random.rand(10,10)
b_max = np.max(b,axis = 0)
print(b_max)
# 3.create a 10*10 random matrix , set all the elements that are greater than 0.5 to 1 ,and all the elements less than 0.5 to 0 to 0
c = np.random.rand(10,10)
c[(c>0)&(c<0.5) ] = 0
c[(c>0.5)&(c<1)] = 1
print (c) 
# 4.create a 10*10 random matrix ,and calculate the mean and standard variations of all columns 
d = np.random.rand(10,10)
d_mean = np.mean(d,axis = 0)
d_std = np.std(d,axis = 0)
print("The means are :",d_mean)
print("The standard variations are :",d_std)
# 5.create a 5*5*3 tensor and a 5*5 random matrix ,then multiply the 3 matrixes of the tensor with the 5*5 matrix in element wise way
e_tensor = np.random.rand(5,5,3)
e_matrix = np.random.rand(5,5)
index = [0,1,2]
results = np.multiply(e_tensor[:,:,index],np.repeat(e_matrix[:,:,np.newaxis],len(index),axis = 2))
print(results) 