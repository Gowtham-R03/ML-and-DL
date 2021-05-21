# numpy

import numpy as np

# 1.creating numpy array

arr = np.array([[1,2,3],[4,6,7]], dtype =np.float32)
 # [1,2,3] 1st row as list, [4,6,7] 2nd row as list and here we assuming float data type
print(repr(arr))

#repr = it is used for better visualization of output array

# example 2 mixes dtype
# python is dynamic it take high level data type that is float 
a = np.array([1,2.5,4])
print(repr(a))

# 2.COPYING

# creating two one dimensional array
a1 = np.array([3,2,1])
a2 = np.array([4,5,6])
c = a1 #refering the arrays not copying
print("array a1: {}".format(repr(a1)))
c[0:3] = [5,4,8] #updating it is approach for copying
print('array a1: {}'.format(repr(a1)))

#better approach for copying
d = a2.copy()
d[2] = 9
print('array a2 is:{}'.format(repr(a2)))
# d does not affect any element in a2 array

# 3. CASTING

# changing from one dtype to another
b1 = np.array([1,2,3])
print(b1.dtype)
#changing from int to float
b1 = b1.astype(np.float32)
print(b1.dtype)

# 4.RANGED DATA

#In an array we cant add hundreads of data so we using range amnt of data
#so make it simple range data is used.

c1 = np.arange(100)  #upto six we can create an one dimesonal array
print(repr(c1))

#float
c2 = np.arange(5.)
print(repr(c2))

#slicing
c3 = np.arange(-1,10,2) #(start,stop,step)
print(repr(c3))

#advance of arange is linspace
c4 = np.linspace(-5,4,num = 4) #(start,stop, the number we want in array)
print(repr(c4))   #here 4 will include but in arange the stop will exclude

c5 = np.linspace(-6,3,num = 4, endpoint=False) #by giving false to endpoint we can exclude end/stop element
print(repr(c5))
#float to int
c6 = c5.astype(np.int32)
print(repr(c6))

# ** 5. RESHAPING DATA **
#creating a array with size 8
z = np.arange(8)
# we can reshape with different rows and coloumns but the multiplication 
#of the rows and columns should be equal or lesser than "8"
reshape_z = np.reshape(z,(4,2)) # 4 for rows and 2 for columns
print(repr(reshape_z))
print("new shape is:{}".format(reshape_z.shape)) #.shape is used for to know the shape of array

# Making 2d or 3d array back to original by flatten 

Flatten = reshape_z.flatten()
print("flattened 1d array is {}".format(Flatten))
print("flattened 1d array shape is {}".format(Flatten.shape))

# 6. Transposing

# like reshaping data we can transpose data
t = np.arange(8)
t = np.reshape(t, (4,2))
transposed = np.transpose(t)
print(repr(t.shape))
print(repr(transposed.shape))

# 7. ZEROS AND ONES

# to create dummy labels it will be used

z1 = np.zeros(4) # here 4 is shape
print(repr(z1))

z2 = np.ones((2,3), dtype = np.int32) # 2-rows and 3- columns
print(repr(z2))

# 8. MATRIX MULTIPLICATION
# one dimensional array multiplication
mat1 = np.array([1,2,3])
mat2 = np.array([-3,8,9])
multiply = np.matmul(mat1, mat2)
print(repr(multiply))

#two dimensional array 
mat3 = np.array([[2,3],[6,7],[8,9]]) #3-row 2-column
mat4 = np.array([[2,3,4],[5,-6,7]]) #2-row 3-column
multiplication = np.matmul(mat3,mat4)
print(repr(multiplication))

# Random Integer
print(np.random.randint(5)) # to print any random num from 0-upto5 

#how to create array from random.randint
x = np.random.randint(2,high = 10, size = (2,3))
print(repr(x))

#in random to get same number as all time we can use seed
np.random.seed(1)
v = np.random.randint(8)
v1 = np.random.randint(3,high = 8,size = (4,2))
print(repr(v))
print(repr(v1))

# 9. ARRAY ACESSING
array1 = np.array([10,6,78,98,4,11,33,45,79,23,66,55])
print(repr(array1[3]))

#two dimensional array
array2 = np.array([[1,2,3],[7,90,8],[9,0,7]])
print(repr(array2[0]))

# 10.SLICING

print(repr(array1[:]))
print(repr(array1[1:]))
print(repr(array1[:6]))
print(repr(array1[3:7]))

print(repr(array2[:,1:2])) #first colun for rows and after coma colun for column

# 11. ANALYSING

print("the minimum num of this array is:{}".format(repr(array2.min())))
print(repr(array2.max()))

# to analysis min and max value in row wise and colmn wise 1-row and 0-column
print(repr(array2.min(axis = 0)))
print(repr(array2.max(axis = 1)))

# mean median mode var 

print(np.median(array2))
print(np.mean(array2))
print(np.var(array2))
print(repr(np.median(array2, axis = 1)))
