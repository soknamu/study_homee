import numpy as np
a = np.array([[1,2,3], 
              [4,5,6],
              [7,8,9],
              [8,1,9]])

print(a.shape)
b = np.delete(a, [0, 1], axis=1)
print(b)
c = a[~np.any(a==1, axis=1)]
print(c)
print(c.shape)





# x = np.array([[[1,2,3], 
#               [4,5,6],
#               [7,8,9],
#               [8,1,9]]])

# print(x.shape)
# y = np.delete(x, (0, 1), axis=1)
# print(y)
# print(y.shape)
# z = x[~np.any(x==1, axis=-1)]
# print(z)
# print(z.shape)


# a1 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
# a2 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13 , 14, 15], [16, 17, 18, 19,20]])
a3 = np.array([[[1, 2, 3, 4, 5],[1,3,5,7,9]], [[6, 7, 8, 9, 10],[5,4,3,2,1]],
               [[11, 12, 13, 14, 15],[5,6,7,8,9]], [[16, 17, 18, 19, 20],[0,1,2,3,4]]])
# a4 = np.array([[[[1]], [[2]], [[3]], [[4]], [[5]], [[6]]]])
# a5 = np.array([1, 2, 3, 4, 5, 6])
# a6 = np.array([[1, 2, 3], [4, 5, 6]])
# a7 = np.array([[[1, 2, 3]], [[4, 5, 6]]])
# a8 = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
# a9 = np.array([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]])
# a10 = np.array([[[[1, 2, 3, 4]]], [[[5, 6, 7, 8]]], [[[9, 10, 11, 12]]]])

# print(a1.shape)
# print(a2.shape)
print(a3.shape)
# print(a4.shape)
# print(a5.shape)
# print(a6.shape)
# print(a7.shape)
# print(a8.shape)
# print(a9.shape)
# print(a10.shape)

# b = np.delete(a3, 1, axis=-2)
# print(b)
# print(b.shape)

# c = np.delete(a3, 1, axis=-1)
# print(c)
# print(c.shape)

# d = a3[~np.any(a3==(1,6), axis=2)]
# print(d)
# print(d.shape)