def pythagoras(x=3, y=4):
    return pow(x**2+y**2, 0.5)


for i in range(10):
    print(pythagoras(3, i))

a = [1, 2, 3, 4, 5, 6, 7]
print(a[1:5:2])


import keyword
print(keyword.kwlist)
