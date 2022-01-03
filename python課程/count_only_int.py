a = [[1, 2, 3, '4','a'], [5, 6], ['7', 8, 9]]
sum_list = []

for i in range(len(a)):
    mysum = 0
    for j in range(len(a[i])):
        if str(a[i][j]).isdigit():
            mysum = mysum + int(a[i][j])
    sum_list.append(mysum)
print("sum_list=%s"%sum_list)

""" output: sum_list=[10, 11, 24] """