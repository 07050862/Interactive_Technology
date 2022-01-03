fp = open("test.txt", "r")
lines = fp.readlines()

row_sum_list = []
col_sum_list = []
for i in range(len(lines)):
    mysum = 0
    for j in range(len(lines[i])):
        if str(lines[i][j]).isdigit():
            mysum = mysum + int(lines[i][j])
    row_sum_list.append(mysum)
print("row_sum_list=%s"%row_sum_list)

for i in range(len(lines[i])):
    mysum = 0
    for j in range(len(lines)):
        if lines[j][i] == ' ': #space
            mysum = -1
            break
        elif str(lines[j][i]).isdigit():
            mysum = mysum + int(lines[j][i])
    col_sum_list.append(mysum)
#print("col_sum_list=%s"%col_sum_list)
col_sum_list2 = [] # remove -1(space)
for n in range(len(col_sum_list)):
    if col_sum_list[n]!=-1:
        col_sum_list2.append(col_sum_list[n])
print("col_sum_list=%s" % col_sum_list2)


for i in range(len(row_sum_list)):
    print("row{} = {}".format(i+1, row_sum_list[i]))
for i in range(len(col_sum_list2)):
    print("column{} = {}".format(i+1, col_sum_list2[i]))
