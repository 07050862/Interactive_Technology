#先查看clean.txt
f = open("clean.txt", 'r')
lines = f.readlines()
print("----------這是clean.txt的資料----------")
for line in lines:
    print(line)
f.close()
#先查看result.txt
f = open("result.txt", 'r')
lines = f.readlines()
print("----------這是result.txt的資料----------")
for line in lines:
    print(line)
f.close()

print("~~~~~~~~~~寫入後~~~~~~~~~~")
f1 = open("clean.txt", 'r')
f2 = open("result.txt", 'w')
for line in f1:
    print("在這裡:%s"%line)
    for i in range(len(line)):
        if  line[i].isdigit():
            continue
        else:
            f2.write(line[i])
f1.close()
f2.close()


#先查看clean.txt
f = open("clean.txt", 'r')
lines = f.readlines()
print("----------這是clean.txt的資料----------")
for line in lines:
    print(line)
f.close()
#先查看result.txt
f = open("result.txt", 'r')
lines = f.readlines()
print("----------這是result.txt的資料----------")
for line in lines:
    print(line)
f.close()