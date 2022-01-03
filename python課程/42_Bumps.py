fp=open("bump.txt","r")
lines=fp.readlines()

d=[]
A=len(lines)
for i in range(A):
	b=list(lines[i])
	d.append(b)

p=[]
B=len(lines[16])

for i in range(A):
	k=0
	for j in range(B):
		if("*" == lines[i][j]):
			k+=1
	p.append(k)

q=[]
for i in range(B):
	k=0
	for j in range(A):
		if("*" == lines[j][i]):
			k+=1
	q.append(k)

total=0
for i in range(len(p)):
	print("第{}橫行 有{}個*".format(i,p[i]))
	total=total+p[i]
	
for i in range(len(q)):
	print("第{}直行 有{}個*".format(i,q[i]))
print("總共有{}個*".format(total))