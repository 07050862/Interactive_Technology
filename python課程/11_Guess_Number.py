import random

game_count = 0
answer = random.randint(0,99)
minNum=0
maxNum=99
print(answer)
while True:
	promptStr= "請猜一個數字("+str(minNum)+"-"+str(maxNum)+")："
	guess = int(input(promptStr))
	game_count+=1
	if guess == answer:
		print("恭禧你，猜中了")
		break;
	if guess > answer:
		print("猜的數字太大了")
		maxNum=guess-1
	if guess < answer:
		print("猜的數字太小了")
		minNum=guess+1

outstr= "你猜了"+str(game_count)+"次"
print(outstr)
print("你猜了"+str(game_count)+"次")

