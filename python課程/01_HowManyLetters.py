Sentence = 'I have a pen , I have an apple'
Sentence.lower()
print(Sentence)
listS = list(Sentence)
wordlist = Sentence.split()

print(listS)
print(wordlist)

letters = []
for let in listS:
    if let not in letters:
        letters.append(let)
print("有{}種不同字母".format(len(letters)))

wordS = []
for word in wordlist:
    if word not in wordS:
        wordS.append(word)
print("有{}種不同字".format(len(wordS)))


#算Histogram, 每個字母有幾個
for let in letters:
	print("{}有{}個".format(let, listS.count(let)))


#算Histogram, 每個字有幾個
for word in wordS:
    print("{}有{}個".format(word, wordlist.count(word)))


# Histogram用字典表示
dicS1 = {}
for let in letters:
    dicS1[let] = listS.count(let)
print(dicS1)

# Histogram用字典表示
dicS2 = {}
for word in wordlist:
    dicS2[word] = wordlist.count(word)
print(dicS2)


import  operator

sorted_dicS1 = sorted(dicS1.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_dicS1)


sorted_dicS2 = sorted(dicS2.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_dicS2)


dicS1 = {}
for let in listS:
	keys=dicS1.keys()
	if let not in keys:
		dicS1[let]=1
	else:
		dicS1[let]+=1
print(dicS1)
sorted_dicS1 = sorted(dicS1.items(), key=operator.itemgetter(1), reverse=True)
for item in sorted_dicS1:
	print("{}有{}個".format(item[0], item[1]))

dicS2 = {}
for word in wordlist:
	keys=dicS2.keys()
	if word not in keys:
		dicS2[word]=1
	else:
		dicS2[word]+=1
print(dicS2)
sorted_dicS2 = sorted(dicS2.items(), key=operator.itemgetter(1), reverse=True)
for item in sorted_dicS2:
	print("{}有{}個".format(item[0], item[1]))