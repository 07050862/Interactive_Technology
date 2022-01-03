import string

def getAvailableLetters(lettersGuessed):
	allAvailableLetters=string.ascii_lowercase
	availableLetters=''
	for letter in allAvailableLetters:
		if letter not in lettersGuessed:
			availableLetters+=letter
	return availableLetters

lettersGuessed = ['e', 'i', 'k', 'p', 'r', 's']
results=getAvailableLetters(lettersGuessed)
print(results)