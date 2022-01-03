def getGuessedWord(secretWord, lettersGuessed):
	printOutGuessedWord=''
	for letter in secretWord:
		if letter in lettersGuessed:
			printOutGuessedWord+=letter
		else:
			printOutGuessedWord+='_ '
	return printOutGuessedWord


secretWord = 'apple' 
lettersGuessed = ['e', 'i', 'k', 'p', 'r', 's']
print(getGuessedWord(secretWord, lettersGuessed))
