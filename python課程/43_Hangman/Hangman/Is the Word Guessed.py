def isWordGuessed(secretWord, lettersGuessed):
	bingo=True
	for token in secretWord:
		if token not in lettersGuessed:
			bingo = False
			break
	return(bingo)



secretWord = 'apple' 
lettersGuessed = ['e', 'i', 'k', 'p', 'r', 's']

print(isWordGuessed(secretWord, lettersGuessed))
