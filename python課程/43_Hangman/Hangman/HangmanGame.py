import string
import random

WORDLIST_FILENAME = "words.txt"

def loadWords():
    print("Loading word list from file...")
    # inFile: file
    inFile = open(WORDLIST_FILENAME, 'r')
    # line: string
    line = inFile.readline()
    # wordlist: list of strings
    wordlist = line.split()
    print("  ", len(wordlist), "words loaded.")
    return wordlist

def chooseWord(wordlist):
    return random.choice(wordlist)


def getAvailableLetters(lettersGuessed):
	allAvailableLetters=string.ascii_lowercase
	availableLetters=''
	for letter in allAvailableLetters:
		if letter not in lettersGuessed:
			availableLetters+=letter
	return availableLetters
	
def getGuessedWord(secretWord, lettersGuessed):
	printOutGuessedWord=''
	for letter in secretWord:
		if letter in lettersGuessed:
			printOutGuessedWord+=letter
		else:
			printOutGuessedWord+='_ '
	return printOutGuessedWord
	
def isWordGuessed(secretWord, lettersGuessed):
	bingo=True
	for token in secretWord:
		if token not in lettersGuessed:
			bingo = False
			break
	return(bingo)



secretWord = '' 
lettersGuessed = []
passed=False
wordlist = loadWords()
secretWord = chooseWord(wordlist).lower()
print(secretWord)

while not passed:
	letterGuessed=input("??? ")
	lettersGuessed.append(letterGuessed)
	passed=isWordGuessed(secretWord, lettersGuessed)
	if not passed:
		print(getGuessedWord(secretWord, lettersGuessed))
		print(getAvailableLetters(lettersGuessed))
		print("----------")
		
print("----------")
print("Good Job")


