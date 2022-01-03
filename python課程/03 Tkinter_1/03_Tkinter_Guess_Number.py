import tkinter as tk
import math

window = tk.Tk()
window.title('Guess Number')
window.geometry('800x600')
window.configure(background='yellow')

import random

game_count = 0
answer = random.randint(0,99)
minNum=0
maxNum=99
print(answer)

def calculate_reslt():
    global answer
    global minNum
    global maxNum
    global game_count

    game_count+=1
    guess = int(Input_entry.get())
    print(guess)
    if guess == answer:
        print("恭禧你，猜中了")
        result ="恭禧你，猜中了"
        result +="\n你猜了"+str(game_count)+"次"

    if guess > answer:
        print("猜的數字太大了")
        result ="猜的數字太大了"
        maxNum=guess-1
        result +="\n 請猜一個數字("+str(minNum)+"-"+str(maxNum)+")"

    if guess < answer:
        print("猜的數字太小了")
        result ="猜的數字太小了"
        minNum=guess+1
        result +="\n 請猜一個數字("+str(minNum)+"-"+str(maxNum)+")"
    result_label.configure(text=result)

header_label = tk.Label(window, text='猜數字', font=('Arial', 30))
Input_label = tk.Label(window, text='輸入', font=('Arial', 30))
Input_entry = tk.Entry(window, font=('Arial', 30))

calculate_btn = tk.Button(window, text='馬上猜', 
       command=calculate_reslt, font=('Arial', 30))
result_label = tk.Label(window, font=('Arial', 30), width=19, height=3 )
# 版面配置
header_label.grid(row=0, column=0, columnspan=2)
Input_label.grid(row=1, column=0)
Input_entry.grid(row=1, column=1)
calculate_btn.grid(row=2, column=0)
result_label.grid(row=2, column=1)

window.mainloop()