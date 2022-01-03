import tkinter as tk
import math

window = tk.Tk()
window.title('BMI')
window.geometry('800x600')
window.configure(background='yellow')

def calculate_ADD():
    numA = float(InputA_entry.get())
    numB = float(InputB_entry.get())
    print(numA, numB)
    result_value = numA+numB
    print(result_value)
    result_label_cal.configure(text=result_value)
def calculate_SUB():
    numA = float(InputA_entry.get())
    numB = float(InputB_entry.get())
    print(numA, numB)
    result_value = numA-numB
    print(result_value)
    result_label_cal.configure(text=result_value)
def calculate_MUL():
    numA = float(InputA_entry.get())
    numB = float(InputB_entry.get())
    print(numA, numB)
    result_value = numA*numB
    print(result_value)
    result_label_cal.configure(text=result_value)
def calculate_DIV():
    numA = float(InputA_entry.get())
    numB = float(InputB_entry.get())
    print(numA, numB)
    result_value = numA/numB
    print(result_value)
    result_label_cal.configure(text=result_value)


InputA_label = tk.Label(window, text='Input A', font=('Arial', 30))
InputB_label = tk.Label(window, text='Input B', font=('Arial', 30))
InputA_entry = tk.Entry(window, font=('Arial', 30))
InputB_entry = tk.Entry(window, font=('Arial', 30))


ADD_btn = tk.Button(window, text='ADD', command=calculate_ADD, font=('Arial', 30))
SUB_btn = tk.Button(window, text='SUB', command=calculate_SUB, font=('Arial', 30))
MUL_btn = tk.Button(window, text='MUL', command=calculate_MUL, font=('Arial', 30))
DIV_btn = tk.Button(window, text='DIV', command=calculate_DIV, font=('Arial', 30))
result_label = tk.Label(window, text='Result', font=('Arial', 30))
result_label_cal = tk.Label(window, font=('Arial', 20), width=30, height=3)
# 版面配置
InputA_label.grid(row=0, column=0)
InputA_entry.grid(row=0, column=1)
InputB_label.grid(row=1, column=0)
InputB_entry.grid(row=1, column=1)

ADD_btn.grid(row=2, column=0)
SUB_btn.grid(row=2, column=1)
MUL_btn.grid(row=3, column=0)
DIV_btn.grid(row=3, column=1)



result_label.grid(row=4, column=0)
result_label_cal.grid(row=4, column=1)



window.mainloop()