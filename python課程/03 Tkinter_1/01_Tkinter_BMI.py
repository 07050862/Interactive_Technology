import tkinter as tk
import math

window = tk.Tk()
window.title('BMI')
window.geometry('800x600')
window.configure(background='yellow')

def calculate_bmi_number():
    height = float(height_entry.get())
    weight = float(weight_entry.get())
    print(height, weight)
    bmi_value = round( weight / math.pow(height, 2),  2)
    result = '你的 BMI 指數為：{} {}'.format(bmi_value, get_bmi_status_description(bmi_value))
    print(result)
    result_label.configure(text=result)

def get_bmi_status_description(bmi_value):
    if bmi_value < 18.5:
        return "\n體重過輕囉，多吃點！"
    elif bmi_value >= 18.5 and bmi_value < 24:
        return "\n體重剛剛好，繼續保持！"
    elif bmi_value >= 24 :
        return "\n體重有點過重囉，少吃多運動！"

header_label = tk.Label(window, text='BMI 計算器', font=('Arial', 30))
height_label = tk.Label(window, text='身高（m）', font=('Arial', 30))
height_entry = tk.Entry(window, font=('Arial', 30))
weight_label = tk.Label(window, text='體重（kg）', font=('Arial', 30))
weight_entry = tk.Entry(window, font=('Arial', 30))
calculate_btn = tk.Button(window, text='馬上計算', command=calculate_bmi_number, font=('Arial', 30))
result_label = tk.Label(window, font=('Arial', 20), width=30, height=3 )
# 版面配置
header_label.grid(row=0, column=0, columnspan=2)
height_label.grid(row=1, column=0)
height_entry.grid(row=1, column=1)
weight_label.grid(row=2, column=0)
weight_entry.grid(row=2, column=1)
calculate_btn.grid(row=3, column=0)
result_label.grid(row=3, column=1)


window.mainloop()