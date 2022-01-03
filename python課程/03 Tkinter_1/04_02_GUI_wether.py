import tkinter as tk 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


#讀csv檔 將不要的資料去除
weather=pd.read_csv('taiwan_weather.csv')
weather=weather[ ['CITY','DISTRICT','DAY','TIME','T'] ]
#weather=weather[weather.TIME=='12:00:00']
print(weather.head(20))


#設計GUI
font_size=('',25)

root=tk.Tk()
root.title('weather')
root.geometry('1080x480')
root.configure(background='yellow')

tk.Label(root,text='City',font=font_size).grid(row=0,column=0)
tk.Label(root,text='District',font=font_size).grid(row=0,column=1)
tk.Label(root,text='Day',font=font_size).grid(row=0,column=2)

e1=tk.Entry(root,font=font_size)
e2=tk.Entry(root,font=font_size)
e3=tk.Entry(root,font=font_size)

e1.grid(row=1,column=0)
e2.grid(row=1,column=1)
e3.grid(row=1,column=2)

# 按下button執行查詢動作
var=tk.StringVar()
def event_get():
    try:
        mask1 = weather['CITY'] == e1.get().replace(' ','')
        mask2 = weather['DISTRICT'] == e2.get().replace(' ','')
        mask3 = weather['DAY'] == e3.get().replace(' ','')
        result01=weather[mask1 & mask2 & mask3]
        result=result01[result01.TIME=='12:00:00']
        print(result)
        temp=result.iloc[0][4] #iloc是找 第0個row 
        #temp=result.loc[0]['T'] #loc是用找 key 是 0 的row 所以用在這是錯的
        var.set(temp)
    except:    
        var.set('invalid input')
        print('invalid input')

		
def event_get_02():
    try:
        mask1 = weather['CITY'] == e1.get().replace(' ','')
        mask2 = weather['DISTRICT'] == e2.get().replace(' ','')
        mask3 = weather['DAY'] == e3.get().replace(' ','')
        result01=weather[mask1 & mask2 & mask3]
        temp=result01['T'].tolist()
        print(temp)
        ind = np.arange(len(temp))
        x_label = ['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00']
        plt.bar(ind, temp)
        plt.xticks(ind, x_label)
        plt.show()
    except:    
        var.set('invalid input')
        print('invalid input')
		
btn=tk.Button(root,text='查詢',font=font_size,command=event_get)
btn.grid(row=2,column=0)
btn_02=tk.Button(root,text='分布圖',font=font_size,command=event_get_02)
btn_02.grid(row=3,column=0)


var.set('Temperature')
tk.Label(root,textvariable=var,font=font_size).grid(row=2,column=1)
root.mainloop()