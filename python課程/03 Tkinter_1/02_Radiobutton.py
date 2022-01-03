from tkinter import Radiobutton, Tk, Label, Entry, Spinbox, Button, messagebox, StringVar, IntVar, scrolledtext, END

def sel():
   selection = "You selected the option " + str(var.get())
   label_01.config(text = selection)

root = Tk()
# 視窗大小
root.geometry("350x200")
root.configure(background='yellow')
var = IntVar()
fontSize=20

R1 = Radiobutton(root, text="Option 1", variable=var, value=1, font=('Arial', fontSize),
                  command=sel)
R1.pack()

R2 = Radiobutton(root, text="Option 2", variable=var, value=2, font=('Arial', fontSize),
                  command=sel)
R2.pack( )

R3 = Radiobutton(root, text="Option 3", variable=var, value=3, font=('Arial', fontSize),
                  command=sel)
R3.pack()

label_01 = Label(root, font=('Arial', fontSize))
label_01.pack()
root.mainloop()