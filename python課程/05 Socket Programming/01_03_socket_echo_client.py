import socket
import sys
import tkinter as tk
from tkinter import scrolledtext
import threading


# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('127.0.0.1', 1000)
print('connecting to {} port {}'.format(*server_address))
sock.connect(server_address)

root=tk.Tk()
root.title('echo server')
root.geometry('640x480')
root.configure(background='yellow')

scrolledmessage=scrolledtext.ScrolledText(root, wrap='word',width=35,height=10, font=('',20))
scrolledmessage.insert(tk.END, 'Welcome to ecko server\n')
scrolledmessage.configure(state='disabled')
scrolledmessage.grid(row=0,column=0)

text=tk.Text(root,width=35,height=10,font=('',20))
text.grid(row=2,column=0)
text.focus_set()
def event_get():
	message=text.get(1.0,tk.END)
	text.delete(1.0, tk.END)
	print('sending {!r}'.format(message.replace('\n','')))
	sock.sendall(message.encode('utf-8'))
	amount_received = 0
	amount_expected = len(message)

	while amount_received < amount_expected:
		data = sock.recv(16).decode('utf-8')
		amount_received += len(data)
		print('received {!r}'.format(data.replace('\n','')))
		scrolledmessage.configure(state='normal')
		scrolledmessage.insert(tk.END, data)
		scrolledmessage.configure(state='disabled')
		scrolledmessage.see(tk.END)
	if(data=='quit\n'):
		print('sock close')
		sock.close()
		root.destroy()
#tk.Label(root,background='yellow').grid(row=1,column=0)
btn=tk.Button(root,text='send',font=('',20),command=event_get)
btn.grid(row=2,column=1,sticky=tk.W+tk.N)

root.mainloop()
