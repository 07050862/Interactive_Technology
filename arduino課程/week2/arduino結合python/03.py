import time
import sys
 
import serial
ser = serial.Serial('COM5', 9600)

while True:
	line = ser.readline()
	print(line)
	data = [int(val) for val in line.split()]
	if data[0] > 512:
		choice = '1'
	else:
		choice = '2'

	if choice == '1':
		print('傳送開燈指令')
		sss='LED_ON\n'
		ser.write(sss.encode())  # 訊息必須是位元組類型
		time.sleep(0.5)              # 暫停0.5秒，再執行底下接收回應訊息的迴圈
	elif choice == '2':
		print('傳送關燈指令')
		ser.write(b'LED_OFF\n')
		time.sleep(0.5)
	elif choice == 'e':
		ser.close()
		print('再見！')
		sys.exit()
	else:
		print('指令錯誤…')
 

quit()