import time
import sys
 
import serial
ser = serial.Serial('COM5', 9600)

while True:
	choice = input('按1開燈、按2關燈、按3開啟喇叭、按4關喇叭、按e關閉程式  ').lower()
 
	if choice == '1':
		print('傳送開燈指令')
		sss='LED_ON\n'
		ser.write(sss.encode())  # 訊息必須是位元組類型
		time.sleep(0.5)              # 暫停0.5秒，再執行底下接收回應訊息的迴圈
	elif choice == '2':
		print('傳送關燈指令')
		ser.write(b'LED_OFF\n')
		time.sleep(0.5)
	elif choice == '3':
		print('傳送喇叭開啟指令')
		sss = 'SPEAKER_ON\n'
		ser.write(sss.encode())  # 訊息必須是位元組類型
		time.sleep(0.5)
	elif choice == '4':
		print('傳送喇叭關閉指令')
		sss = 'SPEAKER_OFF\n'
		ser.write(sss.encode())  # 訊息必須是位元組類型
		time.sleep(0.5)
	elif choice == 'e':
		ser.close()
		print('再見！')
		sys.exit()
	else:
		print('指令錯誤…')
 
	while ser.in_waiting:
		feedback = ser.readline()  # 接收回應訊息並解碼
		print('控制板回應：', feedback)
		feedback = feedback.decode()
		print('控制板回應：', feedback)

quit()