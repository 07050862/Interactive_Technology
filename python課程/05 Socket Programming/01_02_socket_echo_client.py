import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('127.0.0.1', 1000)
print('connecting to {} port {}'.format(*server_address))
sock.connect(server_address)

try:
	while True:
		# Send data
		message = input('input:')
		print('sending {!r}'.format(message))
		sock.sendall(message.encode('utf-8'))

		# Look for the response
		amount_received = 0
		amount_expected = len(message)

		while amount_received < amount_expected:
			data = sock.recv(16).decode('utf-8')
			amount_received += len(data)
			print('received {!r}'.format(data))
		if(data=='quit'):
			break

finally:
	print('closing socket')
	sock.close()