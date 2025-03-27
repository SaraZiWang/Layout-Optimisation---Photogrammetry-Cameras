import ImageFunctions
import skimage as ski
import socket

def PySend(sendmsg):
    global conn
    # send data to the client
    conn.send(sendmsg.encode())
    # receive data stream
    recvdata = conn.recv(1024).decode()
    # print("from PS user: " + str(recvdata))
    return recvdata


"""Establish Connection"""
# get the hostname
host = socket.gethostname()
port = 2023  # initiate port no above 1024

server_socket = socket.socket()  # get instance
server_socket.bind(('127.0.0.1', port))  # bind host address and port together
server_socket.listen(2)

# while True:
print("Waiting for connection...")
conn, address = server_socket.accept()  # accept new connection
print("Connection from: " + str(address))

recv = PySend('Connected?')
print(recv)

# AGV range
x_min, x_max = 4356., 5106.
y_min, y_max = 24641., 25541.

#Camera range
CamL_min, CamL_max = 1280., 1300.
CamR_min, CamR_max = 1200., 1220.

AGV_x=5000.; AGV_y=25209.00
Cam_L=1500.; Cam_R=1300.

command_x = 'Relocate.AGV(1,' + str(AGV_x) + ')' # AGV Moving in x-axis
recv = PySend(command_x)

command_y = 'Relocate.AGV(0,' + str(AGV_y) + ')' # AGV Moving in y axis
recv = PySend(command_y)

command_x = 'Relocate.CamL(' +str(Cam_L) + ')' # Only can move in z-axis, up and down
recv = PySend(command_x)

recv = PySend('LHS Image Set Capture')

# Import image - 'black frame with obstruction'
rgb_img = ski.io.imread('../OutputFiles/VSTARS_L_BLK.png')

# Import image - 'black frame without obstruction'
full_img = ski.io.imread('../OutputFiles/VSTARS_L_ISO.png')

vis, duration = ImageFunctions.ImgBasedVisibility(rgb_img,full_img,True)

# command_x = 'Relocate.CamR(' +str(Cam_R) + ')' # Only can move in z-axis, up and down
# recv = PySend(command_x)
#
# recv = PySend('RHS Image Set Capture')

recv = PySend('Done')
conn.close()  # close the connection


