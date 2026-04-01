# TODO: make a similar script that moves them all to zeros

def send_and_recv(socket, message):
    socket.send_string(message)
    return socket.recv_string().strip()