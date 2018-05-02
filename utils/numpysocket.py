from __future__ import division, print_function, absolute_import

import pickle
import socket

import numpy as np


class numpysocket():
    @staticmethod
    def startServer(do_job):
        port = 3000
        server_socket = socket.socket()
        server_socket.bind(('', port))
        server_socket.listen(1)
        print('waiting for a connection...')
        while 1:
            client_connection, client_address = server_socket.accept()
            print('connected to ', client_address[0])
            ultimate_buffer = b''
            while True:
                receiving_buffer = client_connection.recv(1024)
                if not receiving_buffer: break
                ultimate_buffer += receiving_buffer
                if ultimate_buffer.decode("utf-8")[-4:] == "done":
                    break
            final_image = np.array([np.array(pickle.loads(ultimate_buffer, encoding='latin1'))])
            res = do_job(final_image)
            client_connection.sendall(pickle.dumps(res, protocol=2))
            client_connection.close()
        server_socket.close()

    def __init__(self):
        pass

    @staticmethod
    def startClient(server_address, image):
        if not isinstance(image, np.ndarray):
            print('not a valid numpy image')
            return
        client_socket = socket.socket()
        port = 3000
        try:
            client_socket.connect((server_address, port))
            print('Connected to %s on port %s' % (server_address, port))
        except socket.error as e:
            print('Connection to %s on port %s failed: %s' % (server_address, port, e))
            return
        client_socket.sendall(pickle.dumps(image))
        client_socket.sendall(b'done')
        print(len(pickle.dumps(image)), "bytes")
        ultimate_buffer = b''
        while True:
            receiving_buffer = client_socket.recv(1024)
            if not receiving_buffer: break
            ultimate_buffer += receiving_buffer

        print(pickle.loads(ultimate_buffer))
        client_socket.shutdown(1)
        client_socket.close()
