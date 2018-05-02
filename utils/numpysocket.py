from __future__ import division, print_function, absolute_import

import socket
from cStringIO import StringIO

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
            ultimate_buffer = ''
            while True:
                receiving_buffer = client_connection.recv(1024)
                if not receiving_buffer: break
                ultimate_buffer += receiving_buffer
                print('-')
            final_image = np.load(StringIO(ultimate_buffer))['frame']
            res = do_job(final_image)
            client_connection.sendall(res)
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
        except socket.error, e:
            print('Connection to %s on port %s failed: %s' % (server_address, port, e))
            return
        f = StringIO()
        np.savez_compressed(f, frame=image)
        f.seek(0)
        out = f.read()
        client_socket.sendall(out)
        ultimate_buffer = ''
        while True:
            receiving_buffer = client_socket.recv(1024)
            if not receiving_buffer: break
            ultimate_buffer += receiving_buffer

        print(ultimate_buffer)
        client_socket.shutdown(1)
        client_socket.close()
