import socket, sys
import json
import time
import threading

BUFSIZE = 8198  # size of receiving buffer
PKTSIZE = 8192  # number of bytes in a packet
WINDOW_SIZE = 128
IDX_LENGTH = 2 # 2 bytes of packet index
TIMEOUT = 0.1   # timeout time


def packPacket(type, fileID, seqNum, dataLen, payload=b''):
    header = bytes([type, fileID]) + seqNum.to_bytes(2, 'big') + dataLen.to_bytes(2, 'big')
    return header + payload
    
def unpackPacket(rawData):
    type = rawData[0]
    fileID = rawData[1]
    seqNum = int.from_bytes(rawData[2:4], 'big')
    dataLen = int.from_bytes(rawData[4:6], 'big')
    payload = rawData[6:]
    return type, fileID, seqNum, dataLen, payload

class Server():
    def __init__(self, config_file):
        #Read the config file and initialize the port, peer_num, peer_info, content_info from the config file
        with open(config_file, 'r') as file:
            configFile = json.load(file)
            self.port = configFile['port']
            self.peer_num = configFile['peers']
            self.content_info = configFile['content_info']
            self.peer_info = configFile['peer_info']

            

        # establish a socket according to the information
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) #NOTE THAT THE SOCK_DGRAM will ensure your socket is UDP
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("", self.port)) #This is the only port you can use to receive
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)
        
        self.server_socket.settimeout(1)   # timeout value

        self.remain_threads = True

        self.recv_sessions = {}
        self.send_sessions = {}
        self.sessions_lock = threading.Lock()
        self.next_file_id = 0

        self.cli()
        return
    
    def find_file(self, file_name):
        #A function to find the peer with the file you want!

        for peer in self.peer_info:
            if file_name in peer['content_info']:
                return peer['hostname'], peer['port']
        return

    
    def load_file(self, file_name):
        # find which server has the file
        address = self.find_file(file_name)
        # establish a client socket for downloading file
        #self.cl_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
        
        # use a connect flag to determine if the file name is sent correctly
        connected = False
        #Initiate three-way handshake and use a connect flag
        with self.sessions_lock:
            file_id = self.next_file_id
            self.next_file_id = (self.next_file_id+1) % 256
            self.recv_sessions[file_id] = []

        encoded = file_name.encode()
        synPkt = packPacket(1,file_id, 0, len(encoded), encoded)
        self.server_socket.sendto(synPkt, address)
        lastSent = time.time()
        while not connected:
            try:
                # handshake
                if time.time() - lastSent > TIMEOUT:
                    self.server_socket.sendto(synPkt, address)
                    lastSent = time.time()
                time.sleep(.01)
                with self.sessions_lock:
                    if len(self.recv_sessions[file_id]) > 0:
                        packetType, seqNum, dataLen, payload = self.recv_sessions[file_id].pop(0)
                        if packetType == 2: #SYN
                            packet_count = int.from_bytes(payload[:4], 'big')
                            connected = True
            except socket.timeout:
                # handshake failed
                connected = False


        ackPacket = packPacket(4, file_id, 0,0 )
        self.server_socket.sendto(ackPacket, address)
        # the receiver keeps a record for which part has been acked
        chunks = {}
        # start receiving file
        recieved = 0
        while recieved < packet_count:
            time.sleep(0.01)
            with self.sessions_lock:
                packets = list(self.recv_sessions[file_id])
                self.recv_sessions[file_id] = []
            lafinRecieved = False
            for packetType, seqNum, dataLen, payload in packets:
                if packetType ==3:
                    if seqNum not in chunks:
                        chunks[seqNum] = payload[:dataLen]
                        recieved = recieved + 1
                    ack = packPacket(4,file_id, seqNum, 0)
                    self.server_socket.sendto(ack,address)
                elif packetType == 5:
                    lafinRecieved = True
            if lafinRecieved:
                break
        # transmission complete, close socket

        # write the file
        with open(file_name, 'wb') as file:
            for i in range(packet_count):
                file.write(chunks[i])
        with self.sessions_lock:
            del self.recv_sessions[file_id]




    def read_file(self, file_name):
        #You can write a function that takes the file to be transmitted and converts into chunks of packet_size
        with open(file_name, 'rb') as file:
            data = file.read()
            chunks = []
            for i in range(0, len(data), PKTSIZE):
                chunks.append(data[i:i+PKTSIZE])
            return chunks



    def transmit(self, file_name, addr, file_id):
        # create a udp socket for transmission

        # divide the file into several parts
        chunks = self.read_file(file_name)
        packet_num = len(chunks)
        # use socket to send packet number to the receiver
        #ack = 0
        #print("sending packet num", packet_num, "to", addr)
        with self.sessions_lock:
            self.send_sessions[(addr, file_id)] = []
        payload = packet_num.to_bytes(4,'big')
        synAck = packPacket(2,file_id,0,4,payload)
        self.server_socket.sendto(synAck, addr)


        acknowledged = False
        while not acknowledged:
            time.sleep(0.067)
            with self.sessions_lock:
                if len(self.send_sessions[(addr,file_id)]) > 0:
                    packetType, seqNum, dataLen, payload = self.send_sessions[(addr, file_id)].pop(0)
                    if packetType == 4:
                        #Adknowledge
                        acknowledged = True

        # use a transmit window to determine which file should be transmitted

        # use a time-out array to record which file is time-out and need to be transmitted again
        # -1 indicates received, 0 indicates not transmitted, positive numbers means the time of transmission
        windowStart = [0]
        status = [0] * packet_num
        window_lock = threading.Lock()
        def transmit_thread():
            #Takes the transmit window and transmits every packet that is allowed to be transmitted
            while windowStart[0] < packet_num:
                with window_lock:
                    for i in range(windowStart[0], min(windowStart[0] + WINDOW_SIZE , packet_num)):
                        if status[i] == 0:
                            packet = packPacket(3,file_id,i,len(chunks[i]), chunks[i])
                            self.server_socket.sendto(packet, addr)
                            status[i] = time.time()
                        elif (status[i] > 0 and time.time() - status[i] > TIMEOUT):
                            packet = packPacket(3,file_id,i,len(chunks[i]), chunks[i])
                            self.server_socket.sendto(packet, addr)
                            status[i] = time.time()                  
                time.sleep(0.067)
            return
        
        def ack_thread():
            #Receives acknowledgement and updates the transmit window with sendable packets
            while windowStart[0] < packet_num:
                time.sleep(0.067)
                with self.sessions_lock:
                    packets = list(self.send_sessions[(addr, file_id)])
                    self.send_sessions[(addr, file_id)] =[]
                for packetType, seqNum, dataLen, payload in packets:
                    if packetType == 4:
                        with window_lock:
                            status[seqNum] = -1
                            while (windowStart[0] < packet_num and status[windowStart[0]] == -1):
                                windowStart[0] += 1
        #Create TX and RX threads and start doing it
        tt=threading.Thread(target=transmit_thread)
        at = threading.Thread(target = ack_thread)
        at.start()
        tt.start()
        tt.join()
        at.join()
        laFin = packPacket(5,file_id,0,0)
        self.server_socket.sendto(laFin,addr)
        with self.sessions_lock:
            del self.send_sessions[(addr, file_id)]
        #When done transmitting, close the threads.

    def listener(self): # listen to the socket to see if there's any transmission request
        #Do any initializations that you want


        while self.remain_threads:
            raw = b""
            try:
                raw, addr = self.server_socket.recvfrom(BUFSIZE)
                #Receive the file name and requesting address from the UDP
            except socket.timeout:
                pass
            
            if raw == b"":
                pass
            else:   # start transmission
                #Create a transmit thread (HINT : you can have a large array of transmit threads if you want) and start it
                type, fileID, seqNum, datalen, payload = unpackPacket(raw)
                if type == 1:
                    t = threading.Thread(target=self.transmit, args=(payload.decode(), addr, fileID))
                    t.start()
                else:
                    with self.sessions_lock:
                        if fileID in self.recv_sessions:
                            self.recv_sessions[fileID].append((type, seqNum, datalen, payload))
                        if (addr, fileID) in self.send_sessions:
                            self.send_sessions[(addr, fileID)].append((type, seqNum, datalen, payload))
        return
    
    def cli(self):  # cli interface for input of the file name
        listen_thread = threading.Thread(target=self.listener)
        listen_thread.start()

        while self.remain_threads:
            command_line = input()
            if command_line == "kill":  # for debugging purpose
                #Do the kill stuff
                self.remain_threads = False
                return
            #Otherwise it is a file name!
            else:
                t = threading.Thread(target=self.load_file, args = (command_line,))
                t.start()
        #Exit stuff if you have some?
        return


if __name__ == "__main__":
    server = Server(sys.argv[1])