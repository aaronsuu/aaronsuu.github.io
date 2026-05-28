import socket, sys
import ast
import threading, time
import random

BUFSIZE = 1024  # size of receiving buffer
ALIVE_SGN_INTERVAL = 0.5  # interval to send alive signal
TIMEOUT_INTERVAL = 10*ALIVE_SGN_INTERVAL
UPSTREAM_PORT_NUMBER = 1111 # socket number for UL transmission

##
#
# FOR TRANSMITTING PACKET USE THE FOLLOWING CODE
#
#ul_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#try:
#   ul_socket.connect((host, backend_port))
#   ul_socket.send(("STRING TO SEND").encode())
#   ul_socket.close()
#except socket.error:
#   pass
#
#
#
#

class Content_server():
    def __init__(self, conf_file_addr):
        # load and read configuration file
        
        with open(conf_file_addr, 'r') as line:


            firstLine = line.readline()
            subject, _ , value = firstLine.strip().partition('=')
            subject = subject.strip()
            if subject.startswith("uuid"):
                    self.uuid = value.strip()
            if subject.startswith("name"):
                    self.name = value.strip()
            if subject.startswith("backend_port"):
                    self.backendPort = int(value.strip())
            if subject.startswith("peer_count"):
                    self.peerValue = int(value.strip())

            secondLine = line.readline()
            subject, _ , value = secondLine.strip().partition('=')
            subject = subject.strip()
            if subject.startswith("uuid"):
                    self.uuid = value.strip()
            if subject.startswith("name"):
                    self.name = value.strip()
            if subject.startswith("backend_port"):
                    self.backendPort = int(value.strip())
            if subject.startswith("peer_count"):
                    self.peerValue = int(value.strip())

            thirdLine = line.readline()
            subject, _ , value = thirdLine.strip().partition('=')
            subject = subject.strip()
            if subject.startswith("uuid"):
                    self.uuid = value.strip()
            if subject.startswith("name"):
                    self.name = value.strip()
            if subject.startswith("backend_port"):
                    self.backendPort = int(value.strip())
            if subject.startswith("peer_count"):
                    self.peerValue = int(value.strip())

            fourLine = line.readline()
            subject, _ , value = fourLine.strip().partition('=')
            subject = subject.strip()
            if subject.startswith("uuid"):
                    self.uuid = value.strip()
            if subject.startswith("name"):
                    self.name = value.strip()
            if subject.startswith("backend_port"):
                    self.backendPort = int(value.strip())
            if subject.startswith("peer_count"):
                    self.peerValue = int(value.strip())
                    
            self.peers = []
            self.uuidToName = {}

           

            for i in range(int(self.peerValue)):
                name,_, data = line.readline().strip().partition("=")
                name = name.strip()
                peerdata = data.split(',')
                peerUUID = peerdata[0].strip()
                host = peerdata[1].strip()
                port = int(peerdata[2].strip())
                metric = int(peerdata[3].strip())
                self.peers.append([name, peerUUID, host, port, metric])
                self.uuidToName[peerUUID] = name

        
        # create the receive socket
        self.dl_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dl_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.dl_socket.bind(("", self.backendPort)) #YOU NEED TO READ THIS FROM CONFIGURATION FILE
        self.dl_socket.listen(100)

        # Create all the data structures to store various variables
        self.neighbors = {}
        self.map = {self.name: {}}
        self.seqNumbers = {}
        self.ownSeqNumbers = 0
        
        # Extract neighbor information and populate the initial variables

        for peerInfo in self.peers:
            self.addneighbor(peerInfo[1],peerInfo[2],peerInfo[3],peerInfo[4])
        
        # Update the map 
        own = {}
        for uuid, info in self.neighbors.items():
            i = self.uuidToName.get(uuid,uuid)
            own[i] = info['metric']
        self.map[self.name] = own
        
        # Initialize link state advertisement that repeats using a neighbor variable
        
        #self.link_state_adv()


        self.remain_threads = True
        self.alive()
        
        return
    
    def addneighbor(self, uuid, host, backend_port, metric):
        # Add neighbor code goes here
        self.neighbors[uuid] = {'host': host, 'backend': backend_port, 'metric': metric, 'lastAlive': time.time()}
        return;
    
    def link_state_adv(self):
        while self.remain_threads:
            # Perform Link State Advertisement to all your neighbors periodically 
            neighbor_ids = {}

            for uuid, infoN in list(self.neighbors.items()):
                name = self.uuidToName.get(uuid,uuid)
                neighbor_ids[name] = infoN['metric']

            parts = []
            for uuid, metric in list(neighbor_ids.items()):
                parts.append(uuid+"|"+str(metric))
            neighbor_string = ";".join(parts)

            for info in list(self.neighbors.values()):
                ul_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                #MSG Format: <UUID>:<NAME>:<SEQNUM>:<NEIGHTBOR IDS>






                MSG = "Link State Packet:" + self.uuid + ":" + self.name + ":" + str(self.ownSeqNumbers) + ":" + neighbor_string


                try:   
                   ul_socket.connect((info['host'], info['backend']))
                   ul_socket.send((MSG).encode())
                   ul_socket.close()
                except socket.error:
                   pass

            self.ownSeqNumbers = self.ownSeqNumbers + 1
            
            time.sleep(ALIVE_SGN_INTERVAL)
                
        return;




    
    def link_state_flood(self, send_time, host, msg):
        # If new information then send to all your neighbors, if old information then drop.

        trunc = msg.split(":",3)
        uuid = trunc[0]
        
        name = trunc[1]
        self.uuidToName[uuid] = name
        own = {}
        for nbr_uuid, info in list(self.neighbors.items()):
            nbr_name = self.uuidToName.get(nbr_uuid,nbr_uuid)
            own[nbr_name] = info['metric']
        self.map[self.name]=own
        seq = trunc[2]
        neighbors_str = trunc[3]


        if int(seq) > self.seqNumbers.get(uuid,-1): #If seq is higher than keep it
            
            self.seqNumbers[uuid]= int(seq) #Change the seq number, so that when I see it again it does it 

            neighbors_dict = {}
            if neighbors_str:
                for i in neighbors_str.split(";"):
                    parts = i.split("|")
                    if len(parts) == 2:
                        neighbors_dict[parts[0]] = int(parts[1])
                self.map[name] = neighbors_dict


            for info in list(self.neighbors.values()):
                    
                ul_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    ul_socket.connect((info['host'], info['backend']))
                    ul_socket.send(("Link State Packet:" + msg).encode())
                    ul_socket.close()
                except socket.error:
                    pass

        return
    
    def dead_adv(self, peer):
        # Advertise death before kill
        return
    
    def dead_flood(self, send_time, host, peer):
        # Forward the death message information to other peers
        return

    def keep_alive(self):
        # Tell that you are alive to all your neighbors, periodically.
        while self.remain_threads:
            for uuid, info in list(self.neighbors.items()):
                ul_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                msg = ("Alive:" +  self.uuid + ":" + str(time.time()) + ":" + str(self.backendPort) + ":" + str(info['metric']))
                try:
                   ul_socket.connect((info['host'], info['backend']))
                   ul_socket.send(str(msg).encode())
                   ul_socket.close()
                except socket.error:    
                   pass
            time.sleep(ALIVE_SGN_INTERVAL)

        return
    
   
   ## THIS IS THE RECEIVE FUNCTION THAT IS RECEIVING THE PACKETS


   ##SAMPLE MSG: 
   # ALIVE:[uuid]:[time_sent]
    def listen(self):
        self.dl_socket.settimeout(0.1)  # for killing the application
        while self.remain_threads:
            try:
                connection_socket, client_address = self.dl_socket.accept()
                msg_string = connection_socket.recv(BUFSIZE).decode()
                #print("received", connection_socket, client_address, msg_string)
            except socket.timeout:
                msg_string = ""
                pass

            if msg_string == "":    # empty message
                pass
            elif msg_string.startswith("Alive:"): # Update the timeout time if known node, otherwise add new neighbor
                msg_trunc = msg_string.split(":",4)
                uuid= msg_trunc[1]
                if uuid in self.neighbors:
                    self.neighbors[uuid]['lastAlive'] = float(msg_trunc[2])
                else:
                    # msg = ("Alive:" +  self.uuid + ":" + str(time.time()) + ":" + int(self.backendPort) + ":" + int(info['metric']))
                    self.addneighbor(uuid, client_address[0], int(msg_trunc[3]), int(msg_trunc[4]))

                
            elif msg_string.startswith("Link State Packet"):     # Update the map based on new information, drop if old information
                #If new information, also flood to other neighbors
                #MSG Format: LSP:<UUID>:<NAME>:<SEQNUM>:<NEIGHTBOR IDS>
                self.link_state_flood(str(time.time()), client_address[0], msg_string[len("Link State Packet:"):])
            elif msg_string == "Death message": # Delete the node if it sends the message before executing kill.
                pass
            # otherwise the msg is dropped

    def timeout_old(self):
        # drop the neighbors whose information is old

        while self.remain_threads:
            deadList = []
            for uuid, info in list(self.neighbors.items()):
                if(time.time() - float(info['lastAlive']) > TIMEOUT_INTERVAL):
                    deadList.append(uuid)
            for uuid in deadList:
                dead_name = self.uuidToName.get(uuid)
                del self.neighbors[uuid]
                if dead_name in self.map:
                    del self.map[dead_name]
                if uuid in self.seqNumbers: 
                    del self.seqNumbers[uuid]
            referenced = {self.name}
            for owner, nbrs in list(self.map.items()):
                for nbr in nbrs:
                    referenced.add(nbr)
            for name in list(self.map.keys()):
                if name not in referenced:
                    del self.map[name]
            

            time.sleep(ALIVE_SGN_INTERVAL)

        return;

    def shortest_path(self):
        # derive the shortest path according to the current link state
        rank = {}
        visited = set()
        unvisited = set(self.map.keys())

        dist = {self.name:0}

        while len(visited) < len(self.map):
            best_node = None
            best_dist = float('inf')
            for node in unvisited:
                if dist.get(node, float('inf'))< best_dist:
                    best_dist = dist[node]
                    best_node = node
            if best_node is None:
                break
            current = best_node
            visited.add(current)
            unvisited.remove(current)

            for neighbor, cost in self.map.get(current, {}).items():
                newCost = dist[current] + cost
                if newCost < dist.get(neighbor, float('inf')):
                    dist[neighbor] = newCost

        for node, cost in dist.items():
            if node != self.name:
                rank[node] = cost
            
        return rank

    
    def alive(self):
        keep_alive = threading.Thread(target=self.keep_alive) # A thread that keeps sending keep_alive messages
        listen = threading.Thread(target=self.listen) # A thread that keeps listening to incoming packets
        timeout_old = threading.Thread(target=self.timeout_old) # A thread to eliminate old neighbors
        link_state_adv = threading.Thread(target=self.link_state_adv) # A thread that keeps doing link_state_adv
        keep_alive.start()
        listen.start()
        timeout_old.start()
        link_state_adv.start()
        while self.remain_threads:
            #time.sleep(ALIVE_SGN_INTERVAL)  # wait for the network to settle
            command_line = input().split(" ")
            command = command_line[0]
            #print("Recieved Command: ", command)
            if command == "kill":
                # Send death message
                # Kill all threads
                self.remain_threads = False
            elif command == "uuid":
                print(f'{{"uuid": "{self.uuid}"}}', flush = True)
            elif command == "neighbors":
                # Print Neighbor information
                outer = []

                own = {}
                for nbr_uuid, info in list(self.neighbors.items()):
                    nbr_name = self.uuidToName.get(nbr_uuid,nbr_uuid)
                    own[nbr_name] = info['metric']
                self.map[self.name]=own

                



                for uuid, info in list(self.neighbors.items()):
                    inner = []
                    
                    name = self.uuidToName.get(uuid, uuid)
                    inner.append('"uuid": "' + uuid + '", ' + '"host": "' + info['host'] + '",   ' +'"backend_port": ' + str(info['backend']) + ', ' + '"metric": ' + str(info['metric']))
                    outer.append('"'+ name +'": {'+ ",".join(inner)+'}')

                print('{"neighbors": {' + ",".join(outer) + '}}', flush = True)

                
            elif command == ("addneighbor"):
                # Update Neighbor List with new neighbor
                args = {}
                i = 1
                while i < len(command_line):
                    token = command_line[i]
                    if '=' in token:
                        k, _, v = token.partition('=')
                        if v == '':        # space after =, value is next token
                            i += 1
                            v = command_line[i]
                        args[k.strip()] = v.strip()
                    i += 1

                uuid   = args['uuid']
                hostID = args['host']
                backPort = int(args['backend_port'])
                metric   = int(args['metric'])

                
                

                self.addneighbor(uuid, hostID, backPort, metric)
                
            elif command == "map":
                # Print Map


                outer = []
                own = {}
                for nbr_uuid, info in list(self.neighbors.items()):
                    nbr_name = self.uuidToName.get(nbr_uuid,nbr_uuid)
                    own[nbr_name] = info['metric']
                self.map[self.name]=own

                for node, neighbors in list(self.map.items()):
                    inner = []
                    for neighbor, metric in list(neighbors.items()):
                        inner.append( '"' + neighbor + '": ' + str(metric))
                    outer.append('"'+node +'": {'+ ",".join(inner)+'}')

                print('{"map": {' + ",".join(outer) + '}}', flush = True)

                
                
            elif command == "rank": 

                # {"rank": {"node2": 10, "node3": 20, "node4": 50}}

                items = []
                own = {}
                for nbr_uuid, info in list(self.neighbors.items()):
                    nbr_name = self.uuidToName.get(nbr_uuid,nbr_uuid)
                    own[nbr_name] = info['metric']
                self.map[self.name]=own
                for node, cost in self.shortest_path().items():
                    items.append('"' + node + '":' + str(cost))
                print('{"rank": {'+ ",".join(items) + '}}', flush = True)


                
                

if __name__ == "__main__":
    content_sever = Content_server(sys.argv[2])
