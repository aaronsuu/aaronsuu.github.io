---
title: Content Distribution Overlay (Link-State Routing)
description: An overlay network of peer nodes that discover each other and compute shortest paths using a from-scratch link-state routing protocol — keepalive heartbeats, sequence-numbered LSA flooding, and Dijkstra over a graph of up to 32 nodes.
keywords: [Python, Networking, TCP, Sockets, Routing, Distributed Systems, Threading]
minorTags: ["EE 419", "Computer Networking", "Link-State Routing", "Overlay Network", "Peer-to-Peer", "Dijkstra", "LSA Flooding", "Keepalive", "Sequence Numbers", "UUID", "Graph Discovery"]
order: 1
lastUpdated: 2026-05-27
---

Content on the Internet doesn't live on one machine — an episode on Netflix is replicated across hundreds of servers, and your client has to figure out which one to talk to. This project builds the building block for that: an **overlay network of replicated content servers** that discover each other in real time using a **link-state routing protocol**, then compute the cheapest route to every other server in the network.

Each node (`content_server.py`) reads a `.conf` file that lists its UUID, name, backend port, and the cost to its direct neighbors. From that starting point, every node ends up with a complete graph of the network and a ranked routing table — and recovers automatically when peers die or new ones are added on the fly. The grader runs up to **32 simultaneous nodes** and kills/restarts them at random.

## The big idea (Grade-10 summary)

Picture 30 kids spread across a school, each holding a walkie-talkie. They don't know what the whole school looks like — each one only knows who's sitting *right next to them* and how far those neighbors are. How do they end up with a map of the entire school so they can figure out the shortest path from any kid to any other? Three rules:

1. **Heartbeats ("I'm still alive!").** Every half-second, each kid yells *"I'm alive!"* to their direct neighbors. If a kid doesn't hear from a neighbor for too long (10× the heartbeat interval), they assume that neighbor lost power and remove them. The `kill` command is silent on purpose — it simulates a real power outage, so survival depends on missing heartbeats, not on goodbye messages.

2. **Tell the whole school who you know.** Each kid periodically sends a **link-state advertisement** listing *"these are my direct neighbors and the cost to reach them"*. Every neighbor who receives it forwards it to *their* neighbors, so the info **floods** through the entire network. Each advertisement carries a **sequence number** — if you've already seen a higher number from the same kid, you drop the duplicate. This kills routing loops and the broadcast storm that would otherwise crash the network.

3. **Dijkstra finds the shortest path.** Once every kid has a full map — *"who's connected to who, and how expensive each link is"* — they run **Dijkstra's algorithm** on that map. The output is a ranked list of how far every other kid is, going through the cheapest route. That's `shortest_path()`, returned by the `rank` command.

The whole thing runs on **four threads per node**: one shouting "I'm alive", one listening for incoming packets, one cleaning up dead neighbors, and one broadcasting link-state advertisements. New peers can be wired in at runtime with `addneighbor` — the new neighbor *automatically* detects the link via the incoming heartbeat, without needing its own config file to be edited.

## Commands the node accepts

Once a node is running, the CLI accepts:

| Command | What it does |
|---------|--------------|
| `uuid` | Print this node's UUID as a Python dict |
| `neighbors` | Print active direct neighbors (uuid, host, backend_port, metric) |
| `map` | Print the full network graph this node has reconstructed |
| `rank` | Run Dijkstra and print shortest-path costs to every other node |
| `addneighbor uuid=… host=… backend_port=… metric=…` | Wire in a new neighbor at runtime |
| `kill` | Shut the node down silently — neighbors must detect via missed keepalives |

**Stack:** Python, raw `socket` (SOCK_STREAM / TCP), `threading`

[Design document (PDF)](/projects/link-state-routing/design.pdf) · [Download content_server.py](/projects/link-state-routing/content_server.py)

<details>
<summary><strong>content_server.py</strong></summary>

```python
import socket, sys
import ast
import threading, time
import random

BUFSIZE = 1024  # size of receiving buffer
ALIVE_SGN_INTERVAL = 0.5  # interval to send alive signal
TIMEOUT_INTERVAL = 10*ALIVE_SGN_INTERVAL
UPSTREAM_PORT_NUMBER = 1111 # socket number for UL transmission


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

        self.dl_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dl_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.dl_socket.bind(("", self.backendPort))
        self.dl_socket.listen(100)

        self.neighbors = {}
        self.map = {self.name: {}}
        self.seqNumbers = {}
        self.ownSeqNumbers = 0

        for peerInfo in self.peers:
            self.addneighbor(peerInfo[1],peerInfo[2],peerInfo[3],peerInfo[4])

        own = {}
        for uuid, info in self.neighbors.items():
            i = self.uuidToName.get(uuid,uuid)
            own[i] = info['metric']
        self.map[self.name] = own

        self.remain_threads = True
        self.alive()
        return

    def addneighbor(self, uuid, host, backend_port, metric):
        self.neighbors[uuid] = {'host': host, 'backend': backend_port, 'metric': metric, 'lastAlive': time.time()}
        return

    def link_state_adv(self):
        while self.remain_threads:
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
                # MSG Format: <UUID>:<NAME>:<SEQNUM>:<NEIGHBOR IDS>
                MSG = "Link State Packet:" + self.uuid + ":" + self.name + ":" + str(self.ownSeqNumbers) + ":" + neighbor_string
                try:
                   ul_socket.connect((info['host'], info['backend']))
                   ul_socket.send((MSG).encode())
                   ul_socket.close()
                except socket.error:
                   pass

            self.ownSeqNumbers = self.ownSeqNumbers + 1
            time.sleep(ALIVE_SGN_INTERVAL)
        return

    def link_state_flood(self, send_time, host, msg):
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

        if int(seq) > self.seqNumbers.get(uuid,-1):
            self.seqNumbers[uuid]= int(seq)
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

    def keep_alive(self):
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

    def listen(self):
        self.dl_socket.settimeout(0.1)
        while self.remain_threads:
            try:
                connection_socket, client_address = self.dl_socket.accept()
                msg_string = connection_socket.recv(BUFSIZE).decode()
            except socket.timeout:
                msg_string = ""
                pass

            if msg_string == "":
                pass
            elif msg_string.startswith("Alive:"):
                msg_trunc = msg_string.split(":",4)
                uuid= msg_trunc[1]
                if uuid in self.neighbors:
                    self.neighbors[uuid]['lastAlive'] = float(msg_trunc[2])
                else:
                    self.addneighbor(uuid, client_address[0], int(msg_trunc[3]), int(msg_trunc[4]))
            elif msg_string.startswith("Link State Packet"):
                self.link_state_flood(str(time.time()), client_address[0], msg_string[len("Link State Packet:"):])
            elif msg_string == "Death message":
                pass

    def timeout_old(self):
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
        return

    def shortest_path(self):
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
        keep_alive = threading.Thread(target=self.keep_alive)
        listen = threading.Thread(target=self.listen)
        timeout_old = threading.Thread(target=self.timeout_old)
        link_state_adv = threading.Thread(target=self.link_state_adv)
        keep_alive.start()
        listen.start()
        timeout_old.start()
        link_state_adv.start()
        while self.remain_threads:
            command_line = input().split(" ")
            command = command_line[0]
            if command == "kill":
                self.remain_threads = False
            elif command == "uuid":
                print(f'{{"uuid": "{self.uuid}"}}', flush = True)
            elif command == "neighbors":
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
                args = {}
                i = 1
                while i < len(command_line):
                    token = command_line[i]
                    if '=' in token:
                        k, _, v = token.partition('=')
                        if v == '':
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
```

</details>
