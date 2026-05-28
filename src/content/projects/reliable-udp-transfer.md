---
title: Custom Transport Layer over UDP
description: A from-scratch TCP-style reliable transport protocol built on top of UDP — three-way handshake, sliding-window ARQ, per-packet ACKs and timeouts, file-ID multiplexing on a single port, and ≥5 simultaneous file transfers robust to 5%+ packet loss.
keywords: [Python, Networking, UDP, Sockets, Reliable Transport, Sliding Window, ARQ, Threading]
minorTags: ["EE 419", "Computer Networking", "Transport Layer", "TCP-like", "Three-Way Handshake", "ACK/Retransmit", "Sliding Window", "Multiplexing", "Packet Loss", "Sequence Numbers", "File Transfer"]
order: 0
lastUpdated: 2026-05-27
---

UDP is fast but completely unreliable — packets show up out of order, get duplicated, or disappear entirely. This project builds a **custom transport layer on top of UDP** (TCP is forbidden) that reliably moves files between peer nodes, even when the underlying network is dropping 5–20% of packets. Every node runs the same program (`tcpserver.py`) with a JSON config listing its port and which peers hold which files. The receiver types `FILE_NAME` and the file lands on disk — that's the entire user-facing API.

Three requirements drive the protocol design:

- **Multiplex on a single UDP port.** Every transfer in *both* directions — uploads and downloads, all happening at once — has to fit through one socket. The packet header carries a `File ID` so a node can demultiplex many simultaneous flows from the same port.
- **Robust to ≥5% packet loss.** Tested locally with a 10%–20% drop script to leave safety margin for the 5% grading test.
- **≥5 simultaneous transfers.** Each transfer runs on its own pair of TX/RX threads sharing the one socket.

## The big idea (Grade-10 summary)

Imagine you're texting a long story to a friend, one sentence per text, but the cell tower is flaky — about 1 in 5 messages just never arrive, and the rest sometimes arrive out of order. UDP is exactly that. To still get the whole story across without gaps, I built five rules on top:

1. **The three-way handshake.** Before any data, both sides shake hands with three messages — *"Can I have file X?"* (SYN) → *"Yes, it's 200 packets long"* (SYN-ACK) → *"Got it, start sending"* (ACK). This is the same opening dance TCP uses, just rebuilt by hand on top of UDP.
2. **Numbered packets.** Every chunk of the file gets a sequence number (0, 1, 2, …) so the receiver can stitch them back together in the right order even when they arrive scrambled.
3. **Acknowledgements (ACKs).** Every time a packet lands safely, the receiver shouts back *"Got #42!"*. If the sender doesn't hear an ACK within 100 ms, it assumes the packet was lost and retransmits it.
4. **A sliding window.** Waiting for an ACK after every single packet would be painfully slow, so the sender keeps **128 packets in flight at once**. As ACKs come back, the window slides forward and more packets go out — like a conveyor belt instead of a one-at-a-time queue.
5. **The goodbye packet (FIN).** When every chunk has been acknowledged, the sender sends a final FIN so the receiver knows the file is complete and can close out the session.

Multiplexing is the sneaky-clever part: every packet carries a 1-byte **File ID**, so a single node can be uploading three files *and* downloading two more, all flying through the same UDP socket simultaneously, and each thread can still tell its own packets apart from everyone else's.

To prove robustness, `tcpserver_drop.py` randomly throws away **20% of incoming packets** at the receive side — the file still transfers correctly, just slower, because the timeout-and-retransmit logic refills the gaps in the sliding window.

## Packet format

Every packet is a 6-byte binary header plus an optional payload:

| Byte 0 | Byte 1 | Bytes 2–3 | Bytes 4–5 | Bytes 6+ |
|--------|--------|-----------|-----------|----------|
| Type   | File ID | Seq # | Data Len | Payload |

Types: `1=SYN`, `2=SYN-ACK`, `3=DATA`, `4=ACK`, `5=FIN`. Packet payloads are capped at 8192 bytes; large files are chunked across many sequenced DATA packets.

**Stack:** Python, raw `socket` (SOCK_DGRAM), `threading`

[Design document (PDF)](/projects/reliable-udp-transfer/design.pdf) · [Download tcpserver.py](/projects/reliable-udp-transfer/tcpserver.py) · [Download tcpserver_drop.py](/projects/reliable-udp-transfer/tcpserver_drop.py)

<details>
<summary><strong>tcpserver.py</strong> — main reliable transfer server</summary>

```python
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
        with open(config_file, 'r') as file:
            configFile = json.load(file)
            self.port = configFile['port']
            self.peer_num = configFile['peers']
            self.content_info = configFile['content_info']
            self.peer_info = configFile['peer_info']

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("", self.port))
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)
        self.server_socket.settimeout(1)

        self.remain_threads = True
        self.recv_sessions = {}
        self.send_sessions = {}
        self.sessions_lock = threading.Lock()
        self.next_file_id = 0

        self.cli()
        return

    def find_file(self, file_name):
        for peer in self.peer_info:
            if file_name in peer['content_info']:
                return peer['hostname'], peer['port']
        return

    def load_file(self, file_name):
        address = self.find_file(file_name)
        connected = False
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
                connected = False

        ackPacket = packPacket(4, file_id, 0,0 )
        self.server_socket.sendto(ackPacket, address)
        chunks = {}
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

        with open(file_name, 'wb') as file:
            for i in range(packet_count):
                file.write(chunks[i])
        with self.sessions_lock:
            del self.recv_sessions[file_id]

    def read_file(self, file_name):
        with open(file_name, 'rb') as file:
            data = file.read()
            chunks = []
            for i in range(0, len(data), PKTSIZE):
                chunks.append(data[i:i+PKTSIZE])
            return chunks

    def transmit(self, file_name, addr, file_id):
        chunks = self.read_file(file_name)
        packet_num = len(chunks)
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
                        acknowledged = True

        windowStart = [0]
        status = [0] * packet_num
        window_lock = threading.Lock()
        def transmit_thread():
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

    def listener(self):
        while self.remain_threads:
            raw = b""
            try:
                raw, addr = self.server_socket.recvfrom(BUFSIZE)
            except socket.timeout:
                pass

            if raw == b"":
                pass
            else:
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

    def cli(self):
        listen_thread = threading.Thread(target=self.listener)
        listen_thread.start()

        while self.remain_threads:
            command_line = input()
            if command_line == "kill":
                self.remain_threads = False
                return
            else:
                t = threading.Thread(target=self.load_file, args = (command_line,))
                t.start()
        return


if __name__ == "__main__":
    server = Server(sys.argv[1])
```

</details>

<details>
<summary><strong>tcpserver_drop.py</strong> — same protocol with a 20% packet-drop simulator</summary>

```python
# Identical to tcpserver.py, except the listener() drops ~20% of incoming
# packets at random to test the protocol's robustness under heavy packet loss.

# The key difference is in listener():
def listener(self):
    while self.remain_threads:
        raw = b""
        try:
            raw, addr = self.server_socket.recvfrom(BUFSIZE)
            if random.random() < 0.2:   # <-- drop 1 in 5 packets
                raw = b""
        except socket.timeout:
            pass
        # ... rest identical to tcpserver.py
```

Even with one out of every five packets thrown away, the file still transfers correctly — it just takes longer, because the timeout-and-retransmit logic kicks in repeatedly to refill the gaps in the sliding window.

</details>
