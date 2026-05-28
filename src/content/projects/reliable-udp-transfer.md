---
title: Custom Transport Layer over UDP
description: A from-scratch TCP-style reliable transport protocol built on top of UDP with a three-way handshake, sliding-window ARQ, per-packet ACKs, file-ID multiplexing on a single port, and support for at least five simultaneous file transfers under 5%+ packet loss.
keywords: [Python, Networking, UDP, Sockets, Reliable Transport, Sliding Window, ARQ, Threading]
minorTags: ["EE 419", "Computer Networking", "Transport Layer", "TCP-like", "Three-Way Handshake", "ACK/Retransmit", "Sliding Window", "Multiplexing", "Packet Loss", "Sequence Numbers", "File Transfer"]
coverHtml: /projects/reliable-udp-transfer/cover.html
order: 2
lastUpdated: 2026-05-27
previews:
  - { label: Design Document, href: /projects/reliable-udp-transfer/design.pdf }
  - { label: tcpserver.py, href: reliable-udp-transfer/tcpserver.py }
  - { label: tcpserver_drop.py, href: reliable-udp-transfer/tcpserver_drop.py }
---

UDP is fast but completely unreliable. Packets can show up out of order, get duplicated, or vanish entirely. This project builds a custom transport protocol on top of UDP that reliably moves files between peer nodes, even when the network is randomly dropping packets. Using TCP sockets was not allowed, so the entire reliability mechanism had to be built by hand.

Each node runs `tcpserver.py` with a JSON config file that says which port to use and which peers hold which files. A user types a file name and the file lands on disk. That is the entire interface.

Three hard requirements shaped the protocol design. Every transfer in both directions must share a single UDP port. The protocol must survive at least 5% packet loss. At least five simultaneous file transfers must be possible at the same time.

Think of it like texting a long document to a friend over a flaky network where about one message in five never arrives and the rest sometimes show up out of order. To make it work reliably, five rules run underneath every transfer.

Before any data moves, both sides complete a three-step handshake. The sender says "I want file X," the receiver replies "Yes, it is 200 chunks long," and the sender confirms "Got it, start sending." This is the same opening sequence TCP uses, rebuilt here from scratch on top of UDP.

Every chunk of the file gets a sequence number so the receiver can reassemble them in the right order even when they arrive scrambled.

Every time a chunk lands safely, the receiver sends back an acknowledgment. If the sender does not hear back within 100 milliseconds, it assumes the chunk was lost and sends it again.

Instead of waiting for an acknowledgment after every single chunk, the sender keeps 128 chunks in flight at once. As acknowledgments arrive, the window slides forward and new chunks go out. This keeps the link busy rather than sitting idle waiting for each reply.

When all chunks have been acknowledged, the sender sends a final FIN packet so the receiver knows the file is complete.

The multiplexing works through a one-byte File ID in every packet header. A single node can upload three files and download two more at the same time, all through the same UDP socket, and each thread can sort its own packets from the rest.

`tcpserver_drop.py` is the same protocol with a line added to randomly discard 20% of incoming packets. The file still transfers correctly because the timeout-and-retransmit logic fills in the gaps.

## Packet format

| Byte 0 | Byte 1 | Bytes 2-3 | Bytes 4-5 | Bytes 6+ |
|--------|--------|-----------|-----------|----------|
| Type | File ID | Seq # | Data Len | Payload |

Types: `1=SYN`, `2=SYN-ACK`, `3=DATA`, `4=ACK`, `5=FIN`. Payloads are capped at 8192 bytes.

**Stack:** Python, raw `socket` (SOCK_DGRAM), `threading`
