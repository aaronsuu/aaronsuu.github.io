---
title: Content Distribution Overlay (Link-State Routing)
description: An overlay network of peer nodes that discover each other and compute shortest paths using a from-scratch link-state routing protocol with keepalive heartbeats, sequence-numbered LSA flooding, and Dijkstra over a graph of up to 32 nodes.
keywords: [Python, Networking, TCP, Sockets, Routing, Distributed Systems, Threading]
minorTags: ["EE 419", "Computer Networking", "Link-State Routing", "Overlay Network", "Peer-to-Peer", "Dijkstra", "LSA Flooding", "Keepalive", "Sequence Numbers", "UUID", "Graph Discovery"]
order: 1
lastUpdated: 2026-05-27
previews:
  - { label: Design Document, href: /projects/link-state-routing/design.pdf }
  - { label: content_server.py, href: link-state-routing/content_server.py }
---

Content on the Internet does not live on one machine. A Netflix episode is copied across hundreds of servers, and your device has to figure out which one to reach and how to get there cheaply. This project builds the foundation for that: an overlay network of peer nodes that discover each other in real time using a link-state routing protocol, then compute the shortest path to every other node in the network.

Each node runs `content_server.py` with a config file that lists its UUID, name, backend port, and the cost to each direct neighbor. From that starting point, every node builds a complete map of the entire network and keeps it up to date as peers go offline or new ones join. The grader tests with up to 32 simultaneous nodes and kills or restarts them at random.

Picture 30 people spread across a building, each with a radio. Nobody knows the full layout. Each person only knows who is standing right next to them and how far those neighbors are. Three rules let them figure out the shortest path from anyone to anyone else.

First, every half-second each person broadcasts a short "I am still here" message to their direct neighbors. If a neighbor goes quiet for too long (ten times the heartbeat interval), they are considered offline and removed from the list. When a node is shut down with the `kill` command, it goes silent on purpose rather than sending a goodbye. Neighbors have to notice the missing heartbeats on their own, just like a real server that loses power with no warning.

Second, each node regularly sends a link-state advertisement listing its direct neighbors and the cost to reach each one. Every node that receives this message forwards a copy to its own neighbors, so the information spreads through the whole network. Each advertisement carries a sequence number. If a node has already seen a higher number from the same source, it drops the duplicate. This stops old information from looping forever.

Third, once a node has collected link-state advertisements from everyone, it runs Dijkstra's algorithm on the full graph and produces a ranked list of the cheapest cost to reach every other node. That is the output of the `rank` command.

New neighbors can be added at runtime with `addneighbor`. The new neighbor detects the link automatically when the first heartbeat arrives, without needing its config file changed.

Four threads run per node: one sending heartbeats, one listening for incoming messages, one pruning dead neighbors, and one broadcasting link-state advertisements.

## Commands

| Command | Output |
|---------|--------|
| `uuid` | This node's unique ID |
| `neighbors` | Active direct neighbors with host, port, and link cost |
| `map` | Full network graph reconstructed from link-state advertisements |
| `rank` | Shortest-path cost from this node to every other node |
| `addneighbor uuid=... host=... backend_port=... metric=...` | Adds a neighbor at runtime |
| `kill` | Shuts the node down silently |

**Stack:** Python, raw `socket` (SOCK_STREAM), `threading`
