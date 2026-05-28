---
title: Wi-Fi Transmitter & Receiver
description: Python implementation of a layered Wi-Fi physical layer, built from scratch with interleaving, convolutional coding, QAM modulation, OFDM, and matched-filter packet detection over a noisy channel.
coverHtml: /projects/wifi/cover.html
keywords: [Python, NumPy, DSP, OFDM, Wi-Fi, Signal Processing, Viterbi, AWGN]
minorTags: ["EE 419", "Computer Networking", "802.11", FFT, QAM, Convolutional Coding, Packet Detection, Modulation, Demodulation, IFFT, Preamble Detection]
order: 0
lastUpdated: 2026-05-27
previews:
  - { label: Design Document, href: /projects/wifi/design.pdf }
  - { label: wifitransmitter.py, href: wifi-transmitter/wifitransmitter.py }
  - { label: wifireceiver.py, href: wifi-transmitter/wifireceiver.py }
---

This project builds a Wi-Fi physical layer from the ground up in Python. Text goes in one end and comes out the other as a radio-ready signal. The receiver then takes that signal and recovers the original text, even after a noisy channel scrambles it.

The transmitter is built in four stages, each adding a new layer of protection or efficiency:

1. **Bit interleaving** shuffles the bits so a burst of noise hits spread-out positions instead of wiping out a whole chunk. The message length is triple-repeated so the receiver can recover it even if some bits flip.
2. **Convolutional coding and 4-QAM** adds error-correcting redundancy, then packs two bits into each complex symbol so fewer signals carry more data.
3. **OFDM** splits the signal across 64 parallel sub-carriers using an inverse FFT. This makes the transmission resistant to frequency-selective fading, which is how real Wi-Fi works.
4. **Packet detection** drops the signal into a stream of random noise at a random offset, like a real wireless channel. The receiver scans for a known preamble pattern to find where the packet starts.

The receiver runs each stage in reverse. It uses soft-decision Viterbi decoding to recover the convolutional code, FFT to undo OFDM, and a matched filter to lock onto the preamble in noise.

**Stack:** Python, NumPy, [CommPy](https://commpy.readthedocs.io/)
