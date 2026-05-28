---
title: Wi-Fi Transmitter & Receiver
description: From-scratch Python implementation of a Wi-Fi physical layer receiver — de-interleaving, soft-decision Viterbi decoding over a rate-1/2 trellis, OFDM demodulation via FFT, and matched-filter packet detection that stays reliable down to negative SNR.
coverHtml: /projects/wifi/cover.html
keywords: [Python, NumPy, DSP, OFDM, Wi-Fi, Signal Processing, Viterbi, AWGN]
minorTags: ["EE 419", "Computer Networking", "802.11", FFT, QAM, Trellis, Viterbi, Convolutional Coding, Packet Detection, Modulation, Demodulation, IFFT, Preamble Detection, SNR]
order: 0
lastUpdated: 2026-05-27
previews:
  - { label: Design Document, href: /projects/wifi/design.pdf }
  - { label: wifitransmitter.py, href: wifi-transmitter/wifitransmitter.py }
  - { label: wifireceiver.py, href: wifi-transmitter/wifireceiver.py }
  - { label: noisecomparison.py, href: wifi-transmitter/noisecomparison.py }
---

Every Wi-Fi chip in your phone or laptop runs a physical layer pipeline that turns raw bits into a radio signal and back again. This project builds that pipeline in Python, both sides of it. A transmitter was provided. The job was to write the receiver from scratch, reversing every encoding step to recover the original message.

The transmitter passes a message through four stages before it becomes a signal. The receiver has to undo all four in reverse order.

**Level 1 — Interleaving.** The bits are shuffled using a fixed permutation before transmission. A burst of interference that wipes out 10 consecutive bits is much worse than 10 scattered errors. Interleaving converts the first problem into the second. The receiver applies the inverse permutation to put the bits back in order. The message length is sent ahead of the payload and protected separately using a 3x repetition code, so each length bit is voted on by three received copies.

**Level 2 — Trellis coding and 4-QAM modulation.** A rate-1/2 convolutional code is applied, meaning every input bit produces two output bits that carry redundancy for error correction. The coded bits are then mapped to complex numbers using 4-QAM, packing two bits per symbol. The receiver demodulates the symbols and runs a **soft-decision Viterbi decoder** written from scratch over the trellis to recover the original bits. Using CommPy's built-in `viterbi_decode()` carried a 30-point penalty and zero credit on the SNR robustness test, so the entire trellis search was implemented manually. Soft Viterbi uses the raw signal amplitude rather than hard bit decisions, which makes it significantly more robust at low SNR.

**Level 3 — OFDM.** The modulated symbols are split across 64 parallel sub-carriers using an inverse FFT. Each sub-carrier carries a small piece of the signal independently, which makes the whole transmission resistant to multipath fading. The receiver runs a forward FFT on each 64-sample block to pull the sub-carriers back apart.

**Level 4 — Packet detection in noise.** The signal is buried inside a stream of AWGN noise with a random amount of silence at the front. The receiver uses a matched filter cross-correlating against a known preamble to find where the packet starts, then extracts exactly the right number of samples to decode.

**SNR robustness.** To push the receiver as far as possible, `noisecomparison.py` sweeps SNR from 30 dB down to negative values, running 8 trials per SNR point across two long test messages (the course description and the US Constitution). Soft Viterbi consistently outperforms hard-decision decoding at low SNR, which is the whole reason it was worth implementing from scratch.

**Stack:** Python, NumPy, [CommPy](https://commpy.readthedocs.io/)
