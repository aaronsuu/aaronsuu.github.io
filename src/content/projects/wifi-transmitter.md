---
title: Wi-Fi Transmitter & Receiver
description: Python implementation of a layered Wi-Fi PHY — interleaving, convolutional coding, QAM, OFDM, and packet detection over AWGN.
keywords: [Python, NumPy, DSP, OFDM, Wi-Fi, Signal Processing]
order: 0
---

A from-scratch Python implementation of a Wi-Fi physical layer, built up in four levels:

1. **Bit interleaving** with a length header repetition-coded for robustness
2. **Rate-1/2 convolutional coding** + **4-QAM modulation**
3. **OFDM** via 64-point IFFT/FFT
4. **Packet detection** over an AWGN channel using a known preamble

The receiver inverts each stage — soft-decision Viterbi decoding for the convolutional code, FFT-based OFDM demod, and matched-filter packet detection on the preamble.

**Stack:** Python, NumPy, [CommPy](https://commpy.readthedocs.io/)

[Design document (PDF)](/projects/wifi/design.pdf) · [Download transmitter](/projects/wifi/wifitransmitter.py) · [Download receiver](/projects/wifi/wifireceiver.py)

<details>
<summary><strong>wifitransmitter.py</strong></summary>

```python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import commpy as comm
import commpy.channelcoding.convcode as check

def WifiTransmitter(*args):
    # Default Values
    if len(args)<2:
        # Arg1 = Message, Arg2 = Level, Arg3 = SNR
        message = args[0]
        level=4
        snr=np.inf
    elif len(args)<3:
        # Arg1 = Message, Arg2 = Level, Arg3 = SNR
        message=args[0]
        level=int(args[1])
        snr=np.inf
    elif len(args)<4:
        # Arg1 = Message, Arg2 = Level, Arg3 = SNR
        message=args[0]
        level=int(args[1])
        snr=int(args[2])

    ## Sanity checks
    if len(message) > 10000:
        raise Exception("Error: Message is too long")
    if level>4 or level<1:
        raise Exception("Error:Invalid Level, must be 1-4")


    nfft = 64
    Interleave = np.reshape(np.transpose(np.reshape(np.arange(1, 2*nfft+1, 1),[-1,4])),[-1,])
    length = len(message)
    preamble = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    cc1 = check.Trellis(np.array([3]),np.array([[0o7,0o5]]))
    if level >= 1:
        bits = np.unpackbits(np.array([ord(c) for c in message], dtype=np.uint8))
        bits = np.pad(bits, (0, 2*nfft-len(bits)%(2*nfft)),'constant')
        nsym = int(len(bits)/(2*nfft))
        output = np.zeros(shape=(len(bits),))
        for i in range(nsym):
            symbol = bits[i*2*nfft:(i+1)*2*nfft]
            output[i*2*nfft:(i+1)*2*nfft] = symbol[Interleave-1]

        # repetitive encoding for the length field
        length_encoded = ''.join([b+b+b for b in np.binary_repr(length)])
        len_binary = np.array(list(length_encoded.zfill(2*nfft))).astype(np.int8)
        output = np.concatenate((len_binary, output))

    if level >= 2:
        coded_message = check.conv_encode(output[2*nfft:].astype(bool), cc1)
        coded_message = coded_message[:-6]
        output = np.concatenate((output[:2*nfft],coded_message))
        output = np.concatenate((preamble, output))
        mod = comm.modulation.QAMModem(4)
        output = mod.modulate(output.astype(bool))

    if level >= 3:
        nsym = int(len(output)/nfft)
        for i in range(nsym):
            symbol = output[i*nfft:(i+1)*nfft]
            output[i*nfft:(i+1)*nfft] = np.fft.ifft(symbol)

    if level >= 4:
        noise_pad_begin = np.zeros(np.random.randint(1,1000))
        noise_pad_begin_length = len(noise_pad_begin)
        noise_pad_end = np.zeros(np.random.randint(1,1000))
        output = np.concatenate((noise_pad_begin,output,noise_pad_end))
        output = comm.channels.awgn(output,snr)
        return noise_pad_begin_length, output, length

    return output

if __name__ == '__main__':
    if len(sys.argv)<2:
        raise Exception("Error: No message was provided")
    elif len(sys.argv)<3:
        result = WifiTransmitter(sys.argv[1])
    elif len(sys.argv)<4:
        result = WifiTransmitter(sys.argv[1], sys.argv[2])
    elif len(sys.argv)<5:
        result = WifiTransmitter(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv)>=5:
        raise Exception("Error: Number of arguments exceed the maximum arguments allowed (3)")
    print(result)
```

</details>

<details>
<summary><strong>wifireceiver.py</strong></summary>

```python
# -*- coding: utf-8 -*-
from random import randint
import numpy as np
import sys
import commpy as comm
import commpy.channelcoding.convcode as check
#from pip import main
import matplotlib.pyplot as plt


def WifiReceiver(input_stream, level):

    nfft = 64
    Interleave_tr = np.reshape(np.transpose(np.reshape(np.arange(1, 2*nfft+1, 1),[-1,4])),[-1,])
    preamble = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1,1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    cc1 = check.Trellis(np.array([3]),np.array([[0o7,0o5]]))

    # set zero padding to be 0, by default
    begin_zero_padding = 0
    message=""
    length=0

    if level >= 4:
        #Input QAM modulated + Encoded Bits + OFDM Symbols in a long stream
        #Output Detected Packet set of symbols
        begin_zero_padding = wakeUpAndStartListening(preambleToComplex(preamble), input_stream)
        signalLength = lengthOfSignal(begin_zero_padding, input_stream)
        samples = ((8 * signalLength) // 128 + 1)*128 + 128
        input_stream = input_stream[begin_zero_padding:begin_zero_padding+samples]


    if level >= 3:
        #Input QAM modulated + Encoded Bits + OFDM Symbols
        #Output QAM modulated + Encoded Bits
        input_stream = chunksToSample(nfft, input_stream)


    if level >= 2:
        #Input QAM modulated + Encoded Bits
        #Output Interleaved bits + Encoded Length
        bits = qamDemodulate(input_stream)
        bits = bits[128:]
        # strip the 128-bit preamble prepended at Level 2
        length = bits[:128]
        coded = bits[128:]
        codedComplex = input_stream[128:]
        code = np.column_stack((codedComplex.real, codedComplex.imag))
        links, prev = trellisSoft(code, trellisRoadMap())
        decode = np.array(shortestPath(links,prev, trellisRoadMap(), code), dtype=int)
        input_stream = np.concatenate((length, decode))


    if level >= 1:
        #Input Interleaved bits + Encoded Length
        #Output Deinterleaved bits
        length = decodeLengthHeader(input_stream[:128])
        deinterleaved = unInterLeavePayload(input_stream[128:], Interleave_tr)
        message = bitsBytesAndString(deinterleaved, length)
        return begin_zero_padding, message, length

    raise Exception("Error: Unsupported level")




def decodeLengthHeader(bits): #Level 1
    #Delete leading zeros to the code
    isLeadingZeros = True
    firstOnePosition = 0
    while isLeadingZeros:
        if bits[firstOnePosition] == 1:
            isLeadingZeros = False
        else:
            firstOnePosition += 1
    bitsZerosCleaned = bits[firstOnePosition:]

    #As the transmitter is always divisable into three.
    groups = bitsZerosCleaned.reshape(-1,3)
    decodedBits = []
    for row in groups:
        majority = row[0] + row[1] + row[2]
        if majority >= 2:
            decodedBits.append(1)
        else: #If it is not the majority
            decodedBits.append(0)
    binaryString = ""
    for b in decodedBits:
        binaryString = binaryString + str(b)
    length = int(binaryString, 2)
    return length

def unInterLeavePayload(payload, Interleave_tr): #Level 1
    inverse_perm = np.argsort(Interleave_tr - 1)
    deinterleaved = np.zeros(len(payload), dtype = int)
    for i in range(len(payload) // 128):
        chunk = payload[i*128:(i+1)*128]
        deinterleaved[i*128:(i+1)*128] = chunk[inverse_perm]
    return deinterleaved


def bitsBytesAndString(deinterleaved,length): #Level 1
    packedArray = np.packbits(deinterleaved)
    message = ""
    for i in packedArray:
        message = message + chr(i)
    return message[:length]


def qamDemodulate(symbols): #Level 2
    mod = comm.modulation.QAMModem(4) #https://commpy.readthedocs.io/en/latest/modulation.html
    #input_symbols, demod_type
    demodulatedResult = mod.demodulate(symbols, demod_type='hard').astype(int) #https://commpy.readthedocs.io/en/latest/modulation.html?highlight=demodulate

    return demodulatedResult

def trellisRoadMap(): #lvl2
    trellisCompass = np.array([
        #Assume 0 = 00, 1 = 10, 2 = 01, 3 = 11

        #00
        [[0,0,0],  [1,1,1]],
        #10
        [[2,1,0],[3,0,1]],
        #01
        [[0,1,1], [1,0,0]],
        #11
        [[2,0,1],
          [3,1,0]]
    ])

    return trellisCompass

def trellisActual(code, trellisRoadMap): #lvl2
    totalColumns = len(code) + 1
    #initalize the array
    hammingDist = np.full((4,totalColumns), np.inf) #Intalizes haming distance of 4 x N matrix, where it is all infinity for now as no data has been written in it.
    hammingDist[0,0] = 0 #Start
    previous = np.full((4,totalColumns), 0)
    for i in range(len(code)):
        for j in range(4):
            if hammingDist[j,i] == np.inf:
                continue #As it is no path possible

            edges = trellisRoadMap[j]
            next1, out1_0, out2_0 = edges[0]
            next2, out1_1, out2_1 = edges[1]

            path1Sum = 0
            a,b = code[i]
            if out1_0-a != 0:
                path1Sum = 1
            if out2_0-b != 0:
                path1Sum = path1Sum + 1

            path2Sum = 0
            if out1_1-a != 0:
                path2Sum = 1
            if out2_1-b != 0:
                path2Sum = path2Sum + 1

            candidate1 = hammingDist[j,i] + path1Sum
            candidate2 = hammingDist[j,i] + path2Sum
            if candidate1 < hammingDist[next1, i+1]:
                hammingDist[next1, i+1] = candidate1
                previous[next1, i+1] = j
            if candidate2 < hammingDist[next2, i+1]:
                hammingDist[next2, i+1] = candidate2
                previous[next2, i+1] = j



    return hammingDist, previous




def shortestPath(hammingDist, previous, trellisRoadMap,code): #lvl2
    timeStep = len(code)
    curr = int(np.argmin(hammingDist[:,timeStep]))
    bits=[]
    #Walking Backwards
    for i in range(timeStep, 0 ,-1):
        prev = previous[curr,i]
        if trellisRoadMap[prev, 0, 0] == curr:
            bits.append(0)
        else:
            bits.append(1)
        curr = prev
    bits.reverse()
    return bits


def chunksToSample(nfft, input): #Lvl3
    totalChunks = len(input) // nfft
    complexArr = np.zeros(totalChunks * nfft, dtype=complex)
    for i in range(totalChunks):
        fromRange = i*nfft
        toRange = nfft*(i+1)
        complexArr[fromRange:toRange] = np.fft.fft(input[fromRange:toRange]) #Forier Transforms it into complex array for fur. processing
    return complexArr

def preambleToComplex(preamble): #Lvl 4
    #Turns the preamble into 4 QAM to make it easier to detect it

    fourQAMMod = comm.modulation.QAMModem(4)
    complexPreamble = fourQAMMod.modulate(preamble.astype(bool))
    timePreamble = np.fft.ifft(complexPreamble)
    return timePreamble



def wakeUpAndStartListening(complexPremable, signal): #Lvl 4 Packet Detection


    #y(t) = int(h(tau)s(t-tau) + n(t)) = h(t) * s(t) + n(t)
    lengthSignal = len(signal)
    lengthPreamble = len(complexPremable)
    scores = np.zeros(1+lengthSignal-lengthPreamble)
    conjugate = np.conj(complexPremable)
    for i in range(lengthSignal-lengthPreamble+1):
        window = signal[i:i+lengthPreamble]
        scores[i] = abs(np.trapz(window*conjugate)) #Trapz Integration...still having flashbacks from MATH 125

    return np.argmax(scores)

##########################################################################################################################################

def lengthOfSignal(start, signal):
    ofdmLen = signal[start+64:start+128]
    qamLen = np.fft.fft(ofdmLen)
    length = qamDemodulate(qamLen)
    result = decodeLengthHeader(length)
    return result



def trellisSoft(code, trellisRoadMap): #lvl2
    totalColumns = len(code) + 1
    #initalize the array
    imaginaryDist = np.full((4,totalColumns), np.inf) #Intalizes haming distance of 4 x N matrix, where it is all infinity for now as no data has been written in it.
    imaginaryDist[0,0] = 0 #Start
    previous = np.full((4,totalColumns), 0)
    for i in range(len(code)):
        for j in range(4):
            if imaginaryDist[j,i] == np.inf:
                continue #As it is no path possible

            edges = trellisRoadMap[j]
            next1, out1_0, out2_0 = edges[0]
            next2, out1_1, out2_1 = edges[1]

            path1Sum = 0
            path2Sum = 0
            a,b = code[i]
            path1Sum = (a-(2*out1_0 - 1))**2 + (b-(2*out2_0 - 1))**2
            path2Sum = (a-(2*out1_1 - 1))**2 + (b-(2*out2_1 - 1))**2




    

            candidate1 = imaginaryDist[j,i] + path1Sum
            candidate2 = imaginaryDist[j,i] + path2Sum
            if candidate1 < imaginaryDist[next1, i+1]:
                imaginaryDist[next1, i+1] = candidate1
                previous[next1, i+1] = j
            if candidate2 < imaginaryDist[next2, i+1]:
                imaginaryDist[next2, i+1] = candidate2
                previous[next2, i+1] = j



    return imaginaryDist, previous




# for testing purpose
from wifitransmitter import WifiTransmitter
if __name__ == "__main__":
    test_case = 'The Internet has transformed our everyday lives, bringing people closer together and powering multi-billion dollar industries. The mobile revolution has brought Internet connectivity to the last-mile, connecting billions of users worldwide. But how does the Internet work? What do oft repeated acronyms like "LTE", "TCP", "WWW" or a "HTTP" actually mean and how do they work? This course introduces fundamental concepts of computer networks that form the building blocks of the Internet. We trace the journey of messages sent over the Internet from bits in a computer or phone to packets and eventually signals over the air or wires. We describe commonalities and differences between traditional wired computer networks from wireless and mobile networks. Finally, we build up to exciting new trends in computer networks such as the Internet of Things, 5-G and software defined networking. Topics include: physical layer and coding (CDMA, OFDM, etc.); data link protocol; flow control, congestion control, routing; local area networks (Ethernet, Wi-Fi, etc.); transport layer; and introduction to cellular (LTE) and 5-G networks. The course will be graded based on quizzes (on canvas), a midterm and final exam and four projects (all individual). '
    symbols = [randint(0, 1) for i in range(32*8)]
    print(test_case)
    one, output, three = WifiTransmitter(test_case, 4)
    begin_zero_padding, message, length_y = WifiReceiver(output, 4)
    print(begin_zero_padding, message, length_y)
    print(test_case == message)
```

</details>
