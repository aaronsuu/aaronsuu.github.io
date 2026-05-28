# -*- coding: utf-8 -*-
"""
Sweep SNR (including negative values) and compare decode success
between the soft-Viterbi receiver (wifireceiver.py) and a hard-Viterbi
receiver (wifireceiver_hard.py, if present).

Level 4 is the only level that adds AWGN — see wifitransmitter.py line 70.
"""
import sys
import warnings
import numpy as np

from wifitransmitter import WifiTransmitter
from wifireceiver import WifiReceiver as WifiReceiverSoft

try:
    from wifireceiver_hard import WifiReceiver as WifiReceiverHard
    HARD_AVAILABLE = True
except ImportError:
    HARD_AVAILABLE = False
    print("[note] wifireceiver_hard.py not found — running soft-only sweep.\n")


# ---- config --------------------------------------------------------------
INTERNET_MSG = """The Internet has transformed our everyday lives, bringing people closer together and powering multi-billion dollar industries. The mobile revolution has brought Internet connectivity to the last-mile, connecting billions of users worldwide. But how does the Internet work? What do oft repeated acronyms like "LTE", "TCP", "WWW" or a "HTTP" actually mean and how do they work? This course introduces fundamental concepts of computer networks that form the building blocks of the Internet. We trace the journey of messages sent over the Internet from bits in a computer or phone to packets and eventually signals over the air or wires. We describe commonalities and differences between traditional wired computer networks from wireless and mobile networks. Finally, we build up to exciting new trends in computer networks such as the Internet of Things, 5-G and software defined networking. Topics include: physical layer and coding (CDMA, OFDM, etc.); data link protocol; flow control, congestion control, routing; local area networks (Ethernet, Wi-Fi, etc.); transport layer; and introduction to cellular (LTE) and 5-G networks. The course will be graded based on quizzes (on canvas), a midterm and final exam and four projects (all individual). """

CONSTITUTION_MSG = """Section. 1.
All legislative Powers herein granted shall be vested in a Congress of the United States, which shall consist of a Senate and House of Representatives.

Section. 2.
The House of Representatives shall be composed of Members chosen every second Year by the People of the several States, and the Electors in each State shall have the Qualifications requisite for Electors of the most numerous Branch of the State Legislature.

No Person shall be a Representative who shall not have attained to the Age of twenty five Years, and been seven Years a Citizen of the United States, and who shall not, when elected, be an Inhabitant of that State in which he shall be chosen.

Representatives and direct Taxes shall be apportioned among the several States which may be included within this Union, according to their respective Numbers, which shall be determined by adding to the whole Number of free Persons, including those bound to Service for a Term of Years, and excluding Indians not taxed, three fifths of all other Persons. The actual Enumeration shall be made within three Years after the first Meeting of the Congress of the United States, and within every subsequent Term of ten Years, in such Manner as they shall by Law direct. The Number of Representatives shall not exceed one for every thirty Thousand, but each State shall have at Least one Representative; and until such enumeration shall be made, the State of New Hampshire shall be entitled to chuse three, Massachusetts eight, Rhode-Island and Providence Plantations one, Connecticut five, New-York six, New Jersey four, Pennsylvania eight, Delaware one, Maryland six, Virginia ten, North Carolina five, South Carolina five, and Georgia three.

When vacancies happen in the Representation from any State, the Executive Authority thereof shall issue Writs of Election to fill such Vacancies.

The House of Representatives shall chuse their Speaker and other Officers; and shall have the sole Power of Impeachment.

Section. 3.
The Senate of the United States shall be composed of two Senators from each State, chosen by the Legislature thereof, for six Years; and each Senator shall have one Vote.

Immediately after they shall be assembled in Consequence of the first Election, they shall be divided as equally as may be into three Classes. The Seats of the Senators of the first Class shall be vacated at the Expiration of the second Year, of the second Class at the Expiration of the fourth Year, and of the third Class at the Expiration of the sixth Year, so that one third may be chosen every second Year; and if Vacancies happen by Resignation, or otherwise, during the Recess of the Legislature of any State, the Executive thereof may make temporary Appointments until the next Meeting of the Legislature, which shall then fill such Vacancies.

No Person shall be a Senator who shall not have attained to the Age of thirty Years, and been nine Years a Citizen of the United States, and who shall not, when elected, be an Inhabitant of that State for which he shall be chosen.

The Vice President of the United States shall be President of the Senate, but shall have no Vote, unless they be equally divided.

The Senate shall chuse their other Officers, and also a President pro tempore, in the Absence of the Vice President, or when he shall exercise the Office of President of the United States.

The Senate shall have the sole Power to try all Impeachments. When sitting for that Purpose, they shall be on Oath or Affirmation. When the President of the United States is tried, the Chief Justice shall preside: And no Person shall be convicted without the Concurrence of two thirds of the Members present.

Judgment in Cases of Impeachment shall not extend further than to removal from Office, and disqualification to hold and enjoy any Office of honor, Trust or Profit under the United States: but the Party convicted shall nevertheless be liable and subject to Indictment, Trial, Judgment and Punishment, according to Law.

Section. 4.
The Times, Places and Manner of holding Elections for Senators and Representatives, shall be prescribed in each State by the Legislature thereof; but the Congress may at any time by Law make or alter such Regulations, except as to the Places of chusing Senators.

The Congress shall assemble at least once in every Year, and such Meeting shall be on the first Monday in December, unless they shall by Law appoint a different Day.
"""

# Transmitter caps messages at 10000 chars — slice defensively.
TEST_MESSAGES = [
    INTERNET_MSG[:9500],
    CONSTITUTION_MSG[:9500],
]
SNR_VALUES = [30, 20, 15, 12, 10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10]
TRIALS_PER_SNR = 8
# -------------------------------------------------------------------------


def try_decode(receiver_fn, signal, truth):
    """Run a receiver, return (exact_match, char_match_rate)."""
    try:
        _, message, _ = receiver_fn(signal, 4)
    except Exception:
        return False, 0.0
    if not isinstance(message, str):
        return False, 0.0
    exact = (message == truth)
    n = max(len(truth), 1)
    matched = sum(1 for a, b in zip(message, truth) if a == b)
    return exact, matched / n


def sweep():
    header = f"{'SNR (dB)':>9} | {'soft exact':>11} {'soft char%':>11}"
    if HARD_AVAILABLE:
        header += f" | {'hard exact':>11} {'hard char%':>11}"
    print(header)
    print("-" * len(header))

    for snr in SNR_VALUES:
        soft_exact = 0
        soft_char = 0.0
        hard_exact = 0
        hard_char = 0.0

        for t in range(TRIALS_PER_SNR):
            msg = TEST_MESSAGES[t % len(TEST_MESSAGES)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, signal, _ = WifiTransmitter(msg, 4, snr)

            ok, rate = try_decode(WifiReceiverSoft, signal, msg)
            soft_exact += int(ok)
            soft_char += rate

            if HARD_AVAILABLE:
                ok, rate = try_decode(WifiReceiverHard, signal, msg)
                hard_exact += int(ok)
                hard_char += rate

        soft_rate = soft_exact / TRIALS_PER_SNR
        soft_char_rate = soft_char / TRIALS_PER_SNR
        row = f"{snr:>9} | {soft_rate:>11.2f} {soft_char_rate:>11.2%}"
        if HARD_AVAILABLE:
            hard_rate = hard_exact / TRIALS_PER_SNR
            hard_char_rate = hard_char / TRIALS_PER_SNR
            row += f" | {hard_rate:>11.2f} {hard_char_rate:>11.2%}"
        print(row)


if __name__ == "__main__":
    print(f"Trials per SNR: {TRIALS_PER_SNR}")
    print(f"Messages per sweep: {len(TEST_MESSAGES)}\n")
    sweep()
