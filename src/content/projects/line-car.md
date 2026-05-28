---
title: Line Following Car
description: Autonomous line-following robot built around custom KiCAD PCBs with an infrared photodetector array and a PID control loop running on a microcontroller.
thumbnail: /images/line-car.jpeg
heroImage: /images/line-car.jpeg
keywords: [KiCAD, PCB Design, PID Control, Embedded, Hardware, Photodetectors, Microcontroller, Autonomous]
minorTags: [Schematic Capture, Soldering, Sensor Fusion, Closed-Loop, Analog, C, Robotics, Prototyping, Op-Amp, EE 271]
order: 4
lastUpdated: 2026-05-27
---

This project is a fully autonomous car that follows a line on the floor using a custom-designed circuit board and a feedback control algorithm, built from scratch in hardware and software.

The sensor board holds a row of infrared photodetectors. Each one shines light downward and reads how much bounces back. A dark line absorbs light and a bright floor reflects it, so the array can tell exactly where the line is relative to the car at any moment.

That position reading feeds into a PID controller running on the microcontroller. PID stands for Proportional, Integral, Derivative. In simple terms, it measures how far off-center the line is, how long it has been drifting, and how fast it is moving away, then combines those three numbers into a single steering correction. Too much correction and the car zigzags. Too little and it drives off the line. Tuning those three values by hand until the car tracked smoothly was the core challenge.

The PCBs were designed in KiCAD, including schematic capture, component placement, and trace routing. After fabrication the boards were hand-soldered and tested before being integrated into the chassis.
