# Transit Clock

A compact, alarm-clock-sized device that displays real-time public transit arrival estimates on a small OLED. Built around an ESP32, fetches arrivals from a public transit REST API, and renders the next N upcoming departures.

## Hardware
- ESP32 dev board (any flavor with Wi-Fi)
- 128x64 SSD1306 OLED, I²C
- 3D-printed enclosure (STL forthcoming)

## Firmware
The Arduino sketch polls the transit API every 30 s, parses the JSON, and renders the next departure in large type with the second-next in a smaller line below.

## Notes
This README is a placeholder so the in-site viewer has something to render. Replace it with the real project documentation when you write it up.
