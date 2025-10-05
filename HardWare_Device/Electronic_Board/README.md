# Electronic Board
## Design Overview
The Air Quality Monitoring PCB is a custom-designed electronic board built around the ESP32-WROOM-32UE microcontroller.
It serves as a compact and efficient hardware platform for collecting and transmitting environmental data over Wi-Fi.
The PCB integrates multiple sensor interfaces and power management components to ensure stable operation and accurate measurements.
The design prioritizes:

Ease of assembly: plug-and-play sensor sockets.
Signal integrity: proper routing and isolation for analog readings.
Wireless communication: through the integrated Wi-Fi module of the ESP32.
Compact form factor: suitable for enclosure integration and prototyping.

## Main componets
ESP32-WROOM-32UE: Main processing and communication unit, responsible for reading sensor data and transmitting it wirelessly.

Sensor sockets:

PMS5003 – for particulate matter measurement (PM1.0, PM2.5, PM10).
MQ131 – for ozone (O₃) detection.
MQ-4 – for methane (CH₄) and combustible gas sensing.
Voltage regulation stage: Ensures stable 5V and 3.3V power supply to sensors and microcontroller.
Programming and debugging header: Allows firmware upload and serial monitoring.
Status indicators: LEDs for power and Wi-Fi connection status.

##Connectivity and Power
The PCB is powered via a standard 9V ot 12V input, with onboard regulation to 3.3V for the ESP32 and 5V for the Sensors.
All communication with external systems occurs through Wi-Fi, using the ESP32’s internal antenna and u.FL connector for external antennas when longer range is required.
