# APP BREATHE
## Overview
Breathe is a mobile and IoT application (built with MIT App Inventor) for real-time air-quality monitoring and guidance. It blends satellite observations (e.g., Sentinel-5P/TROPOMI), hyperlocal readings from low-cost sensors, and weather data to deliver clear, personalized health recommendations. An AI chatbot (ChatGPT-powered) explains pollutant levels, safety actions, and concepts in simple language. 

## How it works
Breathe Node (IoT): A portable ESP32-based device (with PMS5003, MQ-131, MQ-4) captures street-level PM and gas data.

Python Server: Receives JSON from nodes, fuses sensor, satellite, and meteorological sources, computes local air-quality indicators, and returns tailored guidance.

Mobile App: Shows current conditions, short-term trends, dominant pollutants, and personalized recommendations; users can chat with the built-in assistant for explanations.

