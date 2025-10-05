/*
  ESP32 + PMS5003 + MQ131 + MQ-4  -> HTTPS POST (Arduino)
  Endpoint: https://breathe-nasa-space-apps-2025.onrender.com/cookie_hardware/004

  JSON sent:
  {
    "PMS5003": {"PM1_0": <int>, "PM2_5": <int>, "PM10": <int>},
    "MQ131":   {"mv": <int>},     // raw millivolts (calibrate to ppm in server or later)
    "MQ4":     {"mv": <int>}
  }
*/

#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>

// ---------- Wi-Fi ----------
const char* WIFI_SSID     = "<WIFI_SSID>";
const char* WIFI_PASSWORD = "<WIFI_PASSWORD>";

// ---------- HTTPS endpoint ----------
const char* URL = "https://breathe-nasa-space-apps-2025.onrender.com/cookie_hardware/004";

// ---------- PMS5003 UART (Serial2) ----------
static const int PMS_RX = 16;  // ESP32 RX2  (connect to PMS5003 TX)
static const int PMS_TX = 17;  // ESP32 TX2  (connect to PMS5003 RX)
HardwareSerial PMSSerial(2);   // UART2

// ---------- MQ sensors (ADC1 pins) ----------
static const int MQ131_PIN = 25;  // ADC1_CH4
static const int MQ4_PIN   = 27;  // ADC1_CH5

// Read averaging
static const int ADC_SAMPLES = 16;

// =============== PMS5003 structures ===============
struct PMData {
  uint16_t pm1_0_cf1;
  uint16_t pm2_5_cf1;
  uint16_t pm10_cf1;
  uint16_t pm1_0_atm;
  uint16_t pm2_5_atm;
  uint16_t pm10_atm;
};

// Utility: read one PMS5003 frame (32 bytes), verify header & checksum
bool readPMS5003(PMData& out) {
  // Frame: 0x42 0x4D + 30 bytes
  const uint8_t HEADER1 = 0x42, HEADER2 = 0x4D;
  // Wait until header found
  while (PMSSerial.available() >= 32) {
    if (PMSSerial.peek() == HEADER1) {
      PMSSerial.read();                      // consume 0x42
      if (PMSSerial.peek() == HEADER2) {
        PMSSerial.read();                    // consume 0x4D
        uint8_t buf[30];
        int n = PMSSerial.readBytes(buf, 30);
        if (n != 30) continue;

        // Checksum = sum of all bytes including 0x42,0x4D and 30 bytes
        uint16_t sum = HEADER1 + HEADER2;
        for (int i = 0; i < 28; ++i) sum += buf[i]; // last 2 bytes are checksum
        uint16_t checksum = (buf[28] << 8) | buf[29];
        if (sum != checksum) {
          // bad frame; shift by one and try again
          continue;
        }

        // buf[0..1] = frame length (usually 28), then data words (big-endian)
        auto u16 = [&](int idx) -> uint16_t {
          return (uint16_t)(buf[idx] << 8) | buf[idx + 1];
        };

        out.pm1_0_cf1 = u16(2);
        out.pm2_5_cf1 = u16(4);
        out.pm10_cf1  = u16(6);
        out.pm1_0_atm = u16(8);
        out.pm2_5_atm = u16(10);
        out.pm10_atm  = u16(12);
        return true;
      } else {
        // not a header, discard byte and continue
        // (we already consumed 0x42)
      }
    } else {
      PMSSerial.read(); // discard byte until we meet 0x42
    }
  }
  return false;
}

// =============== Analog helpers ===============
void setupADC() {
  // Full-scale ~3.3V (11 dB). Use ADC1 only (pins 32-39) to avoid Wi-Fi conflicts.
  analogSetPinAttenuation(MQ131_PIN, ADC_11db);
  analogSetPinAttenuation(MQ4_PIN,   ADC_11db);
  analogReadResolution(12); // 0..4095
}

int readMvAveraged(int pin, int samples) {
  long acc = 0;
  for (int i = 0; i < samples; ++i) {
    acc += analogReadMilliVolts(pin);   // calibrated mV reading on ESP32
    delay(2);
  }
  return (int)(acc / samples);
}

// =============== Networking ===============
void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
    if (millis() - t0 > 20000) {
      Serial.println("\nRetry WiFi...");
      WiFi.disconnect(true);
      delay(500);
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      t0 = millis();
    }
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP: "); Serial.println(WiFi.localIP());
}

bool postJson(const String& url, const String& json) {
  WiFiClientSecure client;
  client.setTimeout(10000);
  client.setInsecure(); // For production: replace with client.setCACert(root_ca);

  HTTPClient http;
  if (!http.begin(client, url)) {
    Serial.println("[HTTP] begin() failed");
    return false;
  }
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(json);
  Serial.printf("[HTTP] POST -> %d\n", code);
  if (code > 0) {
    String resp = http.getString();
    Serial.println(resp);
  } else {
    Serial.printf("[HTTP] Error: %s\n", http.errorToString(code).c_str());
  }
  http.end();
  return (code > 0 && code < 400);
}

// =============== Arduino setup/loop ===============
void setup() {
  Serial.begin(115200);
  delay(300);

  // PMS5003 serial
  PMSSerial.begin(9600, SERIAL_8N1, PMS_RX, PMS_TX);
  setupADC();
  connectWiFi();

  Serial.println("Setup complete.");
}

void loop() {
  // ---- Read PMS5003 (try to get a fresh frame within ~1s) ----
  PMData pm{};
  uint32_t startWait = millis();
  bool gotPM = false;
  while (millis() - startWait < 1000) {
    if (readPMS5003(pm)) { gotPM = true; break; }
    delay(10);
  }

  // ---- Read MQ131 and MQ-4 (raw mV average) ----
  int mq131_mv = readMvAveraged(MQ131_PIN, ADC_SAMPLES);
  int mq4_mv   = readMvAveraged(MQ4_PIN,   ADC_SAMPLES);

  // ---- Build JSON payload ----
  String json = "{";

  // PMS5003 block
  if (gotPM) {
    json += "\"PMS5003\":{";
    json += "\"PM1_0\":" + String(pm.pm1_0_atm) + ",";
    json += "\"PM2_5\":" + String(pm.pm2_5_atm) + ",";
    json += "\"PM10\":"  + String(pm.pm10_atm);
    json += "},";
  } else {
    // If no frame, send zeros (or omit the key if you prefer)
    json += "\"PMS5003\":{";
    json += "\"PM1_0\":0,\"PM2_5\":0,\"PM10\":0},";
  }

  // MQ131 / MQ-4 blocks (raw mV; convert to ppm later server-side or after calibrating R0)
  json += "\"MQ131\":{";
  json += "\"mv\":" + String(mq131_mv);
  json += "},";

  json += "\"MQ4\":{";
  json += "\"mv\":" + String(mq4_mv);
  json += "}";

  json += "}";

  Serial.println("Payload:");
  Serial.println(json);

  // ---- POST to server ----
  postJson(URL, json);

  // Send every 5 seconds
  delay(5000);
}
