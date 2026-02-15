#include <WiFi.h>

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID "YOUR_SSID"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "YOUR_PASSWORD"
#endif

const char* ssid = WIFI_SSID;
const char* password = WIFI_PASSWORD;
WiFiServer server(12345);
WiFiClient client;

#if defined(LED_BUILTIN)
const int LED_PIN = LED_BUILTIN;
#else
const int LED_PIN = 2;
#endif

const size_t FRAME_SIZE = 5;  // <Bhh> = reversing_u8, degree_tenths_i16, hand_distance_px_i16
uint8_t frame_buf[FRAME_SIZE];
size_t frame_len = 0;

static int16_t read_i16_le(const uint8_t* ptr) {
  uint16_t raw = (uint16_t)ptr[0] | ((uint16_t)ptr[1] << 8);
  return (int16_t)raw;
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }

  server.begin();
  Serial.println("\nCommand Link Ready: " + WiFi.localIP().toString());
}

void loop() {
  if (!client || !client.connected()) {
    client = server.available();
    frame_len = 0;
  }

  if (!client || !client.connected()) {
    delay(5);
    return;
  }

  while (client.available() > 0) {
    int b = client.read();
    if (b < 0) {
      break;
    }

    frame_buf[frame_len++] = (uint8_t)b;

    if (frame_len == FRAME_SIZE) {
      frame_len = 0;

      bool reversing = frame_buf[0] != 0;
      int16_t degree_tenths = read_i16_le(&frame_buf[1]);
      int16_t hand_distance_px = read_i16_le(&frame_buf[3]);

      digitalWrite(LED_PIN, reversing ? HIGH : LOW);

      Serial.print("reversing=");
      Serial.print(reversing ? "T" : "F");
      Serial.print(" degree=");
      Serial.print((float)degree_tenths / 10.0f, 1);
      Serial.print(" hand_distance=");
      Serial.println(hand_distance_px);
    }
  }
}
