// Minimal ESP32 WiFi connect sketch (Arduino framework)
// Upload with Arduino IDE / arduino-cli.

#include <WiFi.h>

// Option A (recommended): define WIFI_SSID / WIFI_PASSWORD via build flags or a local secrets.h.
// Option B: edit the fallback strings below.
//
// Example secrets.h (do not commit):
//   #define WIFI_SSID "MyNetwork"
//   #define WIFI_PASSWORD "MyPassword"

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID "YOUR_SSID"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "YOUR_PASSWORD"
#endif

static const char *kSsid = WIFI_SSID;
static const char *kPassword = WIFI_PASSWORD;

static void connect_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);

  Serial.print("wifi ssid=");
  Serial.println(kSsid);
  Serial.print("wifi mac=");
  Serial.println(WiFi.macAddress());

  WiFi.begin(kSsid, kPassword);

  const unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && (millis() - start) < 15000) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("wifi ok ip=");
    Serial.println(WiFi.localIP());
    Serial.print("wifi rssi=");
    Serial.println(WiFi.RSSI());
    return;
  }

  Serial.println("wifi failed; retrying");
  WiFi.disconnect(true /*wifioff*/);
  delay(500);
}

void setup() {
  Serial.begin(115200);
  delay(200);

  connect_wifi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connect_wifi();
    return;
  }

  static unsigned long last = 0;
  if (millis() - last >= 5000) {
    last = millis();
    Serial.print("ip ");
    Serial.print(WiFi.localIP());
    Serial.print(" rssi ");
    Serial.println(WiFi.RSSI());
  }
}
