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

#if defined(LED_BUILTIN)
const int LED_PIN = LED_BUILTIN;
#else
const int LED_PIN = 2;
#endif

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
  static unsigned long last_toggle_ms = 0;
  static bool led_on = false;
  unsigned long now = millis();

  if (now - last_toggle_ms >= 500) {
    led_on = !led_on;
    digitalWrite(LED_PIN, led_on ? HIGH : LOW);
    last_toggle_ms = now;
  }

  WiFiClient client = server.available();
  if (client && client.connected() && client.available()) {
    (void)client.read();
  }
}
