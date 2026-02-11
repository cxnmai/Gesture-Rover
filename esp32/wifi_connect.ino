#include <WiFi.h>
#include <WiFiUdp.h>

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID "Chinmay iPhone"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "password"
#endif

static const char *kSsid = WIFI_SSID;
static const char *kPassword = WIFI_PASSWORD;
static const uint16_t kUdpPort = 4210;

WiFiUDP udp;

static void connect_wifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);

  Serial.print("wifi ssid=");
  Serial.println(kSsid);
  Serial.print("wifi mac=");
  Serial.println(WiFi.macAddress());

  WiFi.begin(kSsid, kPassword);

  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print('.');
  }
  Serial.println();

  Serial.print("wifi ok ip=");
  Serial.println(WiFi.localIP());
  Serial.print("wifi rssi=");
  Serial.println(WiFi.RSSI());
}

void setup() {
  Serial.begin(115200);
  delay(200);

  connect_wifi();
  udp.begin(kUdpPort);
  Serial.print("udp listen port=");
  Serial.println(kUdpPort);
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    connect_wifi();
    udp.begin(kUdpPort);
    return;
  }

  int packetSize = udp.parsePacket();
  if (packetSize <= 0) {
    delay(5);
    return;
  }

  char buffer[256];
  int n = udp.read(buffer, sizeof(buffer) - 1);
  if (n < 0) {
    return;
  }
  buffer[n] = '\0';

  Serial.print("rx ");
  Serial.print(udp.remoteIP());
  Serial.print(":");
  Serial.print(udp.remotePort());
  Serial.print(" ");
  Serial.println(buffer);

  udp.beginPacket(udp.remoteIP(), udp.remotePort());
  udp.print("ACK:");
  udp.print(buffer);
  udp.endPacket();
}
