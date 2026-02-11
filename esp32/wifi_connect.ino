#include <WiFi.h>

const char* ssid = "Chinmay iPhone;
const char* password = "password321";
WiFiServer server(12345);

const int LED_PIN = 4;

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
  WiFiClient client = server.available();
  if (client) {
    while (client.connected()) {
      if (client.available()) {
        char val = client.read();
        int fingers = val - '0'; // Convert char digit to int
        
        if (fingers > 0) {
          digitalWrite(LED_PIN, HIGH);
        } else {
          digitalWrite(LED_PIN, LOW);
        }
        Serial.printf("Fingers detected: %d\n", fingers);
      }
    }
  }
}