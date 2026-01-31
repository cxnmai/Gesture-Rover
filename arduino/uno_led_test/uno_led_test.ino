// Elegoo Uno R3 LED test
// Blinks the onboard "L" LED (pin 13 / LED_BUILTIN).
// Also writes to Serial so the TX LED should flicker.

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);
  Serial.write('1');
  delay(500);

  digitalWrite(LED_BUILTIN, LOW);
  Serial.write('0');
  delay(500);
}
