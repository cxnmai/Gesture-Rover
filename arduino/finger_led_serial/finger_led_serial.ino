// Finger -> Uno LEDs (serial controlled)
// Receives a newline-terminated integer over Serial:
//   0 = no LEDs
//   1 = LED_BUILTIN (L) on
//   2 = LED_BUILTIN on + D4 HIGH
//
// Note: The Uno only has one user-controllable onboard LED (L). The TX LED is
// driven by serial transmit activity; we approximate a second LED by sending
// periodic bytes when level==2.

static const int PIN_D4 = 4;

static int level = 0;
static unsigned long last_rx_ms = 0;
static unsigned long last_tx_ms = 0;

static bool readLevelLine(int *out) {
  static char buf[16];
  static byte n = 0;

  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (n == 0) {
        continue;
      }
      buf[n] = '\0';
      n = 0;
      int v = atoi(buf);
      *out = v;
      return true;
    }

    if (n < (sizeof(buf) - 1)) {
      buf[n++] = c;
    } else {
      // Overflow; reset line.
      n = 0;
    }
  }
  return false;
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  pinMode(PIN_D4, OUTPUT);
  digitalWrite(PIN_D4, LOW);

  Serial.begin(115200);
  last_rx_ms = millis();
  last_tx_ms = millis();
}

void loop() {
  int v;
  if (readLevelLine(&v)) {
    if (v < 0) v = 0;
    if (v > 2) v = 2;
    level = v;
    last_rx_ms = millis();
  }

  // Fail-safe: if the PC stops sending, turn everything off.
  if (millis() - last_rx_ms > 1200) {
    level = 0;
  }

  digitalWrite(LED_BUILTIN, (level >= 1) ? HIGH : LOW);
  digitalWrite(PIN_D4, (level >= 2) ? HIGH : LOW);

  // Keep TX LED lit (approximately) by transmitting periodically.
  if (level >= 2) {
    unsigned long now = millis();
    if (now - last_tx_ms >= 20) {
      Serial.write((uint8_t)0x55);
      last_tx_ms = now;
    }
  }
}
