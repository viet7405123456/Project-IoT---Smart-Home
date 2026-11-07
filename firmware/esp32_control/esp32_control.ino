/*
  ESP32 device controller: receives HTTP commands from the PC and drives
  relays for light, fan and door lock.

  Required library (install via Arduino Library Manager): "ESPAsyncWebServer"
  depends on "AsyncTCP". If you prefer not to install extra libraries, the
  simpler WebServer.h version below also works (default, no extra install).

  Wiring (adjust pins to your relay module):
    LIGHT_PIN  -> relay channel 1 -> lamp (through mains-rated relay)
    FAN_PIN    -> relay channel 2 -> fan
    LOCK_PIN   -> relay channel 3 -> electric door lock

  SAFETY: if your relay switches mains (220V/110V) voltage, use a relay
  module rated for it, keep high-voltage wiring separated from the ESP32,
  and have it checked by someone experienced with mains wiring. For
  prototyping, test with low-voltage LEDs/DC fan/solenoid lock first.
*/

#include <WiFi.h>
#include <WebServer.h>

const char *WIFI_SSID = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

#define LIGHT_PIN 16
#define FAN_PIN 17
#define LOCK_PIN 18

// Most relay modules are active-LOW (LOW = relay energized). Flip if yours is active-HIGH.
#define RELAY_ON LOW
#define RELAY_OFF HIGH

WebServer server(80);

bool lightOn = false;
bool fanOn = false;
bool doorLocked = true;

void setRelay(int pin, bool on) {
  digitalWrite(pin, on ? RELAY_ON : RELAY_OFF);
}

void sendJsonStatus() {
  String json = "{";
  json += "\"light\":\"" + String(lightOn ? "on" : "off") + "\",";
  json += "\"fan\":\"" + String(fanOn ? "on" : "off") + "\",";
  json += "\"door\":\"" + String(doorLocked ? "locked" : "unlocked") + "\"";
  json += "}";
  server.send(200, "application/json", json);
}

void handleLight() {
  if (server.hasArg("state")) {
    String state = server.arg("state");
    lightOn = (state == "on");
    setRelay(LIGHT_PIN, lightOn);
  }
  sendJsonStatus();
}

void handleFan() {
  if (server.hasArg("state")) {
    String state = server.arg("state");
    fanOn = (state == "on");
    setRelay(FAN_PIN, fanOn);
  }
  sendJsonStatus();
}

void handleDoor() {
  if (server.hasArg("state")) {
    String state = server.arg("state");
    doorLocked = (state == "lock");
    setRelay(LOCK_PIN, !doorLocked); // energize relay to unlock, adjust to your lock type
  }
  sendJsonStatus();
}

void handleStatus() {
  sendJsonStatus();
}

void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}

void setup() {
  Serial.begin(115200);

  pinMode(LIGHT_PIN, OUTPUT);
  pinMode(FAN_PIN, OUTPUT);
  pinMode(LOCK_PIN, OUTPUT);
  setRelay(LIGHT_PIN, false);
  setRelay(FAN_PIN, false);
  setRelay(LOCK_PIN, false); // locked by default

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Control server ready at: http://");
  Serial.println(WiFi.localIP());

  server.on("/light", handleLight);
  server.on("/fan", handleFan);
  server.on("/door", handleDoor);
  server.on("/status", handleStatus);
  server.onNotFound(handleNotFound);
  server.begin();
}

void loop() {
  server.handleClient();
}
