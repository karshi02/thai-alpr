/*
 * ESP32 gate / barrier controller for Thai ALPR.
 *
 * Subscribes to MQTT:
 *   alpr/<GATE_ID>/cmd   -> "OPEN"  pulses the relay (servo/barrier/solenoid) for OPEN_MS
 *   alpr/events          -> JSON of every reading, shown on Serial + optional 16x2 LCD
 * Publishes:
 *   alpr/<GATE_ID>/status -> "online" / "opened <plate>"
 *
 * Pair with the `mqtt` output plugin:
 *   - type: mqtt
 *     host: <broker>
 *     topic: alpr/events
 *     command_topic: alpr/gate1/cmd
 * and the `allowlist` filter so only registered plates open the gate.
 *
 * Libraries: WiFi, PubSubClient, ArduinoJson (optional, for LCD text).
 */
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASS = "YOUR_PASS";
const char* MQTT_HOST = "192.168.1.10";
const int   MQTT_PORT = 1883;
const char* GATE_ID   = "gate1";

const int RELAY_PIN = 26;     // relay module IN (active HIGH)
const int LED_OK    = 2;      // onboard LED: allowed plate
const int LED_DENY  = 4;      // red LED: plate seen but not allowed
const int BUZZER    = 27;
const uint32_t OPEN_MS = 3000;

WiFiClient net;
PubSubClient mqtt(net);
char topicCmd[48], topicStatus[48];
uint32_t openedAt = 0;

void openGate(const char* why) {
  digitalWrite(RELAY_PIN, HIGH);
  digitalWrite(LED_OK, HIGH);
  tone(BUZZER, 1500, 120);
  openedAt = millis();
  char msg[96]; snprintf(msg, sizeof msg, "opened %s", why);
  mqtt.publish(topicStatus, msg);
  Serial.println(msg);
}

void onMessage(char* topic, byte* payload, unsigned int len) {
  String body; body.reserve(len);
  for (unsigned int i = 0; i < len; i++) body += (char)payload[i];

  if (strcmp(topic, topicCmd) == 0) {
    if (body == "OPEN") openGate("cmd");
    return;
  }
  // alpr/events: {"plate":"1กข 1234","province":"...","allowed":true,...}
  StaticJsonDocument<512> doc;
  if (deserializeJson(doc, body) == DeserializationError::Ok) {
    const char* plate = doc["plate"] | "?";
    const char* prov  = doc["province"] | "";
    bool allowed = doc["allowed"] | false;
    Serial.printf("plate=%s province=%s allowed=%d\n", plate, prov, allowed);
    if (!allowed) { digitalWrite(LED_DENY, HIGH); tone(BUZZER, 400, 200); delay(200); digitalWrite(LED_DENY, LOW); }
  }
}

void connectMqtt() {
  while (!mqtt.connected()) {
    if (mqtt.connect(GATE_ID)) {
      mqtt.subscribe(topicCmd);
      mqtt.subscribe("alpr/events");
      mqtt.publish(topicStatus, "online", true);
      Serial.println("mqtt connected");
    } else { delay(1000); }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT); pinMode(LED_OK, OUTPUT); pinMode(LED_DENY, OUTPUT); pinMode(BUZZER, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  snprintf(topicCmd, sizeof topicCmd, "alpr/%s/cmd", GATE_ID);
  snprintf(topicStatus, sizeof topicStatus, "alpr/%s/status", GATE_ID);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) delay(300);
  Serial.printf("IP %s\n", WiFi.localIP().toString().c_str());
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMessage);
  mqtt.setBufferSize(1024);
}

void loop() {
  if (!mqtt.connected()) connectMqtt();
  mqtt.loop();
  if (openedAt && millis() - openedAt > OPEN_MS) {
    digitalWrite(RELAY_PIN, LOW); digitalWrite(LED_OK, LOW); openedAt = 0;
  }
}
