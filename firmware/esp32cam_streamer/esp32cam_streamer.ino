/*
 * ESP32-CAM (AI-Thinker) firmware for Thai ALPR.
 *
 * Two ways to feed the server, both enabled:
 *   PULL : server GETs http://<this-ip>/capture       (plugin: esp32cam, mode snapshot)
 *   PUSH : board POSTs a JPEG to http://<server>:8000/ingest/<DEVICE_ID>
 *          every PUSH_INTERVAL_MS, or when a PIR/IR sensor on TRIGGER_PIN goes HIGH
 *          (plugin: http_push - created automatically by `thai_alpr serve`)
 *
 * Board: "AI Thinker ESP32-CAM", partition "Huge APP".
 * Libraries: esp_camera (built in), WiFi, HTTPClient, WebServer.
 */
#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>

// ---------------- config ----------------
const char* WIFI_SSID   = "YOUR_WIFI";
const char* WIFI_PASS   = "YOUR_PASS";
const char* DEVICE_ID   = "gate1";
const char* SERVER_URL  = "http://192.168.1.10:8000";   // thai_alpr serve host
const bool  PUSH_ENABLED = true;
const uint32_t PUSH_INTERVAL_MS = 1500;   // 0 = only on trigger
const int   TRIGGER_PIN = 13;             // PIR / IR beam, -1 to disable
// ----------------------------------------

// AI-Thinker pin map
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22
#define FLASH_LED       4

WebServer server(80);
uint32_t lastPush = 0;

bool initCamera() {
  camera_config_t c;
  c.ledc_channel = LEDC_CHANNEL_0; c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM; c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sscb_sda = SIOD_GPIO_NUM; c.pin_sscb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000; c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size = FRAMESIZE_SVGA;   // 800x600 - good balance for plates at 2-5 m
  c.jpeg_quality = 10;             // lower = better
  c.fb_count = psramFound() ? 2 : 1;
  c.grab_mode = CAMERA_GRAB_LATEST;
  if (esp_camera_init(&c) != ESP_OK) return false;
  sensor_t* s = esp_camera_sensor_get();
  s->set_brightness(s, 0); s->set_contrast(s, 1); s->set_saturation(s, 0);
  s->set_whitebal(s, 1); s->set_exposure_ctrl(s, 1); s->set_gain_ctrl(s, 1);
  return true;
}

void handleCapture() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) { server.send(500, "text/plain", "capture failed"); return; }
  server.sendHeader("X-Device-Id", DEVICE_ID);
  server.setContentLength(fb->len);
  server.send(200, "image/jpeg", "");
  server.client().write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

void handleRoot() {
  server.send(200, "text/plain", String("ESP32-CAM ") + DEVICE_ID + "\nGET /capture -> jpeg\n");
}

bool pushFrame() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) return false;
  HTTPClient http;
  String url = String(SERVER_URL) + "/ingest/" + DEVICE_ID;
  http.begin(url);
  String boundary = "----esp32cam";
  String head = "--" + boundary + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"f.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
  String tail = "\r\n--" + boundary + "--\r\n";
  size_t total = head.length() + fb->len + tail.length();
  uint8_t* body = (uint8_t*)malloc(total);
  if (!body) { esp_camera_fb_return(fb); http.end(); return false; }
  memcpy(body, head.c_str(), head.length());
  memcpy(body + head.length(), fb->buf, fb->len);
  memcpy(body + head.length() + fb->len, tail.c_str(), tail.length());
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
  int code = http.POST(body, total);
  free(body);
  esp_camera_fb_return(fb);
  http.end();
  return code == 200;
}

void setup() {
  Serial.begin(115200);
  pinMode(FLASH_LED, OUTPUT); digitalWrite(FLASH_LED, LOW);
  if (TRIGGER_PIN >= 0) pinMode(TRIGGER_PIN, INPUT);
  if (!initCamera()) { Serial.println("camera init failed"); ESP.restart(); }

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.printf("\nIP: %s\n", WiFi.localIP().toString().c_str());

  server.on("/", handleRoot);
  server.on("/capture", handleCapture);
  server.begin();
}

void loop() {
  server.handleClient();
  if (!PUSH_ENABLED) return;
  bool trig = TRIGGER_PIN >= 0 && digitalRead(TRIGGER_PIN) == HIGH;
  bool due  = PUSH_INTERVAL_MS > 0 && millis() - lastPush >= PUSH_INTERVAL_MS;
  if (trig || due) {
    lastPush = millis();
    bool ok = pushFrame();
    Serial.printf("push %s\n", ok ? "ok" : "fail");
    if (trig) delay(300);
  }
}
