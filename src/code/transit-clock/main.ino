// Transit Clock — sample sketch placeholder.
// Replace with your real firmware when ready.

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* WIFI_SSID = "your-ssid";
const char* WIFI_PASS = "your-pass";
const char* API_URL   = "https://api.example.com/transit/next";

void setup() {
  Serial.begin(115200);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print('.');
  }
  Serial.println("\nWiFi up");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(API_URL);
    int code = http.GET();
    if (code == 200) {
      StaticJsonDocument<512> doc;
      deserializeJson(doc, http.getString());
      int eta = doc["minutes"] | -1;
      Serial.printf("Next arrival in %d min\n", eta);
    }
    http.end();
  }
  delay(30000);
}
