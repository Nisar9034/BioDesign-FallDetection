/*
 * ESP32 + MPU6050 real-time fall detection (1D CNN, TFLite Micro, full INT8)
 *
 * Model (from notebook / model_data.h):
 *   input  : int8 [1, 100, 6]  (2 s @ 50 Hz; ax ay az [g], gx gy gz [deg/s])
 *   output : int8 [1, 1]       sigmoid -> P(fall); class 1 = FALL, 0 = ADL
 *   norm   : per-channel z-score  (x - NORM_MEAN) / NORM_STD  (in model_data.h)
 *   quant  : q = round(x_norm / INPUT_SCALE) + INPUT_ZERO_POINT
 *            p = (q_out - OUTPUT_ZERO_POINT) * OUTPUT_SCALE
 *
 * Library: Chirale_TensorFlowLite (Arduino Library Manager)
 * Keep model_data.h in the same folder as this .ino.
 */
#include <Wire.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>

// ------------------------------------------------------------- Telegram / WiFi credentials
const char* WIFI_SSID = "xxxxxx";
const char* WIFI_PASS = "xxxxxx";
const char* BOT_TOKEN = "xxxxxxxxxx:xxxxxxxxxxxx";
const char* CHAT_ID   = "xxxxxxxxxx";

// ------------------------------------------------------------- pins
#define PIN_SDA 21
#define PIN_SCL 22
#define PIN_LED 26      // LED + 220R to GND
#define PIN_BTN 27      // button to GND (INPUT_PULLUP)

// ------------------------------------------------------------- config
#define MPU_ADDR        0x68
#define WIN             100
#define CH              6
#define SAMPLE_US       20000UL   // 50 Hz
#define INFER_STRIDE    25        // infer every 0.5 s (same hop as ADL training windows)
#define FALL_THRESHOLD  0.85f     // same threshold used in notebook evaluation
#define CONSEC_REQUIRED 3         // consecutive positive windows to confirm (1 = fastest)
#define BLINK_MS        250
#define DEBOUNCE_MS     30
#define PRINT_SAMPLE_EVERY 10     // print transformed sample every N samples (0 = never)

// MPU6050 full-scale: +-16 g, +-2000 deg/s (matches SisFall ADXL345/ITG3200 ranges)
#define ACC_LSB_PER_G    2048.0f
#define GYR_LSB_PER_DPS  16.4f

// ------------------------------------------------------------- state
enum State { MONITORING, ALERT };
State state = MONITORING;

static float win[WIN][CH];
int head = 0;              // next write index (= oldest sample once full)
int filled = 0;
int sinceInfer = 0;
int consec = 0;
unsigned long sampleCount = 0;

uint32_t nextSampleUs = 0;
uint32_t lastBlink = 0;
bool ledOn = false;

// debounce
bool btnStable = HIGH, btnLastRead = HIGH;
uint32_t btnChangeMs = 0;

// TFLM
constexpr int ARENA_SIZE = 32 * 1024;
alignas(16) static uint8_t arena[ARENA_SIZE];
tflite::AllOpsResolver resolver;
tflite::MicroInterpreter* interp = nullptr;
TfLiteTensor* inT = nullptr;
TfLiteTensor* outT = nullptr;

// ------------------------------------------------------------- MPU6050
static void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

bool mpuInit() {
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);
  mpuWrite(0x6B, 0x00);   // wake up
  delay(100);
  mpuWrite(0x1A, 0x03);   // DLPF ~44 Hz (suits 50 Hz sampling)
  mpuWrite(0x1B, 0x18);   // gyro  +-2000 dps
  mpuWrite(0x1C, 0x18);   // accel +-16 g
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x75);       // WHO_AM_I
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
  if (!Wire.available()) return false;
  uint8_t id = Wire.read();
  Serial.printf("MPU WHO_AM_I = 0x%02X\n", id);
  return id == 0x68 || id == 0x72 || id == 0x70;
}

bool mpuRead(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14) != 14) return false;
  int16_t r[7];
  for (int i = 0; i < 7; i++) {
    uint8_t h = Wire.read(), l = Wire.read();
    r[i] = (int16_t)((h << 8) | l);
  }
  ax = r[0] / ACC_LSB_PER_G;  ay = r[1] / ACC_LSB_PER_G;  az = r[2] / ACC_LSB_PER_G;
  gx = r[4] / GYR_LSB_PER_DPS; gy = r[5] / GYR_LSB_PER_DPS; gz = r[6] / GYR_LSB_PER_DPS;
  return true;   // r[3] = temperature, ignored
}

// ------------------------------------------------------------- inference
float runInference() {
  int8_t* in = inT->data.int8;
  const float s = inT->params.scale;
  const int zp = inT->params.zero_point;
  int k = 0;
  for (int i = 0; i < WIN; i++) {
    int idx = (head + i) % WIN;               // oldest -> newest
    for (int c = 0; c < CH; c++) {
      float z = (win[idx][c] - NORM_MEAN[c]) / NORM_STD[c];
      int q = (int)lroundf(z / s) + zp;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      in[k++] = (int8_t)q;
    }
  }
  uint32_t t0 = micros();
  if (interp->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return -1.0f;
  }
  uint32_t dt = micros() - t0;
  float p = (outT->data.int8[0] - outT->params.zero_point) * outT->params.scale;
  Serial.printf("[INFER] P(fall)=%.3f  (%lu us)\n", p, (unsigned long)dt);
  return p;
}

// ------------------------------------------------------------- Telegram (runs on core 0)
// The HTTPS request takes 1-3 s, so it must NOT run in loop() (that would break
// the 50 Hz sampling). loop() just drops a message in a queue; this task sends it.
struct TgMsg { char text[96]; };
QueueHandle_t tgQueue = nullptr;

static String urlEncode(const char* s) {
  String out;
  const char* hex = "0123456789ABCDEF";
  for (; *s; s++) {
    uint8_t c = (uint8_t)*s;
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') out += (char)c;
    else { out += '%'; out += hex[c >> 4]; out += hex[c & 15]; }
  }
  return out;
}

static bool tgSend(const char* text) {
  if (WiFi.status() != WL_CONNECTED) return false;
  WiFiClientSecure client;
  client.setInsecure();               // no certificate check (simple; see notes)
  HTTPClient https;
  https.setTimeout(8000);
  String url = String("https://api.telegram.org/bot") + BOT_TOKEN + "/sendMessage";
  if (!https.begin(client, url)) return false;
  https.addHeader("Content-Type", "application/x-www-form-urlencoded");
  String body = String("chat_id=") + CHAT_ID + "&text=" + urlEncode(text);
  int code = https.POST(body);
  https.end();
  Serial.printf("[TG] HTTP %d\n", code);
  return code == 200;
}

static void tgTask(void*) {
  TgMsg m;
  uint32_t lastWifiTry = 0;
  for (;;) {
    if (WiFi.status() != WL_CONNECTED && millis() - lastWifiTry > 10000) {
      lastWifiTry = millis();
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASS);
    }
    if (xQueueReceive(tgQueue, &m, pdMS_TO_TICKS(500)) == pdTRUE) {
      bool ok = false;
      for (int attempt = 0; attempt < 5 && !ok; attempt++) {
        ok = tgSend(m.text);
        if (!ok) vTaskDelay(pdMS_TO_TICKS(2000));
      }
      Serial.println(ok ? "[TG] sent" : "[TG] FAILED after retries");
    }
  }
}

void tgInit() {
  tgQueue = xQueueCreate(4, sizeof(TgMsg));
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);   // non-blocking: monitoring starts regardless
  xTaskCreatePinnedToCore(tgTask, "tg", 8192, nullptr, 1, nullptr, 0);
}

void tgNotify(const char* text) {     // non-blocking, safe to call from loop()
  TgMsg m;
  strlcpy(m.text, text, sizeof(m.text));
  xQueueSend(tgQueue, &m, 0);
}

// ------------------------------------------------------------- helpers
void resetWindow() {
  head = 0; filled = 0; sinceInfer = 0; consec = 0;
}

void handleButton() {
  bool r = digitalRead(PIN_BTN);
  uint32_t now = millis();
  if (r != btnLastRead) { btnLastRead = r; btnChangeMs = now; }
  if ((now - btnChangeMs) > DEBOUNCE_MS && r != btnStable) {
    btnStable = r;
    if (btnStable == LOW && state == ALERT) {       // pressed
      state = MONITORING;
      digitalWrite(PIN_LED, LOW);
      ledOn = false;
      resetWindow();   // discard window that still contains the fall
      Serial.println("[STATE] Alert acknowledged -> MONITORING");
      tgNotify("Fall alert acknowledged by user.");
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  pinMode(PIN_BTN, INPUT_PULLUP);
  tgInit();

  if (!mpuInit()) {
    Serial.println("MPU6050 not found - check wiring");
    while (true) { digitalWrite(PIN_LED, !digitalRead(PIN_LED)); delay(100); }
  }

  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch");
    while (true) delay(1000);
  }
  static tflite::MicroInterpreter staticInterp(model, resolver, arena, ARENA_SIZE);
  interp = &staticInterp;
  if (interp->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed - increase ARENA_SIZE");
    while (true) delay(1000);
  }
  inT = interp->input(0);
  outT = interp->output(0);
  Serial.printf("Input dims: %d x %d x %d, type=%d, scale=%.6f zp=%d\n",
                inT->dims->data[0], inT->dims->data[1], inT->dims->data[2],
                inT->type, inT->params.scale, inT->params.zero_point);
  Serial.printf("Arena used: %u bytes\n", (unsigned)interp->arena_used_bytes());
  Serial.println("[STATE] MONITORING (filling 2 s window...)");
  nextSampleUs = micros() + SAMPLE_US;
}

void loop() {
  handleButton();

  // blink in alert (non-blocking)
  if (state == ALERT && millis() - lastBlink >= BLINK_MS) {
    lastBlink = millis();
    ledOn = !ledOn;
    digitalWrite(PIN_LED, ledOn);
  }

  // 50 Hz sampling
  if ((int32_t)(micros() - nextSampleUs) < 0) return;
  nextSampleUs += SAMPLE_US;
  if ((int32_t)(micros() - nextSampleUs) > 5 * (int32_t)SAMPLE_US) {
    nextSampleUs = micros() + SAMPLE_US;           // fell behind, resync
    Serial.println("[WARN] sampling overrun");
  }

  float ax, ay, az, gx, gy, gz;
  if (!mpuRead(ax, ay, az, gx, gy, gz)) {
    Serial.println("[WARN] MPU read failed");
    return;
  }

  // sensor frame -> SisFall frame
  float f[CH] = { ay, -ax, az, gy, -gx, gz };

  for (int c = 0; c < CH; c++) win[head][c] = f[c];
  head = (head + 1) % WIN;
  if (filled < WIN) filled++;
  sinceInfer++;
  sampleCount++;

  if (PRINT_SAMPLE_EVERY && sampleCount % PRINT_SAMPLE_EVERY == 0) {
    Serial.printf("[%s] ax=%.3f ay=%.3f az=%.3f | gx=%.1f gy=%.1f gz=%.1f\n",
                  state == ALERT ? "ALERT" : "MON", f[0], f[1], f[2], f[3], f[4], f[5]);
  }

  if (state == MONITORING && filled >= WIN && sinceInfer >= INFER_STRIDE) {
    sinceInfer = 0;
    float p = runInference();
    if (p >= FALL_THRESHOLD) consec++; else consec = 0;
    if (consec >= CONSEC_REQUIRED) {
      state = ALERT;
      lastBlink = millis();
      ledOn = true;
      digitalWrite(PIN_LED, HIGH);
      Serial.printf("[STATE] FALL DETECTED (P=%.3f) -> ALERT, press button\n", p);
      char msg[96];
      snprintf(msg, sizeof(msg), "FALL DETECTED! Confidence %.0f%%. Please check on the wearer.", p * 100.0f);
      tgNotify(msg);
    }
  }
}
