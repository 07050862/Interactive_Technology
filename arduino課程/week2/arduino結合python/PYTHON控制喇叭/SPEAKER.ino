String str;
int speakerOut = 6;
void setup() {
  // initialize serial communications at 9600 bps:
  pinMode(13,OUTPUT);
  pinMode(speakerOut, OUTPUT);
  Serial.begin(9600);
}                                                                 
void loop() {
    if (Serial.available()) {
    // 讀取傳入的字串直到"\n"結尾
        str = Serial.readStringUntil('\n');
        if (str == "LED_ON") {           // 若字串值是 "LED_ON" 開燈
            digitalWrite(13, HIGH);     // 開燈
            Serial.println("LED is ON by Arduino");
        } 
        else if (str == "LED_OFF") {
            digitalWrite(13, LOW);
            Serial.println("LED is OFF by Arduino");
        }else if (str == "SPEAKER_ON") {
            tone(speakerOut,500);
            delay(1000);
            Serial.println("SPEAKER is ON by Arduino");
        }
        else if (str == "SPEAKER_OFF") {
            noTone(speakerOut);
            delay(1000);
            Serial.println("SPEAKER is OFF by Arduino");
        }
  }
}