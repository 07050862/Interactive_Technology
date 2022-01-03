const int analogInPin = A0;  // Analog input pin that the potentiometer is attached to

int sensorValue = 0;        // value read from the pot
int outputValue = 0;        // value output to the PWM (analog out)
String str;

void setup() {
  // initialize serial communications at 9600 bps:
  pinMode(13,OUTPUT);
  Serial.begin(9600);
}

void loop() {

    // read the analog in value:
    sensorValue = analogRead(analogInPin);
    Serial.println(sensorValue);
    
    if (Serial.available()) {

    // 讀取傳入的字串直到"\n"結尾
        str = Serial.readStringUntil('\n');
 
        if (str == "LED_ON") {           // 若字串值是 "LED_ON" 開燈
            digitalWrite(13, HIGH);     // 開燈
        } else if (str == "LED_OFF") {
            digitalWrite(13, LOW);
        }
     }
     
     delay(500);
}
