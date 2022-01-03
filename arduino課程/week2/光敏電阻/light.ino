/*
光敏電阻檢測  暗->  開燈
           一般-> 無動作
             亮-> 喇叭叫
https://blog.jmaker.com.tw/arduino-photoresistor/
*/
int photocellPin = A2; // 光敏電阻 (photocell) 接在 anallog pin 2
int photocellVal = 0; // photocell variable
const int ledPin =  13;      // the number of the LED pin
int speakerOut = 6;
int tone1;
void setup() {
  pinMode(speakerOut, OUTPUT);
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
 
  photocellVal = analogRead(photocellPin);
  Serial.println(photocellVal);  
  if(photocellVal<20)
  {
    digitalWrite(ledPin, HIGH);
  }
  else if  ((photocellVal>20) and (photocellVal<80))
  {
    digitalWrite(ledPin, LOW);
    
  }
  else if(photocellVal>80)
  {
    for(int k=0;k<=99;k++)
    {
    tone1 = 2550;
    
    digitalWrite(speakerOut,HIGH);
    delayMicroseconds(tone1 / 2);
    digitalWrite(speakerOut, LOW);
    delayMicroseconds(tone1 / 2);
    
    } 
  }
  delay(100);    
  
}
