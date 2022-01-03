/*兩個同學合力彈完一首曲子*/
#define c 3830 // 261 Hz
#define d 3400 // 294 Hz
#define e 3038 // 329 Hz
#define f 2864 // 349 Hz
#define g 2550 // 392 Hz
#define a 2272 // 440 Hz
#define b 2028 // 493 Hz
#define C 1912 // 523 Hz

const int buttonPin1 = 2;     // the number of the pushbutton pin
const int buttonPin2 = 3;     // the number of the pushbutton pin
const int buttonPin3 = 4;     // the number of the pushbutton pin
const int buttonPin4 = 8;     // the number of the pushbutton pin
const int ledPin =  13;      // the number of the LED pin

// variables will change:
int buttonState1 = 0;         // variable for reading the pushbutton status
int buttonState2 = 0;         // variable for reading the pushbutton status
int buttonState3 = 0;         // variable for reading the pushbutton status
int buttonState4 = 0;         // variable for reading the pushbutton status
int speakerOut = 6;
int tone1;
int tone2;
int tone3;
int tone4;
void setup() {
pinMode(speakerOut, OUTPUT);
pinMode(ledPin, OUTPUT);
  
  pinMode(buttonPin1, INPUT);
  pinMode(buttonPin2, INPUT);
  pinMode(buttonPin3, INPUT);
  pinMode(buttonPin4, INPUT);
}

void loop() {
 buttonState1 = digitalRead(buttonPin1);
 buttonState2 = digitalRead(buttonPin2);
 buttonState3 = digitalRead(buttonPin3);
 buttonState4 = digitalRead(buttonPin4);

  if (buttonState1 == HIGH) {
   tone1 = 2550;
    digitalWrite(ledPin, HIGH);
    digitalWrite(speakerOut,HIGH);
    delayMicroseconds(tone1 / 2);
    digitalWrite(speakerOut, LOW);
    delayMicroseconds(tone1 / 2);
  } 
   else if (buttonState2 == HIGH) {
   tone2 = 2272;
    digitalWrite(ledPin, HIGH);
    digitalWrite(speakerOut,HIGH);
    delayMicroseconds(tone2 / 2);
    digitalWrite(speakerOut, LOW);
    delayMicroseconds(tone2 / 2);
  } 
    else if (buttonState3 == HIGH) {
   tone3 = 2028;
    digitalWrite(ledPin, HIGH);
    digitalWrite(speakerOut,HIGH);
    delayMicroseconds(tone3 / 2);
    digitalWrite(speakerOut, LOW);
    delayMicroseconds(tone3 / 2);
  } 
      else if (buttonState4 == HIGH) {
   tone4 = 1912;
    digitalWrite(ledPin, HIGH);
    digitalWrite(speakerOut,HIGH);
    delayMicroseconds(tone4 / 2);
    digitalWrite(speakerOut, LOW);
    delayMicroseconds(tone4 / 2);
  } 
  else
  {digitalWrite(ledPin, LOW);
    }



}
