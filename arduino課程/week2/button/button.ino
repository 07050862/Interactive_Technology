/*https://www.arduino.cc/en/Tutorial/BuiltInExamples/Button*/
/*按按鈕撥放音樂*/
int speakerPin = 6;

int length = 49; // the number of notes
char notes[] = "geefddcdefggggeefddceggedddddefeeeeefggeefddceggc ";
int beats[] = { 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2,
                1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 3,
                1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1,1,1,2,
               1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 3,4};
int tempo = 300;
void playTone(int tone, int duration) {
  for (long i = 0; i < duration * 1000L; i += tone * 2) {
    digitalWrite(speakerPin, HIGH);
    delayMicroseconds(tone);
    digitalWrite(speakerPin, LOW);
    delayMicroseconds(tone);
  }
}

void playNote(char note, int duration) {
  char names[] = { 'c', 'd', 'e', 'f', 'g', 'a', 'b', 'C' };
  int tones[] = { 1915, 1700, 1519, 1432, 1275, 1136, 1014, 956 };

  // play the tone corresponding to the note name
  for (int i = 0; i < 8; i++) {
    if (names[i] == note) {
      playTone(tones[i], duration);
    }
  }
}
// constants won't change. They're used here to set pin numbers:
const int buttonPin = 2;     // the number of the pushbutton pin
const int ledPin =  13;      // the number of the LED pin

// variables will change:
int buttonState = 0;         // variable for reading the pushbutton status

void setup() {
  pinMode(speakerPin, OUTPUT);
  // initialize the LED pin as an output:
  pinMode(ledPin, OUTPUT);
  // initialize the pushbutton pin as an input:
  pinMode(buttonPin, INPUT);
}

void loop() {
  // read the state of the pushbutton value:
  buttonState = digitalRead(buttonPin);

  // check if the pushbutton is pressed. If it is, the buttonState is HIGH:
  if (buttonState == HIGH) {
    // turn LED on:
    digitalWrite(ledPin, HIGH);
      for (int i = 0; i < length; i++) {
    if (notes[i] == ' ') {
      delay(beats[i] * tempo); // rest
    } else {
      playNote(notes[i], beats[i] * tempo);
    }

    // pause between notes
    delay(tempo / 2); 
  }
  } else {
    // turn LED off:
    digitalWrite(ledPin, LOW);
  }
}
