int BASE = 10; // 從腳位 10 開始
int NUM = 4;   // LED 的總數
int analogPin = A0;  //可變電阻
int val = 0;      // 存放 ADC 數值
int photocellPin = A2; // 光敏電阻 (photocell) 接在 anallog pin 2
int photocellVal = 0; // photocell variable
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
void setup()
{
   Serial.begin(9600);
   pinMode(speakerPin, OUTPUT);
    // 用迴圈設定腳位
   for (int i = BASE; i < BASE + NUM; i ++) 
   {
     pinMode(i, OUTPUT);
   }
   // 設定 A0 為輸入
   pinMode(analogPin, INPUT);
}

void loop()
{
   for (int i = BASE; i < BASE + NUM; i ++) 
   {
     photocellVal = analogRead(photocellPin);
     Serial.println(photocellVal);
     digitalWrite(i, HIGH);
     val = analogRead(analogPin);
     delay(val);
     digitalWrite(i, LOW);
     if(photocellVal<20)
     {    while(photocellVal<20)
          { digitalWrite(i, HIGH);
               Serial.println("press");
              photocellVal = analogRead(photocellPin);
              if( i == BASE+1){ 
                  Serial.println("bingo!");
                  //digitalWrite(i, HIGH);
                  for (int jj = 0; jj < length; jj++) {
                      if (notes[jj] == ' ') {
                          delay(beats[jj] * tempo); // rest
                      } 
                      else {
                          playNote(notes[jj], beats[jj] * tempo);
                      }
                  delay(tempo / 2); 
                  }
                  
                  break;
             }
             else
             {
                 Serial.println("NO");
                 //digitalWrite(i, HIGH);
                 delay(100);
             }    
         } 
         digitalWrite(i, LOW);
     }  
     
   }  
}