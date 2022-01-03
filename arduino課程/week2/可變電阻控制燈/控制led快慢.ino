int BASE = 10; // 從腳位 10 開始
int NUM = 4;   // LED 的總數
int analogPin = A0;
int val = 0;      // 存放 ADC 數值

void setup()
{
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
     digitalWrite(i, LOW);
     val = analogRead(analogPin);
     delay(val);
   }
   for (int i = BASE; i < BASE + NUM; i ++) 
   {
     digitalWrite(i, HIGH);
     val = analogRead(analogPin);
     delay(val);
   }  
}