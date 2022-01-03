#define LED_BUILTIN 13 
int led[]={7,9,10,11,12};
int NUM =5;
int counttime = 5;
void setup() {
  for(int ii=0;ii<NUM;ii++)
  {
    pinMode(led[ii],OUTPUT);
    digitalWrite(led[ii],LOW);
  }
}
void loop() {
  for(int jj=0;jj<counttime;jj++){
      for(int ii=0;ii<NUM;ii++)
      {
          digitalWrite(led[ii], HIGH);                            
      }              
      delay(1000);
      for(int ii=0;ii<NUM;ii++)
      {
          digitalWrite(led[ii], LOW);                            
      }  
      delay(1000);
  }
  
  for(int jj=0;jj<counttime;jj++){
      for(int ii=0;ii<NUM;ii++)
      {
          digitalWrite(led[ii], HIGH);  
          delay(100);                    
          digitalWrite(led[ii], LOW);                        
      }   
  }
  for(int jj=0;jj<counttime;jj++){
      for(int ii=0;ii<NUM;ii++)
      {
          digitalWrite(led[ii], HIGH);  
          delay(100);                                    
      }    
    for(int ii=NUM-1;ii>=0;ii--)
      {
          digitalWrite(led[ii], LOW); 

          delay(100); 
      }   
  }
}