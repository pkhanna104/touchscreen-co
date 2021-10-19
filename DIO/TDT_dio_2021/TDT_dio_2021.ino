int led = 13; 
char c; 
char d; 
char dio_data[1]; // One byte; 

// BYTE A PINOUT BYTEs :   [7, 6, 5, 4, 3, 2, 1, 0]
// BYTE A PINOUT DB25  :   [9, 21,  8, 20, 7,  19,  6, 18]
// BYTE A PINOUT ARDUINO:  [42, 29, 44, 30, 41, 33, 45, 32]

// Make sure DB25, PIN 5 = GND
// Make sure ARD GND --> DB25

int data_pins[] = {32, 45, 33, 41, 30, 44, 29, 42}; // here bit 0 is first 
int strobe = 48; // PIN 2 --> C2 

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); 
  for (int k=0; k < (sizeof(data_pins)); k += 1) {
    pinMode(data_pins[k], OUTPUT); 
  }
  pinMode(strobe, OUTPUT); 
  pinMode(led, OUTPUT); 
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() >= 1) {
    c = Serial.read(); 
    if (c=='d') {
      Serial.readBytes(dio_data, 1); 
      for (int byte_idx = 0; byte_idx < 1; byte_idx +=1) {
        
        byte data_byte = dio_data[byte_idx]; 
        
        for (int bit_idx = 0; bit_idx < 8; bit_idx +=1) {
          int pin_idx = 8*byte_idx + bit_idx; 
          byte mask = 1 << bit_idx; 
          
          if (mask & data_byte) {
            digitalWrite(data_pins[pin_idx], HIGH);
          } else {
            digitalWrite(data_pins[pin_idx], LOW);
          }
        }
      }
      digitalWrite(strobe, HIGH); 
      delay(1); 
      digitalWrite(strobe, LOW); 
    }
  }
}
