// Arduino: 22 --> 46
// Pinout: 1 --> 25

// BYTE A PINOUT BYTEs : [7, 6, 5, 4, 3, 2, 1, 0]
// BYTE A PINOUT DB25  : [9, 21, 8, 20, 7, 19, 6, 18]
// BYTE A PINOUT ARDUINO: [30, 42, 29, 41, 28, 40, 27, 39]

// Make sure DB25, PIN 5 = GND
// Make sure ARD GND --> DB25

int led = 13;
char c;
char d;
char dio_data[1]; // One byte
int data_pins[] = {30, 42, 29, 41, 28, 40, 27, 39};

void setup () {
  Serial.begin(115200);
  
  // initialize the output pins
  for (int k = 0; k < (sizeof(data_pins)/sizeof(int)); k += 1) {
    pinMode(data_pins[k], OUTPUT);
  }
  pinMode(led, OUTPUT);  
}


void loop() {
  if (Serial.available() >= 1) {
    c = Serial.read();
    
    // Digital data
    if (c == 'd') {
      handle_word();
    }  
  }
}

void handle_word() { 
  Serial.readBytes(dio_data, 1);
  char d1 = dio_data[0];
  
    // set all the data bits
    for (int byte_idx = 0; byte_idx < 1; byte_idx += 1) {
      byte data_byte = dio_data[byte_idx];
      for (int bit_idx = 0; bit_idx < 8; bit_idx += 1) {
        int pin_idx = 8*byte_idx + bit_idx;
        byte mask = 1 << bit_idx;
        if (mask & data_byte) {
          digitalWrite(data_pins[pin_idx], HIGH);
        } else {
          digitalWrite(data_pins[pin_idx], LOW);         
        }
      }
    }  

  //digitalWrite(rstart, HIGH);

  // Write on strobe only: 
  //digitalWrite(strobe, HIGH);
  //delay(0.5);
  //digitalWrite(strobe, LOW);  
}
