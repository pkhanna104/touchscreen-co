char c; 
int record_trig = 3;
int strobe = 3;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); 
  pinMode(strobe, OUTPUT); 
  pinMode(record_trig, OUTPUT); 
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() >= 1) {
    c = Serial.read(); 
    if (c=='t') {
      digitalWrite(strobe, HIGH); 
      delay(1); 
      digitalWrite(strobe, LOW); 
    } else if (c=='s') {
      digitalWrite(record_trig, HIGH); 
    } else if (c=='e') {
      digitalWrite(record_trig, LOW); 
    }
  }
}
