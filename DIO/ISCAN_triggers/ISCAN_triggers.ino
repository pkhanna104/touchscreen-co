char c; 
int record_trig = 2;
int strobe = 3;
float t_now = 0;
float t_last_trig = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); 
  pinMode(strobe, OUTPUT); 
  pinMode(record_trig, OUTPUT); 
  digitalWrite(strobe, LOW); 
  digitalWrite(record_trig, HIGH); 
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() >= 1) {
    c = Serial.read(); 
    if (c=='s') {
      digitalWrite(record_trig, LOW); 
    } else if (c=='e') {
      digitalWrite(record_trig, HIGH); 
    }
  }

  t_now = millis();
  if (t_now > t_last_trig + 1000){
    t_last_trig = t_now;
    digitalWrite(strobe, HIGH);
    delay(500);
    digitalWrite(strobe, LOW);
  }
}
