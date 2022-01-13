char c; 
int record_trig = 2;
int strobe_iscan = 3;
int strobe_tdt = 8;
float t_now = 0;
float t_last_trig = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); 
  pinMode(strobe_iscan, OUTPUT); 
  pinMode(strobe_tdt, OUTPUT); 
  pinMode(record_trig, OUTPUT); 
  digitalWrite(strobe_iscan, LOW); 
  digitalWrite(strobe_tdt, LOW); 
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
    digitalWrite(strobe_iscan, HIGH);
    digitalWrite(strobe_tdt, HIGH);
    delay(10);
    digitalWrite(strobe_iscan, LOW);
    digitalWrite(strobe_tdt, LOW);
  }
}
