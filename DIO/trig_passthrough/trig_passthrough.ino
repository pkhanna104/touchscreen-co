int strobe_out_pin = 8;
int trig = 0;
float t_now = 0;
float t_last_trig = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); 
  pinMode(strobe_out_pin, OUTPUT); 
  digitalWrite(strobe_out_pin, LOW); 
}

void loop() {
  // put your main code here, to run repeatedly:
  t_now = millis();
  if (t_now > t_last_trig + 1000){
    t_last_trig = t_now;
    digitalWrite(strobe_out_pin, HIGH);
    delay(10);
    digitalWrite(strobe_out_pin, LOW);
  }
}
