char RCP = ' '; // Read Character from the Pad
int LED = 13; // LED out
int TTL = 12; // TTL pulse out

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200); // set baudrate
  pinMode(LED,OUTPUT); // set LED for output mode
  pinMode(TTL,OUTPUT); // set TTL for output mode
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() >= 1) {
    RCP = Serial.read();
    if(RCP == '1'){
      digitalWrite(LED,HIGH);
      digitalWrite(TTL,HIGH);
    }
    else {
      digitalWrite(LED,LOW);
      digitalWrite(TTL,LOW);
    }
  }
}
