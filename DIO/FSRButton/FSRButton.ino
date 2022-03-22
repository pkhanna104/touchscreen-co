// button FSRs
const int FSR1_PIN = A0; // Pin connected to FSR/resistor divider
const int FSR2_PIN = A5; // Pin connected to FSR/resistor divider
int fsr1ADC = 0;
int fsr2ADC = 0;

void setup() {
  pinMode (FSR1_PIN, INPUT);
  pinMode (FSR2_PIN, INPUT);
  Serial.begin( 9600 );
}

void loop() {

  // Determine and send the state of the button and the door
  fsr1ADC = analogRead(FSR1_PIN);
  fsr2ADC = analogRead(FSR2_PIN);
  Serial.print(fsr1ADC);
  Serial.print('/');
  Serial.println(fsr2ADC);
}
