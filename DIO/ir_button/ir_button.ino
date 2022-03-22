/* 
  IR Breakbeam sensor demo!
*/

#define LED1PIN 2
#define LED2PIN 3

#define SENSOR1PIN 4
#define SENSOR2PIN 5

// variables will change:
int sensor1State = 0, sensor2State = 0, lastState=0;         // variable for reading the pushbutton status

void setup() {
  // initialize the LED pins as an output:
  pinMode(LED1PIN, OUTPUT);     
  digitalWrite(LED1PIN, HIGH);

  pinMode(LED2PIN, OUTPUT);     
  digitalWrite(LED2PIN, HIGH);
   
  // initialize the sensor pin as an input:
  pinMode(SENSOR1PIN, INPUT);     
  digitalWrite(SENSOR1PIN, HIGH); // turn on the pullup
  pinMode(SENSOR2PIN, INPUT);     
  digitalWrite(SENSOR2PIN, HIGH); // turn on the pullup
  
  Serial.begin(9600);
}

void loop(){
  // read the state of the pushbutton value:
  sensor1State = digitalRead(SENSOR1PIN);
  sensor2State = digitalRead(SENSOR2PIN);

  // check if the sensor beam is broken
  // if it is, the sensorState is LOW:
//  if (sensorState == LOW) {     
//    // turn LED on:
//    digitalWrite(LEDPIN, HIGH);  
//  } 
//  else {
//    // turn LED off:
//    digitalWrite(LEDPIN, LOW); 
//  }
//  


//  if (sensor1State && sensor2State && !lastState) {
//    Serial.println("Unbroken");
//    lastState = 1;
//  } 
//  if ((!sensor1State || !sensor2State) && lastState) {
//    Serial.println("Broken");
//    lastState = 0;
//  }
  if (sensor1State && sensor2State) {
    Serial.println(0);
  } 
  if (!sensor1State || !sensor2State) {
    Serial.println(1);
  }
}
