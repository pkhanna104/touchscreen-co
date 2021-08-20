#include <SparkFun_TB6612.h>

// Motor params 
const int offset = 1;
#define STBY 0
#define E3_IN1 8
#define E3_IN2 9
#define E3_PWM 4
Motor motor_E3 = Motor(E3_IN1, E3_IN2, E3_PWM, offset, STBY);

// Connect to the two encoder outputs!
// https://store.arduino.cc/usa/mega-2560-r3
// https://www.instructables.com/ATTiny-Port-Manipulation-Part-15-DigitalRead/

// 
#define ENCODER_A   3 // 3 --> PE5; 
#define ENCODER_B   2 // 2 --> PE4; 

// These let us convert ticks-to-RPM
//#define GEARING_ENCODERMULT     260

// Encoder variables 
byte motordir; 
int count = 0;  
int targ; 

// velocity estimate variation
float vel = 0.; 
float lastVel = 0.; 
float lastTm = 0.; 
float now = 0.;
int lastTarg; 

// rotation estimator 
float full_rot = 265.3; 

// pause 
int nwait=0; 
int nwait_tot = 100; 
bool keep_spinning; 

// data reading variables 
char dio_data[1];
double target_count; 
int tc; 

// motor control variables 
int thresh=0; // diff between true and target pos
int nthresh_tms=20; // number fo steps true and targ pos should be < thresh (corresponds to 50ms) 

int n_tms=0; // starting number of times 
int diff;  // diff b/w true and target pos 
int diff2; // diff b/w/ true and lastTarg 
int drivez=100; // drive PWM 
int tm; // lenght of PWM in ms; 
int tm2; 

char c; 

// IR sensor pin
const int IR_sensor_pin = 52; 

// FSR pins 
const int analog_FSR1 = A14;
const int analog_FSR2 = A15;

void interruptB() {
  //motordir = digitalRead(ENCODER_A);  
  motordir = PINE; 
  motordir = motordir & B00100000; 
  if (motordir) {
    count += 1; 
  }
  else {
    count -= 1; 
  }
  // update last velocity 
  lastVel = vel; 

  // re-calc current velocity 
  now = micros(); 
  vel = 1./(now - lastTm); // counts-per-uS
  vel = vel*1000; // counts-per-ms; 
  
  // update last time for next calc
  lastTm = now; 
}

void setup() {

  // Motor Params Setup // 
  Serial.begin(115200);           
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  pinMode(ENCODER_B, INPUT_PULLUP);
  pinMode(ENCODER_A, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_B), interruptB, RISING);
  //attachInterrupt(digitalPinToInterrupt(ENCODER_A), interruptA, RISING);
  delay(100);

  // IR sensor setup // 
  pinMode(IR_sensor_pin, INPUT_PULLUP);
}

void loop() {
  if (Serial.available() >= 1) {
    c = Serial.read();
    if (c=='d') {
      handle_word(); 
      target_count = target_count + 0.1;
      tc = (int) target_count;
      int ct_start = count; 
      go_to_target();
      
      // keep this last target just in case 
      lastTarg = targ; 
    }
  }
  print_serial();
  delay(5); // runs at 200 hz   
}

void go_to_target() {
  n_tms=0; 
  nwait=0; 
  keep_spinning = true; 
  
  // Make it so that targ is always less than count
  diff = abs(count - tc);

  float targc = (float) tc; 
  while (targc < count) {
    targc += full_rot; 
  }
  targ = (int) targc; 
  
  while ((diff >= thresh) and (n_tms < nthresh_tms)) {

    // Only do this if we're supposed to keep spinning // 
    if (keep_spinning) {
      if ((diff - thresh) > 50 ) {
        tm = 10;
      }
      else { 
        tm = 5; 
      }

      // Forward movement (last resort)  
      if (count > targ) {
        motor_E3.drive(drivez, tm);
        motor_E3.brake();
      }
//  
      // Reverse movement (main thing) 
       if (count < targ) {
        motor_E3.drive(-1*drivez, tm);
        motor_E3.brake();
      }

      // Check for deceleration // 
      //check_decel(); 
    }

    // Wait  // 
    delay(5);

    // Re-calc diff //
    diff = abs(count - targ);
    print_serial();

    // If diff == thresh increment // 
    if (diff == thresh) {
      n_tms += 1; 
    }
    // Re-set if ever unequal //
    else {
      n_tms = 0;
    }

    // Deal with this // 
    if (!keep_spinning) {

      // Drive back to last target and wait here // 
      diff2 = abs(count - lastTarg); 
      if (diff2 > 50) {
        tm2 = 10; 
      }
      
      else {
        tm2 = 5;
      }
      
      if (count > lastTarg) {
        motor_E3.drive(drivez, tm2);
        motor_E3.brake();
      }
      //  
      // Reverse movement (main thing) 
       if (count < lastTarg) {
        motor_E3.drive(-1*drivez, tm2);
        motor_E3.brake();
      }

      // Waiting time
      if (count == lastTarg) {
        nwait += 1; 
      }
      else {
        nwait = 1;
      }

      // If we've waited long enough, set keep_spinning to true //
      if (nwait == nwait_tot) {
        nwait = 0; 
        keep_spinning = true;
      }
    }
  }
}

void print_serial() {
//  Serial.print(digitalRead(ENCODER_A)); 
//  Serial.print("\t");
//  Serial.println(digitalRead(ENCODER_B)); 
  
//   IR sensor pin //
  Serial.print(digitalRead(IR_sensor_pin)); 
  Serial.print("     \t     "); 

  // FSR 1 // 
  Serial.print(analogRead(analog_FSR1)); 
  Serial.print("     \t     "); 

  // FSR 2 // 
  Serial.print(analogRead(analog_FSR2)); 
  Serial.print("     \t     "); 

  // Wheel location 
  Serial.print(count); 
  Serial.print  ("     \t     "); 

  // Wheel location 
  Serial.print(targ); 
  Serial.println  ("     \t     "); 

//  // Diff 
//  Serial.print(diff); 
//  Serial.print("     \t     "); 
//  // Vel
//  Serial.print(vel); 
//  Serial.print("     \t     "); 
//
//  // Last vel
//  Serial.print(lastVel); 
//  Serial.println("     \t     "); 

}

void handle_word() {
  Serial.readBytes(dio_data, 1);
  char d1 = dio_data[0];
  byte data_byte = dio_data[0];
  target_count = 0; 
  
  for (int bit_idx = 0; bit_idx < 8; bit_idx += 1) {
      if (bitRead(data_byte, bit_idx)==1) {
        target_count += pow(2, bit_idx); 
      }
  }
}

void check_decel() {
  // Time since last update // 
  float v = 1./(micros() - lastTm); // counts-per-uS
  v = v*1000.; //  counts-per-ms; 

  // Average number of counts per ms;  
  // float avgSpd = 0.5*(abs(vel) + abs(lastVel)); 

  // If dt is slower t and you're within the zone 
  if ((v < 0.25*abs(vel)) and (abs(count - lastTarg) < 40)) {
    keep_spinning = false;
  }

  else {
    keep_spinning = true;
  }
}
