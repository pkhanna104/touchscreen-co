#include <SparkFun_TB6612.h>

// Spinning motor params
const int offset = 1;
#define STBY 0
#define spin_IN1 8
#define spin_IN2 9
#define spin_PWM 4
Motor motor_spin = Motor(spin_IN1, spin_IN2, spin_PWM, offset, STBY);

// Spin motor enocder
#define ENCODER_A   6 // 1 --> PH3; 
#define ENCODER_B   2 // 2 --> PE4; 

// Motor encoder variables
volatile byte motordir;
volatile int count = 0;
volatile int spin_ir_count = 0;

// Solenoid motor params
#define sol_IN1 10
#define sol_IN2 11
#define sol_PWM 5
Motor motor_sol = Motor(sol_IN1, sol_IN2, sol_PWM, offset, STBY);

// Lift IR sensor
const int Lift_IR = 52;

// spinning IR
#define spin_IR 3

// FSR pins
const int analog_FSR1 = A14;
const int analog_FSR2 = A15;

// Variables for dealing with serial inputs
char c;
char dio_data[1];

// Motor control variables
bool keep_spinning = true;
int n_tms = 0;
int n_tms2 = 0;
int nthresh_tms = 1;
float target_count;
int tc;
int diff;
int encoder_targ;
int encoder_diff;
int tm;
int drivez = 50;

int tmp;
int tmp1;
int d;
int d1;

int lastCnt = 0;
int in_targ = 0;
int nsteps;
float us2s = pow(10, 6);
volatile float factor = 1;

float vel = 0;
float lastTm = micros();

volatile int count_spin_tm1=0; 

//////////////////////////////////////////
//////////////////////////////////////////


// This method is for estimating the current velocity using the hall sensor encoder // 
void interrupt_motorencoder() {
  motordir = PINH;
  motordir = motordir & B00001000;
  if (motordir) {
    count += 1;
  }
  else {
    count -= 1;
  }

  // get velocity
  vel = (1.) / (micros() - lastTm); // counts per uS
  vel *= us2s; // counts per sec
  vel /= (22 * 12); // approximately revs per sec

    // some sort  of coarse speed control // 
    if ((vel < 0.25) and (in_targ == 1)) {

      // double, triple, quadruple etc. /
      if (factor >= 1) {
      factor += 1;
      }

      // double factor 
      else {
        factor *= 2;
      }
    }

    // if velocity too high 
    else if ((vel > 0.75) and (in_targ == 1)) {
      if (factor <= 1) {
        // halve
        factor /= 2;
      }
      else {
        factor -= 1;
      }
    }

    // bound the 'factor' 
    factor = min(factor, 3);
    factor = max(factor, 0.1);

  // save current time so we can use it for next time; 
  lastTm = micros();

  if ((in_targ == 1) and ((spin_ir_count + 1)%12 == tc)) {
    count_spin_tm1 += 1; 
  }
  else {
    count_spin_tm1 = 0; 
  }
}

// Use the spin IR sensor 
void interrupt_spinIR() {

  // Keep track of own spin_ir_count
  if (abs(count - lastCnt) > 5) {

    // This really counts then // 
    if (motordir) {
      spin_ir_count += 1;
    }
    else {
      spin_ir_count -= 1;
    }

    // Now use this update to make the other count more accurate // 
    // Adjust count to the nearest 22 --> use the spin IR sensor to keep the encoder more accurate / stable // 
    int tmp = count / 22 ;
    int tmp1 = tmp + 1;
    int d = abs(count - 22 * tmp);
    int d1 = abs(count - 22 * tmp1);

    if (d < d1) {
      count = 22 * tmp;
    }
    else {
      count = 22 * tmp1;
    }
        
    // Mod 12 //
    spin_ir_count = spin_ir_count % 12;

    // Last count where you iterated
    lastCnt = count;
  }
}

void setup() {
  // put your setup code here, to run once:

  // Motor encoder Setup //
  Serial.begin(115200);
  pinMode(ENCODER_B, INPUT_PULLUP);
  pinMode(ENCODER_A, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_B), interrupt_motorencoder, RISING);

  // IR spin setup
  pinMode(spin_IR, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(spin_IR), interrupt_spinIR, RISING);

  // Lift IR:
  pinMode(Lift_IR, INPUT_PULLUP);

  // Pause to get this going//
  delay(100);
}

void loop() {

  // process serial input
  if (Serial.available() >= 1) {
    c = Serial.read();
    if (c == 'd') {
      handle_word();

      // add 0.1 so that when we convert this from char --> int it rounds correclty 
      target_count = target_count + 0.1;

      // Integer from 0-11 of target to hit //
      tc = (int) target_count;

      // Activate solenoid to get outta the way //
      act_solenoid();

      // Go to target method //
      go_to_target();

      // no longer in "go to target" mode 
      in_targ = 0;

    }
  }
  print_serial();
  delay(10);
}

// activate solenoid
void act_solenoid() {
  forward(motor_sol, motor_sol, 200);
  delay(30);
  forward(motor_sol, motor_sol, 50);
}

// deactivate solenoid
void deact_solenoid() {
  motor_sol.brake();
}

// read the target value into "target_count"
void handle_word() {
  Serial.readBytes(dio_data, 1);
  char d1 = dio_data[0];
  byte data_byte = dio_data[0];
  target_count = 0.;
  for (int bit_idx = 0; bit_idx < 8; bit_idx += 1) {
    if (bitRead(data_byte, bit_idx) == 1) {
      target_count += pow(2, bit_idx);
    }
  }
}

// go to the target, based on knowing where we are right now;
void go_to_target() {
  
  // these are variables needed to say whether we've successfully reached the target 
  n_tms = 0;
  n_tms2 = 0;

  // set in-targ equal to "1" 
  in_targ = 1;

  // set factor equal to "1" 
  factor = 1;
    
  // eventually this will be used to make sure spinning stops // 
  keep_spinning = true;

  // Compute the difference, mod 12 so its always between 0 and 11; 
  if (tc < spin_ir_count) {
    diff = (tc + 12 - spin_ir_count); 
  }
  else {
    diff = tc - spin_ir_count; 
  }

  // Criteria based on IR sensor encoder //
  while (( diff > 0) or (n_tms < nthresh_tms) or (n_tms2 < nthresh_tms)) {

    // Only do this if we're supposed to keep spinning //
    if (keep_spinning) {

      // Set factor equal to 1; 
      if (diff > 2 ) {
        tm = diff;
        drivez = 50;
      }
      else if (diff > 0) {
        tm = 1;
        drivez = 50; 
      }
      else {
        tm = 0;
        drivez = 50;
      }

      // Drive the motor forward, factor adjusts time; 
      motor_spin.drive(-1 * drivez, factor * tm);

      // stop the motor 
      motor_spin.brake();
    }

    // Wait  //
    delay(5);

    // Re-calc diff //
    if (tc < spin_ir_count) {
      diff = (tc + 12 - spin_ir_count); 
    }
    else {
      diff = tc - spin_ir_count; 
    }

    // Print serial //
    print_serial();

    // If diff == thresh increment //
    if ( diff == 0 ) {
      n_tms += 1;
    }
    
    // Re-set if ever unequal //
    else {
      n_tms = 0;
    }

    // Make IR criteria 
    if (digitalRead(spin_IR) == 1) {
      n_tms2 += 1;
    }
    else {
      n_tms2 = 0;
    }

    // Trigger solenoid 
//    if ((n_tms > 0) and (n_tms2 > 0)) {
//      deact_solenoid();
//    }
      if (count_spin_tm1 == 21) {
          deact_solenoid(); 
        }
      }
  }

void print_serial() {
  Serial.print(digitalRead(Lift_IR)); 
  Serial.print("\t");
  Serial.print(analogRead(analog_FSR1)); 
  Serial.print("\t");
  Serial.print(analogRead(analog_FSR2)); 
  Serial.print("\t");
  Serial.print(spin_ir_count);   
  Serial.print("\t");
  Serial.print(in_targ);
  // Extra 
  Serial.print("\t");
  Serial.print(factor);
  Serial.print("\t");
  Serial.print(vel);
  Serial.print("\t"); 
  Serial.println(tc); 
}
