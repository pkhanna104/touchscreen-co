#include <SparkFun_TB6612.h>

// Spinning motor params
const int offset = 1;
#define STBY 0
#define spin_IN1 11
#define spin_IN2 12
#define spin_PWM 13
Motor motor_spin = Motor(spin_IN1, spin_IN2, spin_PWM, offset, STBY);

// Spin motor enocder
#define ENCODER_A   14 //--> PH3; #### 3 
#define ENCODER_B   2 //--> PE4; #### 2

// Motor encoder variables
volatile byte motordir;
volatile int count = 0;
volatile int spin_ir_count = 0;

// Solenoid motor params
#define sol_IN1 10 
#define sol_IN2 9
#define sol_PWM 8
Motor motor_sol = Motor(sol_IN1, sol_IN2, sol_PWM, offset, STBY);

// Slide pot door motor 
#define pot_in1 6
#define pot_in2 5
#define pot_pwm 4
Motor motor_pot = Motor(pot_in1, pot_in2, pot_pwm, offset, STBY);

// Slidepot sensor 
const int slidepot_sensor = A0; //-- slidepot
int pot_sensorValue;
bool abortclose = false;
int startTrying; 
int lastT = -1; 
int lastpos = -1;
int velo = 0;
int lastvelo = -1500;
int accel = -1500;
int T = 0;
int lastT_register = 0;
int Tdiff = -1;

// Lift IR sensor
const int Lift_IR = 52;   

// spinning IR
#define spin_IR 3

// FSR pins
const int analog_FSR1 = A2;
const int analog_FSR2 = A1;

// stop spinning variable 
bool stop_spinning = false; 

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

float last_spin_cnt = 0; 
float vel = 0.02;
float last_vel = 0.02; 
float avg_vel = 0.02; 
float lastTm = -1;
int last_tc = 0; 

volatile int count_spin_tm1=0; 

//////////////////////////////////////////
//////////////////////////////////////////

// This method is for estimating the current velocity using the hall sensor encoder // 
void interrupt_motorencoder() {
  motordir = digitalRead(ENCODER_A); //PINJ;
  //motordir = motordir & B00000010;
  if (motordir) {
    count += 1;
  }
  else {
    count -= 1;
  }

  // get velocity
  if (lastTm == -1) {
    lastTm = micros(); 
  }
  vel = (1.) / (micros() - lastTm); // counts per uS
  vel *= us2s; // counts per sec
  vel /= (22 * 12); // approximately revs per sec

  // smooth velocity estimate // 
  avg_vel = vel*.5 + avg_vel*.5;
  
    // some sort  of coarse speed control // 
    if ((avg_vel < 0.35) and (in_targ == 1)) {

      // double, triple, quadruple etc. /
      if (factor >= 1) {
      factor += .5;
      }

      // double factor 
      else {
        factor *= 1.25;
      }
    }

    // if velocity too high 
    else if ((avg_vel > 0.8) and (in_targ == 1)) {
      if (factor <= 1) {
        // halve
        factor /= 1.25;
      }
      else {
        factor -= 1;
      }
    }

    // bound the 'factor' 
    factor = min(factor, 10);
    factor = max(factor, 0.1);

  // save current time so we can use it for next time; 
  lastTm = micros();

  // if the next spin_ir_count is correct, then start counting on this one 
  if ((in_targ == 1) and ((spin_ir_count + 1) %12 == tc)) {
    if (motordir) {
      count_spin_tm1 += 1; 
    }
    else {
      count_spin_tm1 -=1; 
    }
  }
  else {
    count_spin_tm1 = 0; 
  }
}

// Use the spin IR sensor 
void interrupt_spinIR() {
  // Only count if > 200 ms after original 
    // Keep track of own spin_ir_count
    // made this 10 so that you really have to be past in order to increment 
  if (abs(count - lastCnt) > 10) {
  
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

  // set this regardless -- you need to undergo a period of non activity to actually increment; 
  //last_spin_cnt = micros(); 
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

  // Get LED setup 
  pinMode(LED_BUILTIN, OUTPUT);

  // Pause to get this going//
  delay(100);
}

void loop() {

  // process serial input
  if (Serial.available() >= 1) {
    c = Serial.read();
    if (c == 'd') {
      digitalWrite(LED_BUILTIN, HIGH);
      handle_word();

      // add 0.1 so that when we convert this from char --> int it rounds correclty 
      target_count = target_count + 0.1;

      // Integer from 0-11 of target to hit //
      tc = (int) target_count;

      // Activate solenoid to get outta the way //
      act_solenoid();

      // Set in-targ
      in_targ = 1; 

      // these are variables needed to say whether we've successfully reached the target 
      n_tms = 0;
      n_tms2 = 0;
  
      // set factor equal to "1" 
      factor = 1;
      
      // eventually this will be used to make sure spinning stops // 
      keep_spinning = true;
    }
    
    else if (c == 'n') {
      deact_solenoid(); 
      in_targ = 0; 
      stop_spinning = true; 
      digitalWrite(LED_BUILTIN, LOW);
    }

    else if (c == 'c') {
      // close the door wherever we are
      close_door(); 
    }

    else if (c == 'o') {
      // open the door wherever we are 
      open_door(); 
    }
  }
 
  // deal with spinning now //
  if (in_targ == 1) {
    // Go to target method //
    go_to_target();

    if ((diff == 0) and (n_tms >= nthresh_tms) and (n_tms2 >= nthresh_tms)) {
      in_targ = 0; 
      motor_spin.brake(); 
      
      // deactive solenoid if we haven't already 
      deact_solenoid(); 
      digitalWrite(LED_BUILTIN, LOW);
  
      // set last tc 
      last_tc = tc; 

      
    }
  }
  print_serial();
  delay(5);
}

void open_door() {
  pot_sensorValue = analogRead(slidepot_sensor); 
    
   // Run the motor until the door is open
   while (pot_sensorValue > 100) {
    motor_pot.drive(-255, 100);
    pot_sensorValue = analogRead(slidepot_sensor);
    delay(5); 
    print_serial(); 
  }
  motor_pot.brake(); 
}

void close_door() {
    pot_sensorValue = analogRead(slidepot_sensor); 
    lastT = -1; 
    lastpos = -1; 

    // Try closing 
    motor_pot.drive(200, 80); 
    startTrying = millis(); // time that we tried starting to close the door 

    // Run the motor until the door is closed
    while (pot_sensorValue < 1000) {
      print_serial(); 
      if (abortclose == false){
        
        pot_sensorValue = analogRead(slidepot_sensor); 
        
        if (pot_sensorValue < 1000) {
         
          // Track acceleration to open the door if the slider decelerates
          T = millis();
  
          // Check after 2nd iteration 
          if (lastT != -1) {
            Tdiff = T - lastT_register;
            if (Tdiff > 5) {
              if (lastpos != -1) {
                velo = (pot_sensorValue - lastpos)/(Tdiff);
                if (lastvelo != -1500) {
                  accel = 100*(velo - lastvelo)/(Tdiff);
                  if (accel < 0) {
                    abortclose = true;
                  }
                } // end if there is a previous velo calc
                lastvelo = velo;
              } // end if there is a previous pos calc
              lastpos = pot_sensorValue;
              lastT_register = T;
            }
          }
          lastT = T;

          if ((!abortclose) and ((millis() - startTrying) > 1000)) {
            abortclose=true; 
          }

        }
        delay(5);
      }

      else if (abortclose == true) {
        // Run the motor until the door is open
        motor_pot.drive(-100, 50); 
        motor_pot.brake(); 
        abortclose = false;
        delay(500); 
        // Try closing again 
        motor_pot.drive(200, 80); 
    }
  }
  motor_pot.brake(); 
}

// activate solenoid
void act_solenoid() {
  forward(motor_sol, motor_sol, 200);
  delay(30);
  forward(motor_sol, motor_sol, 100);
}

// deactivate solenoid
void deact_solenoid() {
  motor_sol.brake();
  count_spin_tm1 = 0;  
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
  
  // Compute the difference, mod 12 so its always between 0 and 11; 
  if (tc < spin_ir_count) {
    diff = (tc + 12 - spin_ir_count); 
  }
  else {
    diff = tc - spin_ir_count; 
  }

  // Criteria based on IR sensor encoder //
  if (( diff > 0) or (n_tms < nthresh_tms) or (n_tms2 < nthresh_tms)) {

    // Only do this if we're supposed to keep spinning //
    if (keep_spinning) {

      // Set factor equal to 1; 
      if (diff > 2 ) {
        tm = diff;
        drivez = 15;
      }
      else if (diff > 0) {
        if (avg_vel >= .25) {
        tm = diff;
        drivez = 15; 
        }
        else if (avg_vel < .25) {
        tm = diff; 
        drivez = 15; 
        }
      }
      else {
        tm = 0;
        drivez = 15;
      }

      // Drive the motor forward, factor adjusts time; 
      motor_spin.drive(-1 * drivez, factor * tm);
    }

    else {
      // back drive and try again // 
      //motor_spin.drive(80, 5); 
      motor_spin.brake(); 
      delay(100); 
      keep_spinning = true;  
      lastTm = micros(); 
    } 
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

  // Trigger solenoid -- make this speed dependent
  if ((count_spin_tm1 == 18) and (avg_vel > 0.7)) {
    deact_solenoid(); 
  }
  if ((count_spin_tm1 == 19) and (avg_vel > .6)) {
    deact_solenoid(); 
  }
  if ((count_spin_tm1 == 20) and (avg_vel > .4)) {
      deact_solenoid(); 
    }
  if ((count_spin_tm1 == 21) and (avg_vel > .2)) {
    deact_solenoid();
 }
 if ((count_spin_tm1 == 22) and (avg_vel > .1)) {
    deact_solenoid(); 
 }
 if (count_spin_tm1 == 24) {
  deact_solenoid(); 
 }

  // Check if we should keep spinning 
  check_decel();
}

void check_decel() {
 
  if ((micros() - lastTm) > 100000) {
    keep_spinning = false; 
  }
  else {
    keep_spinning = true;  
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
  Serial.print("\t"); 
  Serial.print(avg_vel); 
  Serial.print("\t"); 
  Serial.print(count%22); 
  Serial.print("\t");
  Serial.print(count_spin_tm1); 
  Serial.print("\t"); 
  Serial.print(analogRead(slidepot_sensor)); 
  Serial.print("\t");
  Serial.println(abortclose);  
}
