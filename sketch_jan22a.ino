#include <Servo.h>

Servo myServo; // Create a servo object

int currentPos = 0; // initial position

void setup() {
  Serial.begin(9600); // Initialize serial communication
  myServo.attach(9); // Attach the servo to pin 9
}

void loop() {
  if (Serial.available() > 0) {
    int targetPos = Serial.parseInt();
    Serial.print(targetPos); // Read the target position from Serial Monitor
    targetPos = constrain(targetPos, 0, 180); // Ensure target position is within valid range

    // Move to the target position
    if (currentPos < targetPos) {
      for (int pos = currentPos; pos <= targetPos; pos++) {
        myServo.write(pos);
        delay(15); // Adjust the delay for the desired speed
      }
    } else if (currentPos > targetPos) {
      for (int pos = currentPos; pos >= targetPos; pos--) {
        myServo.write(pos);
        delay(15); // Adjust the delay for the desired speed
      }
    }

    currentPos = targetPos; // Update the current position
  }
}