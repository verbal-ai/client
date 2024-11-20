import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)

try:
    while True:
        GPIO.output(17, GPIO.HIGH)
        sleep(1)
        GPIO.output(17, GPIO.LOW)
        sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()