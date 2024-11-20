import RPi.GPIO as GPIO

red_pin = 17
green_pin = 27


def setup_leds():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(red_pin, GPIO.OUT)
    GPIO.setup(green_pin, GPIO.OUT)