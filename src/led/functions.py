import RPi.GPIO as GPIO
import logging


def turn_on_pin(pin):
    GPIO.output(pin, GPIO.HIGH)
    logging.info(f"LED on pin {pin} is ON")


def turn_off_pin(pin):
    GPIO.output(pin, GPIO.LOW)
    logging.info(f"LED on pin {pin} is OFF")