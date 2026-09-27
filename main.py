import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solution.hands
mp.draw = mp.solution.drawing_utils

cap = cv2.VideoCapture(0)

def main():
    print("Hello, World!")

    if __name__ == "__main__":
        main()
