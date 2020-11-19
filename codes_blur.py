from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import cv2


def get_mask(frame, bodypix_url='http://localhost:9000'):
    _, data = cv2.imencode(".jpg", frame)
    r = requests.post(
        url=bodypix_url,
        data=data.tobytes(),
        headers={'Content-Type': 'application/octet-stream'})
    mask = np.frombuffer(r.content, dtype=np.uint8)
    mask = mask.reshape((frame.shape[0], frame.shape[1]))
    return mask

def get_frame(cap, blur):
    mask = None
    while mask is None:
        try:
            mask = get_mask(cap)
        except requests.RequestException:
            print("mask request failed, retrying")
    #mask = post_process_mask(mask)

    inv_mask = 1 - mask
    # cap = adjust_gamma(cap, 1.5)
    # cap = warmImage(cap)
    for c in range(cap.shape[2]):
        cap[:, :, c] = cap[:, :, c] * mask + blur[:, :, c] * inv_mask
    return cap

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        blur = cv2.GaussianBlur(frame, (35,35),0)
        msk = get_frame(frame,blur)
        mask = cv2.flip(msk,1)
        cv2.imshow('final',mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=='__main__':
    main()