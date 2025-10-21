#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ultralytics import YOLO
from uvctypes import *
import time
import cv2
import numpy as np

from multiprocessing import Process, Manager
import socket
import random
import json

HOST = '192.168.43.101'   # 서버 B 주소
PORT = 5001        # 서버 B 포트

try:
  from queue import Queue
except ImportError:
  from Queue import Queue
import platform
MIN_TEMP = 10 + 273.15
MAX_TEMP = 60 + 273.15
BUF_SIZE = 2
q = Queue(BUF_SIZE)

def send_loop(status):
    print("Start Socket")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print(f"[A] Connected to {HOST}:{PORT}")
        while True:
            payload = json.dumps({'model_result': status.value})
            s.sendall(payload.encode('utf-8') + b'\n')
            print(f"[A] Sent -> a: {status.value}")
            time.sleep(5)  # 5초마다 전송

def py_frame_callback(frame, userptr):

  array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
  data = np.frombuffer(
    array_pointer.contents, dtype=np.dtype(np.uint16)
  ).reshape(
    frame.contents.height, frame.contents.width
  ) # no copy

  # data = np.fromiter(
  #   frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
  # ).reshape(
  #   frame.contents.height, frame.contents.width, 2
  # ) # copy

  if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
    return

  if not q.full():
    q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

def ktof(val):
  return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
  return (val - 27315) / 100.0

def raw_to_8bit(data):
  cv2.normalize(data, None, alpha = 0, beta = 65535, norm_type = cv2.NORM_MINMAX)
  #np.right_shift(data, 8, data)
  data *= 256
  return np.uint8(data)

def display_temperature(img, val_k, loc, color):
  val = ktoc(val_k)
  cv2.putText(img,"{0:.1f} degC".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
  x, y = loc
  cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
  cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

def main(status):
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devh = POINTER(uvc_device_handle)()
  ctrl = uvc_stream_ctrl()
  model = YOLO("best.pt")
  res = libuvc.uvc_init(byref(ctx), 0)

  if res < 0:
    print("uvc_init error")
    exit(1)

  try:
    res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
      print("uvc_find_device error")
      exit(1)

    try:
      res = libuvc.uvc_open(dev, byref(devh))
      if res < 0:
        print("uvc_open error")
        exit(1)

      print("device opened!")

      print_device_info(devh)
      print_device_formats(devh)

      frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
      if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

      libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
      )

      res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
      if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)

      try:
        while True:
          data = q.get(True, 5)
          if data is None:
            break
          min_adc = MIN_TEMP * 100
          max_adc = MAX_TEMP * 100
          data[data<=min_adc] = min_adc+1
          data[data>=max_adc] = max_adc-1
          #data = cv2.resize(data[:,:], (640, 480))
          minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
          scaled = np.clip((data - min_adc) / (max_adc - min_adc), 0, 1)
          img = raw_to_8bit(scaled)
          img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
          results = model(img)
          if results[0].boxes:
            box = results[0].boxes[0]
            class_id = int(box.cls)  # Get class ID
            class_label = results[0].names[class_id]
            status.value = class_id  # Get class label from class ID
            print(f'Detected class: {class_label}')  # Print class label
          annotated_image = results[0].plot()
          display_temperature(annotated_image, minVal, minLoc, (255, 0, 0))
          display_temperature(annotated_image, maxVal, maxLoc, (0, 0, 255))
          annotated_image = cv2.resize(annotated_image[:,:], (320, 240))
          cv2.imshow('Lepton Radiometry', annotated_image)
          cv2.waitKey(1)

        cv2.destroyAllWindows()
      finally:
        libuvc.uvc_stop_streaming(devh)

      print("done")
    finally:
      libuvc.uvc_unref_device(dev)
  finally:
    libuvc.uvc_exit(ctx)

if __name__ == '__main__':
  manager = Manager()
  status = manager.Value('i', 0)
  status.value = 0
  process1 = Process(target=send_loop, args=(status,))
  process1.start()
  main(status)
  process1.terminate()
