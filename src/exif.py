#!/usr/bin/env python3
import json
from .util import popen, EXECUTABLE_NAME

METADATA_PAYLOAD_MAGIC = f"{EXECUTABLE_NAME} "


def load_exif_metadata(image_path):
  p = popen(f"exiv2 --binary -K Exif.Photo.UserComment -Pv {image_path}")
  if p.returncode != 0: return {}
  payload_body = p.stdout.decode().strip().partition(METADATA_PAYLOAD_MAGIC)[2]
  try: return json.loads(payload_body)
  except json.decoder.JSONDecodeError: return {}


def save_exif_metadata(image_path, args, payload):
  payload = {**payload}   # deep copy
  payload['config'] = args.__dict__
  payload = json.dumps(payload, indent=None, separators=(',',':'))
  body = f"{METADATA_PAYLOAD_MAGIC}{payload}"
  p = popen(f"exiv2 -M 'set Exif.Photo.UserComment Ascii {body}' {image_path}")
  return p.returncode == 0