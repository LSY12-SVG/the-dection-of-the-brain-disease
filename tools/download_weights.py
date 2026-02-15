import os
import requests
import time
from pathlib import Path

def download_weights():
    url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-seg.pt"
    filename = "yolov8s-seg.pt"
    path = Path(filename)
    
    # We do NOT delete existing file anymore, we try to resume
    
    print(f"Downloading {filename} from {url}...")
    
    max_retries = 20 # Increased retries
    for attempt in range(max_retries):
        try:
            current_size = path.stat().st_size if path.exists() else 0
            headers = {'Range': f'bytes={current_size}-'}
            
            print(f"Attempt {attempt + 1}/{max_retries}... Resuming from {current_size/(1024*1024):.2f} MB")
            
            response = requests.get(url, stream=True, timeout=10, headers=headers)
            
            # 416 Range Not Satisfiable means we probably have the whole file (or server doesn't support range)
            if response.status_code == 416:
                print("Range not satisfiable, assuming download complete.")
                break
                
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0)) + current_size
            block_size = 8192
            
            mode = 'ab' if current_size > 0 else 'wb'
            
            with open(path, mode) as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    current_size += len(data)
                    if total_size > 0:
                        percent = current_size / total_size * 100
                        print(f"Downloaded {current_size / (1024*1024):.2f}/{total_size / (1024*1024):.2f} MB ({percent:.1f}%)", end='\r')
            
            print(f"\nDownload finished (this chunk). Total size: {path.stat().st_size / (1024*1024):.2f} MB")
            
            # Verify size (approx 22.8 MB)
            if path.stat().st_size < 22 * 1024 * 1024:
                print("\nFile seems incomplete. Retrying...")
                time.sleep(1)
                continue
            else:
                print("File size looks correct.")
                return # Success
                
        except Exception as e:
            print(f"\nError downloading file: {e}")
            time.sleep(2) # Wait a bit before retry
            
    if path.exists() and path.stat().st_size > 22 * 1024 * 1024:
        print("File seems large enough, assuming success.")
    else:
        print("\nFailed to download after multiple attempts.")

if __name__ == "__main__":
    download_weights()
