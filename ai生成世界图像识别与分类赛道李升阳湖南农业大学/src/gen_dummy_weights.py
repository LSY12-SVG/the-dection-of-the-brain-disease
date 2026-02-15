from pathlib import Path
import json
import torch
import timm

def main():
    Path('weights').mkdir(exist_ok=True)
    model = timm.create_model('efficientnetv2_s', pretrained=False, num_classes=1)
    torch.save(model.state_dict(), Path('weights/best.pt'))
    with open('weights/meta.json','w',encoding='utf-8') as f:
        json.dump({'model_name':'efficientnetv2_s','threshold':0.5}, f, ensure_ascii=False, indent=2)
    print('saved dummy weights to ./weights')

if __name__ == '__main__':
    main()

