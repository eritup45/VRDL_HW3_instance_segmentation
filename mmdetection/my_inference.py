import numpy as np
import json
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from pycocotools.mask import encode, decode, area, toBbox

def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError

def produce_coco_result(image_id, bbox, score, RLE):
    result = {
        "image_id": image_id,
        "bbox": bbox,
        "score": score,
        "category_id": 1,
        "segmentation": RLE
    }
    return result

images=[
    'TCGA-A7-A13E-01Z-00-DX1.png',
    'TCGA-50-5931-01Z-00-DX1.png',
    'TCGA-G2-A2EK-01A-02-TSB.png',
    'TCGA-AY-A8YK-01A-01-TS1.png',
    'TCGA-G9-6336-01Z-00-DX1.png',
    'TCGA-G9-6348-01Z-00-DX1.png',
]
images = ['./data/coco/test2017/test/' +  img for img in images]

config_file = './config_cascade_mask_rcnn_50.py'
# 1212 mAP:0.06
checkpoint_file = './work_dirs/1212_ep40_cascade_mask_rcnn/epoch_1.pth'

# checkpoint_file = './work_dirs/config_cascade_mask_rcnn_50/epoch_20.pth'
# 1207
# checkpoint_file = './work_dirs/1207_ep60_config_cascade_mask_rcnn_50/epoch_9.pth'

# swin map:0.058
# config_file = './config_swinT.py'
# checkpoint_file = './work_dirs/1213_ep36_mask_rcnn_swin-t/epoch_32.pth'

device = 'cuda:0'
model = init_detector(config_file, checkpoint_file, device=device)
preds = inference_detector(model, images)



results = []
for i, res in enumerate(preds):
    bbox_scores = iter(res[0][0]) # bbox results
    masks = iter(res[1][0]) # Segmentation results
    for j in range(len(res[0][0])):
        # Get result one by one
        bbox_score = next(bbox_scores)
        mask = next(masks)

        img_id = i + 1

        # bbox
        score = bbox_score[4]
        bbox = bbox_score[:4].tolist()
        print('box & mask count: ', j)
        print('img_id:', img_id)
        print('score:', score)
        print('bbox :', bbox)

        # mask
        # mask: 2D array
        RLE = encode(np.asfortranarray(mask))
        RLE['counts'] = RLE['counts'].decode("utf-8")
        print(RLE)

        result = produce_coco_result(img_id, bbox, score, RLE)
        results.append(result)

with open('./answer.json', 'w') as fp:
    results = json.dumps(results, indent=4, sort_keys=False, default=convert)
    fp.write(results)

    # bbox
    # for j, bbox in enumerate(res[0][0]):
    #     print('box count: ', j)
    #     print('img_id:', i+1)
    #     print('score:', bbox[4])
    #     print('bbox :', bbox[:4])

    # # mask (boolean)
    # for j, mask in enumerate(res[1][0]):
    #     print('mask count: ', j)
    #     # mask: 2D array
    #     print('img_id:', i+1)
    #     # print('mask: ', mask)
    #     RLE = encode(np.asfortranarray(mask))
    #     RLE['counts'] = RLE['counts'].decode("utf-8")
    #     print(RLE)



# DEBUG
# OUT: 
# len(preds): 6 pictures
# len(preds[0]): 2 tasks (bbox + mask)
# len(preds[0][0]): 1 (No meaning here??)
# len(preds[0][0][0]): 100 detection results
print(f'len(preds): {len(preds)}')
print(f'len(preds[0]): {len(preds[0])}')
print(f'len(preds[0][0]): {len(preds[0][0])}')
print(f'len(preds[0][0][0]): {len(preds[0][0][0])}') 