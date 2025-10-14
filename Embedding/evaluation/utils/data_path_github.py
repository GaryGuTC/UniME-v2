import os
ABS_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#### CoCo
coco = {
    "data_path": f'{ABS_PATH}/data/coco_test',
}

### Flickr30k 
flickr30K = {
    "data_path": f'{ABS_PATH}/data/flickr30k_test',
}

#### Urban1k
Urban1k = {
    "image_root": f'{ABS_PATH}/data/Urban1k/Urban1k/image/',
    "caption_root": f'{ABS_PATH}/data/Urban1k/Urban1k/caption/'
}

#### ShareGPT4V
ShareGPT4v = {
    "data4v_root" : f'{ABS_PATH}/data/shareGPT4v/data/shareGPT4v/', 
    "json_name" : 'share-captioner_coco_lcs_sam_1246k_1107.json',
    "image_root" : f'{ABS_PATH}/data/shareGPT4v/data/'
}

#### SugarCrepe
SugarCrepe = {
    "data_root" : f'{ABS_PATH}/data/sugar-crepe/data',
    "image_root" : f'{ABS_PATH}/data/coco2017/val2017'
}