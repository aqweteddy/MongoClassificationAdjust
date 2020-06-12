# MongoClassificationAdjust

## 模型

* 放在 weights 資料夾下
    * *.pth + *.json: normal
    * *.ckpt + *.json: lightning
* 記得修改 json 檔中 test_loader 位置為 Mongo_data/test/a

## Usage

```
usage: Inferer [-h] [--gpuid [GPUID]] [--weights WEIGHTS [WEIGHTS ...]]
               [--to TO]

optional arguments:
  -h, --help            show this help message and exit
  --gpuid [GPUID]       set GPU id (default: 1)
  --weights WEIGHTS [WEIGHTS ...]
                        weight for each model
  --to TO               output to csv file
```

### example

```bash
python main.py --gpuid 6 --weights 1 2 --to v1.csv
```