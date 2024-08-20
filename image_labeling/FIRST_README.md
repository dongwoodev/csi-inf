# Labeling
## File system
```
├── FIRST_README.md
├── action
│   ├── labeling_action.py
│   ├── labeling_action_csi.py
│   └── labeling_action_merge.py
├── labeling_csi.py
└── location
    ├── labeling_loc.py
    ├── labeling_loc_csi.py
    └── labeling_loc_merge.py
```

## Location Labeling
#### First, you need a Skeleton data file that you created using Vision Recognition.

|path|0_X|0_Y|1_X|...|16_Y|
|---|---|---|---|---|---|
|2024-05-05_22:50:50.50__R.jpg|1053|45|0|...|1|

- The data file can be imported as `-f` or `--file` Argument.
- if you have images in `./test/L` and `./test/R`, execute `class LocationVisualizer`.
  - you can confirm visualized image in `./loc_visual`. 

#### Start with location labeling through `labeling_loc.py`.
```bash
python labeling_loc.py -f data_L.csv -s L
# python labeling_loc.py --file data_R.csv --side R
```

#### Merge Two label files(Left side, Right side) through `labeling_loc_merge.py`.
- data_L_complete.csv
- data_R_complete.csv
```bash
python labeling_loc_merge.py
```
|path|location|
|---|---|
|2024-05-05_22:50:50.50__R.jpg|AP|

#### The timestamp and label of csi data generate data **for training**. ([csi raw data required](https://github.com/dongwoodev/csi-inf/tree/main/collect))
    - grid : Generates sequence data in N second. (`-g`, `--grid`)
    - stride : Proceed to the sliding window every 0.1 seconds. (`-s`, `--stride`)

```bash
├── csi
│   ├── 2024-05-05_22:50:50.50__R.csv
│   └── ...
├── complete # generated data
│   ├── AP/ # data file directory
│   ├── ESP/
│   └── Mid/
└── labeling_loc_csi.py # execution file
```

```bash
python labeling_loc_csi.py -g 1 -s 0.1
python labeling_loc_csi.py --grid 1 --stride 0.1
```

As a result...
|Timestamp|0|1|...|383|location|
|---|---|---|---|---|---|
|2024-05-05 22:50:50.500|0|0|...|0|AP|

---

## Action Labeling
#### First, you need Image file that you Classified Sit/Stand.
   - Label each with a Sit/Stand action as `-a` or `--action` Argument.

```bash
├── sit
│   ├── 2024-05-05_22:50:50.50__R.jpg
│   └── 2024-05-05_22:50:50.50__L.jpg
├── stand 
│   ├── 2024-05-05_22:51:50.50__R.jpg
│   └── 2024-05-05_22:51:50.50__L.jpg
└── labeling_action.py # execution file
└── 2024-05-05_22:50:50.50_sit.csv # result
└── 2024-05-05_22:51:50.50_stand.csv # result
```

```bash
python labeling_action.py -a stand
# python labeling_action.py --action sit
```

|Timestamp|action|
|---|---|
|2024-05-05 22:50:50.500|sit|


#### Merge Two label files(sit, stand) through `labeling_action_merge.py`.
- 2024-05-05_22:50:50.50_sit.csv.csv
- 2024-05-05_22:51:50.50_stand.csv

```bash
python labeling_loc_merge.py
```

#### The timestamp and label of csi data generate data **for training**. ([csi raw data required](https://github.com/dongwoodev/csi-inf/tree/main/collect))

 - grid : Generates sequence data in N second. (`-g`, `--grid`)
 - stride : Proceed to the sliding window every 0.1 seconds. (`-s`, `--stride`)


```bash
├── csi
│   ├── 2024-05-05_22:50:50.50__R.csv
│   └── ...
├── complete # generated data
│   ├── sit/ # data file directory
│   └── stand/
└── labeling_action_csi.py # execution file
```

```bash
python labeling_action_csi.py -g 1 -s 0.1
python labeling_action_csi.py --grid 1 --stride 0.1
```

As a result...
|Timestamp|0|1|...|383|action|
|---|---|---|---|---|---|
|2024-05-05 22:50:50.500|0|0|...|0|sit|