# Project
This repository is for efficient pytorch-experiment.

## Contents and descriptions
```
├── arch                (neural network architectures)
│   ├── __init__.py
│   ├── mlp.py          (mlp)
│   └── resnet.py       (resnet)
├── datasets 
│   └── raws            (download and place raw datasets)
├── results             (directory to save experiment result)
├── tools               (supportive tools for experiment)
│   ├── metric
│   ├── plot
│   └── utils.py
├── configs.py          (separately configure arg_parser func for easy checking.)
├── main.py             (main code.)
├── README.md           (this file)
├── run_experiments.sh  (possible to conduct several experiments at once.)
└── test.py             (inference code)
```

