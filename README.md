#### micrograd

```
Forward pass:
R data = 13.797, grad = 0.000
└── + data = 13.797, grad = 0.000
    ├── + data = 13.797, grad = 0.000
    │   ├── * data = -2.576, grad = 0.000
    │   │   ├── data = 0.125, grad = 0.000 ← weight 0
    │   │   └── data = -20.660, grad = 0.000 ← input 0
    │   └── * data = 16.374, grad = 0.000
    │       ├── data = 0.163, grad = 0.000 ← weight 1
    │       └── data = 100.625, grad = 0.000 ← input 1
    └── data = 0.000, grad = 0.000 ← bias

Backward pass:
R data = 13.797, grad = 1.000
└── + data = 13.797, grad = 1.000
    ├── + data = 13.797, grad = 1.000
    │   ├── * data = -2.576, grad = 1.000
    │   │   ├── data = 0.125, grad = -20.660 ← weight 0
    │   │   └── data = -20.660, grad = 0.125 ← input 0
    │   └── * data = 16.374, grad = 1.000
    │       ├── data = 0.163, grad = 100.625 ← weight 1
    │       └── data = 100.625, grad = 0.163 ← input 1
    └── data = 0.000, grad = 1.000 ← bias
```

WIP

###### Credits
- [karpathy/micrograd](https://github.com/karpathy/micrograd)
- [Mathemmagician/rustygrad](https://github.com/Mathemmagician/rustygrad)
