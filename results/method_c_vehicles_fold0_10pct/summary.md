# Method C — 5-class COCO Vehicles, 10% labeled, fold 0

| Method | Round | mAP50 | mAP50-95 | PL boxes | Notes |
|---|---:|---:|---:|---:|---|
| Baseline A | R0 | 0.4120 | 0.2466 | — | supervised |
| Baseline B | R1 | 0.4796 | 0.3041 | 30597 | naive PL |
| Baseline B | R2 | 0.4736 | 0.3030 | 27784 | naive PL |
| Baseline B | R3 | 0.4594 | 0.2899 | 29738 | ep5 collapse |
| Method C | R1 | 0.4836 | 0.3071 | 29709 | stability-gated |
| Method C | R2 | 0.4619 | 0.2902 | 25120 | stability-gated |
| Method C | R3 | 0.4603 | 0.2898 | 24935 | best ep19 |

Signal history:
- R1: RawChurn=0.0293, StableYield=0.2613, ClassDrift=0.0000
- R2: RawChurn=0.5968, StableYield=0.3642, ClassDrift=0.0013
- R3: RawChurn=0.3972, StableYield=0.4094, ClassDrift=0.0013
