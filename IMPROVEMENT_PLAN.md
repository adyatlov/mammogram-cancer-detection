# Improvement Plan

## Current State

| Metric | Baseline | Target (Reference) | Gap |
|--------|----------|-------------------|-----|
| pF1 | 0.067 | 0.5714 | 8.5x |
| ROC AUC | 0.86 | 0.93 | 8% |

## Key Insight from Reference

The single biggest improvement appears to be **positive-only label smoothing**:
- Instead of hard labels {0, 1}, use {0.0, 0.8}
- This prevents overconfidence on rare positive class
- May eliminate need for pos_weight

---

## Experiment Strategy

### Principle: Isolate Variables

Each experiment changes **ONE thing** from the previous best. This lets us measure the true impact of each change.

```
Baseline (0.067)
    │
    ├── Exp 1: Label smoothing only
    │       └── Exp 2: + Remove pos_weight
    │               └── Exp 3: + More epochs (20)
    │                       └── Exp 4: + Larger image (768)
    │                               └── Exp 5: + Larger model (B4)
    │                                       └── Exp 6: + Better augmentation
```

### Why This Order?

1. **Label smoothing first** - Reference shows this is the key differentiator
2. **Remove pos_weight** - May conflict with label smoothing
3. **More epochs** - Cheap to test, reference used 20 epochs
4. **Larger image** - Medium cost, likely helpful for mammograms
5. **Larger model** - Higher cost, diminishing returns
6. **Augmentation** - Fine-tuning after major gains

---

## Planned Experiments

### Experiment 1: Positive Label Smoothing
**Hypothesis**: Smoothing positive labels to 0.8 will dramatically improve pF1

| Parameter | Baseline | Experiment 1 |
|-----------|----------|--------------|
| Loss | BCE | BCE |
| Positive label | 1.0 | **0.8** |
| Negative label | 0.0 | 0.0 |
| pos_weight | 23.67 | 23.67 |
| Everything else | same | same |

**Expected**: pF1 improvement from 0.067 → 0.3+ (if this is the key factor)

---

### Experiment 2: Remove pos_weight
**Hypothesis**: With label smoothing, pos_weight may hurt more than help

| Parameter | Exp 1 | Experiment 2 |
|-----------|-------|--------------|
| pos_weight | 23.67 | **1.0 (none)** |
| Everything else | same | same |

**Expected**: May improve or be neutral

---

### Experiment 3: More Epochs
**Hypothesis**: 10 epochs may be underfitting; reference used 20

| Parameter | Exp 2 | Experiment 3 |
|-----------|-------|--------------|
| num_epochs | 10 | **20** |
| Everything else | same | same |

**Expected**: Modest improvement, better convergence

---

### Experiment 4: Larger Image Size
**Hypothesis**: 512px loses detail important for cancer detection

| Parameter | Exp 3 | Experiment 4 |
|-----------|-------|--------------|
| image_size | 512 | **768** |
| batch_size | 32 | **16** (memory) |
| Everything else | same | same |

**Expected**: Improvement if detail matters

---

### Experiment 5: Larger Model
**Hypothesis**: EfficientNet-B4 has more capacity for complex patterns

| Parameter | Exp 4 | Experiment 5 |
|-----------|-------|--------------|
| model | efficientnet_b0 | **efficientnet_b4** |
| batch_size | 16 | **8** (memory) |
| Everything else | same | same |

**Expected**: Modest improvement, slower training

---

### Experiment 6: Enhanced Augmentation
**Hypothesis**: More augmentation improves generalization

| Parameter | Exp 5 | Experiment 6 |
|-----------|-------|--------------|
| augmentation | basic | **+ elastic, cutout, CLAHE** |
| Everything else | same | same |

**Expected**: Small improvement, better robustness

---

## Implementation Checklist

### Code Changes Needed

- [ ] **Label smoothing**: Modify loss function or labels in dataset
- [ ] **Configurable pos_weight**: Make it a CLI argument (default=None)
- [ ] **Threshold tuning**: Add post-training threshold search
- [ ] **Proper validation metrics**: Calculate pF1 on full val set, not just QA

### Metrics to Track (per experiment)

1. **val_pF1** - Primary metric (competition metric)
2. **val_auc_roc** - Ranking quality
3. **val_auc_pr** - Precision-recall tradeoff
4. **best_threshold** - Optimal classification threshold
5. **training_time** - Practical consideration

### Results Table Template

| Exp | Change | val_pF1 | Δ pF1 | ROC AUC | Best Thresh | Time |
|-----|--------|---------|-------|---------|-------------|------|
| 0 | Baseline | 0.067 | - | 0.86 | 0.5 | 17m |
| 1 | +label_smooth | ? | ? | ? | ? | ? |
| 2 | -pos_weight | ? | ? | ? | ? | ? |
| 3 | +epochs=20 | ? | ? | ? | ? | ? |
| 4 | +image=768 | ? | ? | ? | ? | ? |
| 5 | +model=B4 | ? | ? | ? | ? | ? |
| 6 | +augmentation | ? | ? | ? | ? | ? |

---

## Alternative Strategies (If Sequential Fails)

### If Label Smoothing Alone Doesn't Work

The reference may have other differences we don't see:
- Different preprocessing
- Different model initialization
- Patient-level aggregation
- Multi-view fusion

**Fallback**: Try combining multiple changes at once to match reference config

### Ablation Study (After Reaching Target)

Once we reach ~0.5 pF1, work backwards:
- Remove augmentation → measure drop
- Remove larger model → measure drop
- etc.

This validates which components actually matter.

---

## Success Criteria

| Level | pF1 | Status |
|-------|-----|--------|
| Broken | < 0.05 | ❌ |
| Baseline | 0.05-0.10 | ✓ Current |
| Improved | 0.10-0.30 | Target 1 |
| Competitive | 0.30-0.50 | Target 2 |
| Reference | 0.50-0.60 | Target 3 |
| SOTA | > 0.60 | Stretch |

---

## Notes

- Run each experiment with **same random seed** for fair comparison
- Save checkpoints for all experiments (can analyze later)
- Document unexpected findings in EXPERIMENT_LOG.md
