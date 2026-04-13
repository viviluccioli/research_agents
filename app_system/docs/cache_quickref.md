# Cache Quick Reference Card

## 🎯 Quick Start

**Enable cache (default):**
```
☑️ Use cached results (if available)
```

**Force fresh results:**
```
☑️ Force refresh (ignore cache)
```

## 💰 Cost Savings

| Cache Hit Rate | Savings per Run |
|---------------|-----------------|
| 0% (first run) | $0.00 |
| 50% | ~$0.75 |
| 80% | ~$1.20 |
| 100% (rerun) | ~$1.50-2.00 |

## 🔑 What Affects Cache Key?

✅ **Included** (changes invalidate cache):
- Paper text content
- Selected personas
- Persona weights
- Model name
- Paper type (empirical/theoretical/policy)
- Custom evaluation context

❌ **Excluded** (don't affect cache):
- File name
- Timestamps
- UI settings
- Output paths

## 🗂️ Cache Location

```
app_system/.referee_cache/
├── {cache_key_1}/
│   ├── metadata.json
│   ├── round_0_selection.json
│   ├── round_1_reports.json
│   └── ...
└── {cache_key_2}/
    └── ...
```

## 🧹 Cache Management

| Task | Method |
|------|--------|
| Clear current paper | Click "Clear Cache" button |
| Clear old entries (>30d) | Expand "Cache Statistics" → "Clear Old Cache" |
| Clear everything | Expand "Cache Statistics" → "Clear All Cache" |
| Check size | See "Cache Statistics" expander |

## 🔍 Understanding Cache Status

**In UI:**
```
💾 Cache available for: round_1_reports, round_2a_cross_exam
```

**In results:**
```
💰 Estimated Cost: $0.75 USD (7 LLM calls)
💾 Cache: 3/6 rounds cached (50% hit rate) — Saved ~$0.75 USD
```

**In expandable details:**
```
Cache Status by Round:
- ✅ round_0: HIT (cached)
- ✅ round_1: HIT (cached)
- ❌ round_2a: MISS (computed)
- ❌ round_2b: MISS (computed)
- ❌ round_2c: MISS (computed)
- ✅ round_3: HIT (cached)
```

## 💡 Best Practices

1. **Keep cache ON** for normal use (default behavior)
2. **Use force refresh** when:
   - Testing new prompt versions
   - Want to verify results are still valid
   - Debugging unexpected behavior
3. **Clear old cache** monthly to save disk space
4. **Monitor cache size** in statistics expander

## 🐛 Troubleshooting

**Cache not working?**
1. Check "Use cached results" is enabled
2. Verify paper text hasn't changed
3. Check personas/weights are identical

**Wrong results?**
1. Click "Clear Cache" button
2. Enable "Force refresh"
3. Rerun evaluation

**Disk space issues?**
1. Open "Cache Statistics" expander
2. Check total size
3. Click "Clear Old Cache (>30 days)"

## 📊 Cache Performance

| Operation | Time |
|-----------|------|
| Compute cache key | <1ms |
| Save round | ~5ms |
| Load round | ~3ms |
| Check status | ~1ms |

## 🔗 More Information

See full documentation: `docs/caching.md`
