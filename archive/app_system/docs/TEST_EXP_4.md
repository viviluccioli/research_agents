# Testing Experiment 4 Fixes

## Quick Start

Run the Experiment 4 app:
```bash
cd /casl/home/m1vcl00/FS-CASL/research_agents/app_system
source ../venv/bin/activate
streamlit run app_exp_4.py
```

Or use the script:
```bash
cd /casl/home/m1vcl00/FS-CASL/research_agents/app_system
bash run_app_exp_4.sh
```

## What to Test

### 1. Persona Selection (Round 0)
- Upload a paper or paste text
- Click "Run Debate"
- **Expected**: Round 0 should display 3 selected personas from the valid list:
  - Theorist, Econometrician, ML_Expert, Data_Scientist, CS_Expert
  - Historian, Visionary, Policymaker, Ethicist, Perspective
- **Check logs**: Look for any `[WARNING]` messages about invalid personas
- **Previous bug**: Would fail with KeyError: 'Mathematician'

### 2. Full Debate Execution
- **Expected**: All rounds should complete without errors
  - Round 0: Persona Selection
  - Round 1: Independent Evaluation
  - Round 2A: Cross-Examination
  - Round 2B: Direct Examination
  - Round 2C: Final Amendments
  - Round 3: Editor Decision
- **Check**: All selected personas should have reports displayed with correct icons

### 3. Paper Type Selection
- **Expected**: Paper type is now OPTIONAL for Referee Report
- **Note**: The UI should clarify that paper type is only needed for Section Evaluator
- **Test**: Try running a debate WITHOUT selecting paper type - it should work fine

### 4. Icon Display
- **Expected**: Each persona should have a unique emoji icon:
  - 🔢 Theorist
  - 📊 Econometrician
  - 🤖 ML_Expert
  - 📈 Data_Scientist
  - 💻 CS_Expert
  - 📚 Historian
  - 🚀 Visionary
  - 🏛️ Policymaker
  - ⚖️ Ethicist
  - 🌍 Perspective
- **Previous bug**: Icons for old personas (Mathematician, Empiricist) were undefined

## Common Issues & Solutions

### Issue: "KeyError: 'Mathematician'" or similar
**Cause**: LLM selected an invalid persona name
**Solution**: The new validation should catch this and either:
- Filter out invalid personas and continue with valid ones
- Fall back to default personas (Econometrician, ML_Expert, Policymaker)
**Check logs**: You should see `[WARNING]` messages explaining what happened

### Issue: Wrong number of personas selected
**Cause**: LLM returned more or fewer than 3 personas
**Solution**: The system will log a warning but proceed if it has valid personas
**Check logs**: Look for `[WARNING] Requested 3 personas, but LLM returned X`

### Issue: Weights don't sum to 1.0
**Cause**: Invalid personas were filtered out
**Solution**: The system automatically renormalizes weights
**Check output**: Verify that displayed weights sum to 1.0

## Monitoring

Check the terminal/console output for these log messages:

```
[Round 0] Starting persona selection (N=3)...
[Round 0] Selected personas: ['Econometrician', 'ML_Expert', 'Policymaker']
[Round 0] Weights: {'Econometrician': 0.4, 'ML_Expert': 0.35, 'Policymaker': 0.25}
[Round 1] Independent evaluations starting...
[Round 2A] Cross-examination starting...
[Round 2B] Direct examination starting...
[Round 2C] Final amendments starting...
[Round 3] Editor decision starting...
```

If you see `[WARNING]` messages, investigate:
- `[WARNING] LLM selected invalid personas: ['Mathematician']`
- `[WARNING] Valid personas are: [...]`
- `[WARNING] Only X valid personas, need 3. Using defaults.`

## Expected Behavior

### With the Fixes
✅ Debate completes successfully
✅ All 3 personas are from the valid list
✅ Display shows correct icons
✅ No KeyErrors
✅ Paper type is optional

### Without the Fixes (Previous Behavior)
❌ KeyError: 'Mathematician' during display
❌ Missing icons for selected personas
❌ Paper type was required

## Test Cases

1. **Empirical Paper**: Should select Econometrician, Data_Scientist, ML_Expert
2. **Theoretical Paper**: Should select Theorist, Econometrician, Visionary
3. **Policy Paper**: Should select Policymaker, Econometrician, Perspective
4. **ML/AI Paper**: Should select ML_Expert, CS_Expert, Data_Scientist
5. **Ethics Paper**: Should select Ethicist, Perspective, Policymaker

Try uploading different types of papers and verify that persona selection makes sense.

## Rollback

If issues persist, you can revert to the previous version:
```bash
cd /casl/home/m1vcl00/FS-CASL/research_agents/app_system
git diff referee/engine_exp_4.py
git diff referee/workflow_exp_4.py
git diff app_exp_4.py
```

## Report Issues

If you encounter errors, capture:
1. The console/terminal output (especially Round 0 logs)
2. The error message and stack trace
3. The paper type/content you were testing with
4. Whether paper type was selected or not
