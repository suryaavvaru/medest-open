#!/usr/bin/env bash
set -e
mkdir -p data/raw/maccrobat
cat > data/raw/maccrobat/toy_case_001.txt <<'EOF'
The patient presented with fever and cough. Tuberculosis was considered in the differential diagnosis.
Initial sputum smear was negative for tuberculosis.
After three days, culture grew Mycobacterium tuberculosis, confirming tuberculosis.
The patient was treated with isoniazid and rifampin.
At follow-up, symptoms improved and the infection resolved.
EOF
cat > data/raw/maccrobat/toy_case_001.ann <<'EOF'
T1 Disease 44 56 tuberculosis
T2 Disease 125 137 tuberculosis
T3 Disease 212 224 tuberculosis
T4 Medication 264 273 isoniazid
T5 Medication 278 286 rifampin
T6 Disease 332 341 infection
EOF
cat > data/raw/maccrobat/toy_case_002.txt <<'EOF'
Pulmonary embolism could not be excluded at presentation.
CT pulmonary angiography was negative for pulmonary embolism.
Pneumonia was suspected because of fever and consolidation.
The findings were consistent with bacterial pneumonia.
Antibiotics were started and the pneumonia improved.
EOF
cat > data/raw/maccrobat/toy_case_002.ann <<'EOF'
T1 Disease 0 18 Pulmonary embolism
T2 Disease 100 118 pulmonary embolism
T3 Disease 120 129 Pneumonia
T4 Disease 209 228 bacterial pneumonia
T5 Disease 264 273 pneumonia
EOF
echo '[OK] Toy MACCROBAT-style data created.'
