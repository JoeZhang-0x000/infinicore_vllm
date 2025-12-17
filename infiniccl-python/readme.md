## Install
```
pip install -e .
```
## Test
```
python example/test.py
```

Expected Output:
```
============================================================
test all reduce op CCLOP.MAX
local max: 1.394141674041748
local max: 1.975557565689087
local max: 1.8150972127914429
local max: 1.6108579635620117
local max: 2.1035311222076416
local max: 1.622300148010254
local max: 1.3384374380111694
local max: 1.9998770952224731
global max: 2.1035311222076416
============================================================
test all reduce op CCLOP.MIN
local min: -1.6415112018585205
local min: -0.9000676870346069
local min: -2.666456937789917
local min: -1.1266416311264038
local min: -2.1200790405273438
local min: -1.1276723146438599
local min: -0.18165776133537292
local min: -1.5238943099975586
global min: -2.666456937789917
============================================================
test all reduce op CCLOP.SUM
local sum: 4.576282501220703
local sum: -0.20976495742797852
local sum: 7.356270790100098
local sum: 1.259900450706482
local sum: -0.7814431190490723
local sum: 2.495176315307617
local sum: 2.1513943672180176
local sum: 4.2291412353515625
global sum: 21.076955795288086
```

