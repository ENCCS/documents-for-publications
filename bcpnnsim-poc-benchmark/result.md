# BCPNNSIM2.0 Profiling on LUMI

**trnpat = 1000
tenpat = 1000**


## profiling from HIP trace

the following statistics are present based on https://github.com/anderslan/BCPNNSim2.0P/tree/master :

| Name | Calls | TotalDurationNs | AverageNs | Percentage |
|------|-------|-----------------|-----------|------------|
|hipDeviceSynchronize|530384|106813774780|201389|65.16823577913857|
|hipMemcpy|120002|50178765102|418149|30.614605672427167|
|hipMemset|261657|3300150684|12612|2.0134575182326255|
|hipLaunchKernel|1339759|3286295255|2452|2.005004171594958|



the following statistics are present based on https://github.com/anderslan/BCPNNSim2.0P/tree/qiang :

| Name | Calls | TotalDurationNs | AverageNs | Percentage |
|------|-------|-----------------|-----------|------------|
|hipMemcpy|441800|112820927458|255366|64.16667943376588|
|hipDeviceSynchronize|530384|51563543826|97219|29.326663604884825|
|hipLaunchKernel|1339759|4141509871|3091|2.3554755509625114|
|hipMemset|261657|3751161230|14336|2.133465533151077|
|hipMalloc|268205|1857691342|6926|1.056558277925625|
|hipFree|172165|1365793853|7933|0.776792553586162|


## profiling on each subroutine

the following statistics are present based on https://github.com/anderslan/BCPNNSim2.0P/tree/master :

| Name | Calls | TotalDurationNs | AverageNs | Percentage |
|------|-------|-----------------|-----------|------------|
|void rocblas_gemvt_kernel|888000|46485793655|52348|40.98087539732446|
|updtrcjip_kernel|48000|38836029619|809083|34.23700803206263|
|BCPupdbwC_kernel|5630|16560106091|2941404|14.599033186749725|
|fullnorm_kernel|72000|6541191720|90849|5.766573865916939|
|upddenact_kernel|24000|1208550061|50356|1.0654317279993932|
|updtrcizp_kernel|48000|973377114|20278|0.8581083184141945|
|updMIsc_kernelv2|625|774828002|1239724|0.6830716936873145|



the following statistics are present based on https://github.com/anderslan/BCPNNSim2.0P/tree/qiang :

| Name | Calls | TotalDurationNs | AverageNs | Percentage |
|------|-------|-----------------|-----------|------------|
|void rocblas_gemvt_kernel|888000|41168743820|46361|75.42773741626934|
|fullnorm_kernel|72000|6413675850|89078|11.750882174156853|
|updtrcjip_kernel|48000|1567528985|32656|2.8719643521290035|
|BCPupdbwC_kernel|5630|1413629835|251088|2.589996441581592|
|upddenact_kernel|24000|1194936295|49789|2.1893148229760526|
|updMIsc_kernelv2|625|717474015|1147958|1.3145273959057733|
|updtrcizp_kernel|48000|687825730|14329|1.2602069856061442|

