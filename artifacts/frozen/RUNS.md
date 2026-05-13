# Run Ledger

Submission commit: d5b4f75cea4c632dae8887a46d4ac64e47c48d21. Per-row git_sha fields record `unknown` because benchmark runs executed outside a git checkout on the compute pod; submission-level traceability is provided by this commit hash, configuration hashes, the run ledger, and checksums.

| run_id | git_sha | config_hash | JSON path | model | optimization | perturbation | one-line result |
|---|---|---|---|---|---|---|---|
| 20260510T170525846283Z_test_5fcc03777218 | unknown | a97ab269115d | outputs/benchmarks/20260510T170525846283Z_test_5fcc03777218.json | segresnet | baseline | clean | dice=0.0021, latency=5.4599s |
| 20260510T171027533732Z_test_100839537bb1 | unknown | 924cc9fcd209 | outputs/benchmarks/20260510T171027533732Z_test_100839537bb1.json | segresnet | amp | clean | dice=0.0021, latency=5.4647s |
| 20260510T171411035509Z_test_fc0d134ea328 | unknown | 4b8a8a97a7ed | outputs/benchmarks/20260510T171411035509Z_test_fc0d134ea328.json | segresnet | data_pipeline | clean | dice=0.0021, latency=5.4588s |
| 20260510T171443556054Z_test_2bd72fba118a | unknown | b7b92e0a9afa | outputs/benchmarks/20260510T171443556054Z_test_2bd72fba118a.json | segresnet | amp | clean | dice=0.0021, latency=5.5673s |
| 20260510T172114961275Z_test_100839537bb1 | unknown | 924cc9fcd209 | outputs/benchmarks/20260510T172114961275Z_test_100839537bb1.json | segresnet | amp | clean | dice=0.0021, latency=5.4664s |
| 20260510T172115230536Z_test_fc0d134ea328 | unknown | 4b8a8a97a7ed | outputs/benchmarks/20260510T172115230536Z_test_fc0d134ea328.json | segresnet | data_pipeline | clean | dice=0.0021, latency=5.4674s |
| 20260510T172304122408Z_test_2bd72fba118a | unknown | b7b92e0a9afa | outputs/benchmarks/20260510T172304122408Z_test_2bd72fba118a.json | segresnet | amp | clean | dice=0.0021, latency=5.5667s |
| 20260510T172308920726Z_test_c96c2e833f19 | unknown | a0dcb765fade | outputs/benchmarks/20260510T172308920726Z_test_c96c2e833f19.json | segresnet | compile | clean | dice=0.0021, latency=3.6849s |
| 20260510T172450171026Z_test_56eec198ecf2 | unknown | 0a7e12ec55f8 | outputs/benchmarks/20260510T172450171026Z_test_56eec198ecf2.json | segresnet | amp | clean | dice=0.0022, latency=6.1787s |
| 20260510T172506568802Z_test_f1e204d89534 | unknown | e40bd92cc770 | outputs/benchmarks/20260510T172506568802Z_test_f1e204d89534.json | segresnet | all | clean | dice=0.0022, latency=3.3463s |
| robust_20260510T172857831474Z_4b01777a1395 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T172857831474Z_4b01777a1395.json | segresnet | baseline | clean | dice=0.0021, latency=5.4648s |
| robust_20260510T173344949910Z_4b01777a1395 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T173344949910Z_4b01777a1395.json | segresnet | baseline | clean | dice=0.0021, latency=5.4593s |
| robust_20260510T173434361141Z_32b7d440b2f1 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T173434361141Z_32b7d440b2f1.json | segresnet | baseline | downsample_resample:1 | dice=0.0021, latency=5.4643s |
| robust_20260510T173710687782Z_32b7d440b2f1 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T173710687782Z_32b7d440b2f1.json | segresnet | baseline | downsample_resample:1 | dice=0.0008, latency=4.6077s |
| robust_20260510T173713430944Z_4b01777a1395 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T173713430944Z_4b01777a1395.json | segresnet | baseline | clean | dice=0.0008, latency=4.5941s |
| robust_20260510T173801638736Z_57f2b728970a | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T173801638736Z_57f2b728970a.json | segresnet | baseline | gaussian_noise:1 | dice=0.0008, latency=4.4347s |
| robust_20260510T173810613385Z_a73e53b2ca63 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T173810613385Z_a73e53b2ca63.json | segresnet | baseline | downsample_resample:2 | dice=0.0008, latency=4.4408s |
| robust_20260510T173855190221Z_23f11fdbcf47 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T173855190221Z_23f11fdbcf47.json | segresnet | baseline | gaussian_noise:2 | dice=0.0008, latency=4.4357s |
| robust_20260510T173906011681Z_3862a29640bb | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T173906011681Z_3862a29640bb.json | segresnet | baseline | downsample_resample:3 | dice=0.0008, latency=4.4409s |
| robust_20260510T173943661985Z_37498d3c931c | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T173943661985Z_37498d3c931c.json | segresnet | baseline | gaussian_noise:3 | dice=0.0008, latency=4.4343s |
| robust_20260510T174001933302Z_e92d4b21e6c6 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T174001933302Z_e92d4b21e6c6.json | segresnet | baseline | intensity_shift:1 | dice=0.0004, latency=4.4400s |
| robust_20260510T174032988933Z_a16c4bfe98a7 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T174032988933Z_a16c4bfe98a7.json | segresnet | baseline | gaussian_blur:1 | dice=0.0008, latency=4.4355s |
| robust_20260510T174120609930Z_673152417b7d | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T174120609930Z_673152417b7d.json | segresnet | baseline | intensity_shift:2 | dice=0.0002, latency=4.4411s |
| robust_20260510T174126532409Z_46d74c5bda69 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T174126532409Z_46d74c5bda69.json | segresnet | baseline | gaussian_blur:2 | dice=0.0008, latency=4.4351s |
| robust_20260510T174219562416Z_88545c95e56c | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T174219562416Z_88545c95e56c.json | segresnet | baseline | gaussian_blur:3 | dice=0.0007, latency=4.4347s |
| robust_20260510T174227036104Z_f82e4bb15ac4 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T174227036104Z_f82e4bb15ac4.json | segresnet | baseline | intensity_shift:3 | dice=0.0001, latency=4.4411s |
| robust_20260510T174311876901Z_cb92ee3d26f5 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T174311876901Z_cb92ee3d26f5.json | segresnet | baseline | contrast_shift:1 | dice=0.0008, latency=4.4407s |
| robust_20260510T174357326157Z_bb165616155b | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T174357326157Z_bb165616155b.json | segresnet | baseline | contrast_shift:2 | dice=0.0008, latency=4.4407s |
| robust_20260510T174441337346Z_3602806f4880 | unknown | a97ab269115d | outputs/benchmarks/robust_20260510T174441337346Z_3602806f4880.json | segresnet | baseline | contrast_shift:3 | dice=0.0008, latency=4.5286s |
| robust_20260510T174523715012Z_49e25c8f82d1 | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T174523715012Z_49e25c8f82d1.json | segresnet | all | clean | dice=0.0008, latency=14.1487s |
| robust_20260510T174607486461Z_df2ee60559e9 | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T174607486461Z_df2ee60559e9.json | segresnet | all | downsample_resample:1 | dice=0.0008, latency=14.6270s |
| robust_20260510T174619635297Z_57b4b772a6c3 | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T174619635297Z_57b4b772a6c3.json | segresnet | all | gaussian_noise:1 | dice=0.0008, latency=9.2166s |
| robust_20260510T174650450081Z_49e25c8f82d1 | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T174650450081Z_49e25c8f82d1.json | segresnet | all | clean | dice=0.0008, latency=14.2403s |
| robust_20260510T174716808907Z_99155c71e8f8 | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T174716808907Z_99155c71e8f8.json | segresnet | all | downsample_resample:2 | dice=0.0008, latency=8.8206s |
| robust_20260510T174753326333Z_57b4b772a6c3 | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T174753326333Z_57b4b772a6c3.json | segresnet | all | gaussian_noise:1 | dice=0.0008, latency=8.6972s |
| robust_20260510T174816585969Z_cbc33cd4779b | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T174816585969Z_cbc33cd4779b.json | segresnet | all | downsample_resample:3 | dice=0.0008, latency=8.9246s |
| robust_20260510T174844952029Z_8c761ac82e47 | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T174844952029Z_8c761ac82e47.json | segresnet | all | gaussian_noise:2 | dice=0.0008, latency=8.0391s |
| robust_20260510T174918295085Z_29694548805c | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T174918295085Z_29694548805c.json | segresnet | all | intensity_shift:1 | dice=0.0004, latency=8.0589s |
| robust_20260510T174937259411Z_0ddfe71568f9 | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T174937259411Z_0ddfe71568f9.json | segresnet | all | gaussian_noise:3 | dice=0.0008, latency=7.8532s |
| robust_20260510T175020093445Z_66c699cf79ad | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T175020093445Z_66c699cf79ad.json | segresnet | all | intensity_shift:2 | dice=0.0002, latency=7.9876s |
| robust_20260510T175026986208Z_3f2d6b5528da | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T175026986208Z_3f2d6b5528da.json | segresnet | all | gaussian_blur:1 | dice=0.0008, latency=7.9899s |
| robust_20260510T175121456575Z_dd0a47229ab2 | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T175121456575Z_dd0a47229ab2.json | segresnet | all | gaussian_blur:2 | dice=0.0008, latency=8.0869s |
| robust_20260510T175122581040Z_2e06439744b5 | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T175122581040Z_2e06439744b5.json | segresnet | all | intensity_shift:3 | dice=0.0001, latency=7.9761s |
| robust_20260510T175213308222Z_47a9af189fa6 | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T175213308222Z_47a9af189fa6.json | segresnet | all | contrast_shift:1 | dice=0.0008, latency=8.4320s |
| robust_20260510T175218006963Z_15a5ff9585b9 | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T175218006963Z_15a5ff9585b9.json | segresnet | all | gaussian_blur:3 | dice=0.0007, latency=8.3329s |
| robust_20260510T175300815703Z_dce45b3ee6ce | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T175300815703Z_dce45b3ee6ce.json | segresnet | all | contrast_shift:2 | dice=0.0008, latency=8.3618s |
| robust_20260510T175348410698Z_65d22d7f3f1e | unknown | e40bd92cc770 | outputs/benchmarks/robust_20260510T175348410698Z_65d22d7f3f1e.json | segresnet | all | contrast_shift:3 | dice=0.0008, latency=8.1399s |
| unet_baseline_real_fulltest | unknown | a97ab269115d | outputs/benchmarks/unet_baseline_real_fulltest.json | unet | baseline | clean | dice=0.0032, latency=2.7401s |
