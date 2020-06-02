# Time-Scale Modification Dataset with Subjective Quality Labels

<iframe width="100%" height="600" src="https://ieee-dataport.org/open-access/time-scale-modification-dataset-subjective-quality-labels/embed" frameborder="0" class="embed-textarea" allowfullscreen="true" webkitallowfullscreen="true" mozallowfullscreen="true"></iframe>

The dataset has been released under the Creative Commons Attribution International (CC BY 4.0) license.

The training subset contains 88 source files time-scaled by 6 methods at 10 different time-scales.
- Phase Vocoder (PV)
- Identity Phase Locking Phase Vocoder (IPL)
- Fuzzy Epoch Synchronous Overlap-Add (FESOLA)
- Waveform Similarity Overlap-Add (WSOLA)
- Harmonic Percussive Separation Time-Scale Modification (HPTSM)
- (Subjective version) Mel-scale sub-band modelling (uTVS)

The testing subset contains 20 source files time-scaled by 3 methods at 4 different time-scales.
- Elastique
- Phase Vocoder with Fuzzy Classification of Spectral Bins (FuzzyPV)
- Non-Negative Matrix Factorization Time-Scale Modification (NMFTSM)

The unlabeled evaluation subset contains the 20 testing source files, processed at 20 time-scales ratios from 0.2 to 2.
It uses the methods listed above in addition to:
- Scaled Phase Locking Phase Vocoder (SPL)
- Phavorit Identity Phase Locking Phase Vocoder (PhIPL)
- Phavorit Scaled Phase Locking Phase Vocoder (PhSPL)
- Identity Phase Locking Phase Vocoder by Drideger (DrIPL)
- Epoch Synchronous Overlap-Add (ESOLA)
- Mel-scale sub-band modelling (uTVS Updated Version)
  - uTVS is the fixed version, uTVSSubj is the version used in subjective testing


The associated paper is currently under review.  Full results will be made available on acceptance of the paper.

The website used for subjective testing can be found at http://www.timrobertssound.com.au/TSM/index.html
