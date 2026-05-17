# TWARDIS vs Tierpsy comparison output data

This folder contains output data used for the comparison between TWARDIS and Tierpsy tracking/feature extraction presented in the paper.

## Files

The folder `twarids_data/` contains one zip file per experimental condition:

- `food_twardis_data.zip`
- `nofood_twardis_data.zip`
- `thick_twardis_data.zip`

where the condition `food` refers to worms crawling on an agar plate covered with a thin, fresh layer of OP50 bacteria, `nofood` refers to worms crawling on an empty agar plate, and `thick` refers to worms crawling on a plate covered with a thick layer of OP50 bacteria (seede, then incubated at 37C for 24h).

Each zip contains one CSV file per selected video. The columns are described in https://github.com/lillyguisnet/TWARDISv0.1?tab=readme-ov-file.

The folder `tierpsy_data/` contains the analysis for the same videos, but done by the Tierpsy tracker. Only the `_featuresN` and `skeletons` files from Tierpsy are provided. Please refer to Tierpsy documentation for column descriptions https://github.com/Tierpsy/tierpsy-tracker. 
