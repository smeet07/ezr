# Define variables
TMP_DIR=~/tmp
MEASUREMENT_FILES=data/optimize/config/SS-T.csv data/optimize/config/Apache_AllMeasurements.csv data/optimize/config/HSMGP_num.csv data/optimize/config/SQL_AllMeasurements.csv data/optimize/config/SS-J.csv data/optimize/config/SS-K.csv data/optimize/config/SS-L.csv data/optimize/config/SS-M.csv data/optimize/config/SS-N.csv data/optimize/config/SS-O.csv data/optimize/config/SS-P.csv data/optimize/config/SS-Q.csv data/optimize/config/SS-R.csv data/optimize/config/SS-S.csv data/optimize/config/SS-U.csv data/optimize/config/SS-V.csv data/optimize/config/SS-W.csv data/optimize/config/SS-X.csv data/optimize/config/X264_AllMeasurements.csv data/optimize/config/rs-6d-c3_obj1.csv data/optimize/config/rs-6d-c3_obj2.csv data/optimize/config/sol-6d-c2-obj1.csv data/optimize/config/wc-6d-c1-obj1.csv
SCRIPT=python3.13 -B extend3.py -t

# Define the action to perform (branch in this case)
Act ?= branch

# Define the target for running experiments (actb4)
actb4:
	mkdir -p $(TMP_DIR)/$(Act)
	rm -f $(TMP_DIR)/$(Act)/*
	@$(foreach file,$(MEASUREMENT_FILES), \
		$(SCRIPT) $(file) | tee $(TMP_DIR)/$(Act)/$(notdir $(file)) & \
	)

# Target for generating a todo file
todo:
	@echo "Generating output files for each measurement..."
	make Act=$(Act) actb4 > $(TMP_DIR)/$(Act).sh

# New target for storing the outputs separately
outputs:
	mkdir -p $(TMP_DIR)/$(Act)/outputs
	@$(foreach file,$(MEASUREMENT_FILES), \
		$(SCRIPT) $(file) | tee $(TMP_DIR)/$(Act)/outputs/$(notdir $(file)) & \
	)

