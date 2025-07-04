# Define folders
DATA_DIR=project/data
PLOT_DIR=project/plots

# Cleaned data and model training
clean-pune:
	python3 analytics.py $(DATA_DIR)/pune_raw.csv

train-models:
	python3 modelling.py

# Grouped analytics plots
grouped-analytics:
	python3 analytics_grouped.py

# Clean up plots folder
clean-plots:
	rm -f $(PLOT_DIR)/*.png

# Run everything in sequence command-> make grouped-analytics
all: clean-pune train-models grouped-analytics
