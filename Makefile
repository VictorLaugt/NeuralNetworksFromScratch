DATASET_DIR = dataset

all: extract_mnist

extract_mnist:
	tar -xvf $(DATASET_DIR)/mnist.tgz -C $(DATASET_DIR)

clean:
	find . -name __pycache__ -type d | while read -r pycachepath; do rm -rf $$pycachepath; done
	rm -f $(DATASET_DIR)/*.npy

.PHONY: all extract_mnist clean
