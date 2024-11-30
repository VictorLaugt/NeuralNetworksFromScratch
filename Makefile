DATASET_DIR = demo/mnist_dataset

all: extract_mnist

build:
	python3 setup.py sdist bdist_wheel

install_dev:
	python3 -m pip install --editable .

upload_test:
	twine upload -r testpypi "dist/*"

upload:
	twine upload "dist/*"

extract_mnist:
	tar -xvf $(DATASET_DIR)/mnist.tgz -C $(DATASET_DIR)

clean_mnist:
	rm -f $(DATASET_DIR)/*.npy

clean_pycache:
	find . -name __pycache__ -type d | while read -r pycachepath; do rm -rf $$pycachepath; done

clean_package_build:
	rm -rf build dist *.egg-info

clean: clean_mnist clean_pycache clean_package_build

.PHONY: all build install_dev upload_test upload extract_mnist clean_mnist, clean_pycache, clean_package_build clean
