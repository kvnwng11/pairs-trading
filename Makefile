.PHONY: build state

build:
	pip3 install -r requirements.txt

state:
	find state -type f -not -name "*template*" -delete

bt:
	find state -type f -name "*-BT.csv" -delete
