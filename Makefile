install:
	sudo pip3 install -r requirements.txt
	sudo pip3 install --upgrade https://github.com/pytorch/vision/archive/master.zip

install-dev:
	sudo pip3 install --upgrade -r requirements-dev.txt

codecheck:
	-flake8 --ignore E402,W503,W504
	-mypy --python-version 3.6 --ignore-missing-imports --warn-unused-ignores --warn-unused-configs --warn-return-any --warn-redundant-casts *.py
