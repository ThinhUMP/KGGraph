#!/bin/bash

flake8 . --count --max-complexity=13 --max-line-length=90 \
	--per-file-ignores="__init__.py:F401 " \
	--statistics
#!/bin/bash

flake8 . --count --max-complexity=13 --max-line-length=90 \
	--per-file-ignores="__init__.py:F401 " \
	--statistics