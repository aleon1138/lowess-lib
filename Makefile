install:
	pip install .

format:
	astyle -A4 -S -z2 -n -j *.cc

clean:
	rm -rf build dist lowesslib.egg-info
