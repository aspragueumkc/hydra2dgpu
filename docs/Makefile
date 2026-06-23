# HYDRA 2D GPU — Documentation
#
# Build API reference docs using pdoc.
# Usage: make api   (or just: make)
#
# Prerequisites:
#   pip install pdoc

default: api

# Generate API docs for the swe2d Python package
api:
	pdoc ../swe2d \
		--skip-errors \
		--docformat numpy \
		--logo "HYDRA 2D GPU" \
		--logo-link "https://github.com/aspragueumkc/hydra2dgpu" \
		--footer-text "HYDRA 2D GPU — GPU-Accelerated 2D Shallow Water Equation Plugin" \
		--output-dir _build/html

# Generate C++/CUDA API docs using Doxygen
cpp-api:
	doxygen Doxyfile

# Build all docs (Python API + C++ API)
all: api cpp-api

# Clean build artifacts
clean:
	rm -rf _build/
	rm -rf cpp/api/

# Serve locally for preview
serve: api
	python3 -m http.server 8080 --directory _build/html

.PHONY: api cpp-api all clean serve
