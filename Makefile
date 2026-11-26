# Convenience wrapper around cargo, mainly for testing frameworks such as
# OpenBench that build engines with `make EXE=<output path>`.
EXE ?= tabx

.PHONY: all clean

all:
	cargo rustc --release -- -C target-cpu=native --emit link=$(EXE)

clean:
	cargo clean
