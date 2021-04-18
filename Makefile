.PHONY: all
all: test bench doc

.PHONY: doc
doc:
	cargo doc --no-deps --features "debug arbitrary serde-serialize abomonation"

.PHONY: bench
bench:
	cargo bench --features "rand"

.PHONY: test-nalgebra
test-nalgebra:
	cargo test --features "arbitrary rand serde-serialize abomonation-serialize sparse debug io compare libm proptest-support slow-tests"

.PHONY: test
test: test-nalgebra
