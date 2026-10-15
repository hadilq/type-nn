.PHONY: all clean

# Rocq >= 9 with its stdlib package (rocq-stdlib; in nixpkgs
# coqPackages.stdlib). The sources use the `From Stdlib` prefix.
COQC ?= coqc
QFLAGS = -Q TypeNN TypeNN

VO = TypeNN/Basic.vo TypeNN/Forward.vo TypeNN/Backward.vo TypeNN/Surgery.vo \
     TypeNN/Readout.vo TypeNN/SignedPower.vo TypeNN/Threshold.vo TypeNN/Graph.vo

all: $(VO)

TypeNN/Forward.vo TypeNN/Backward.vo TypeNN/Surgery.vo: TypeNN/Basic.vo
TypeNN/SignedPower.vo: TypeNN/Readout.vo
TypeNN/Graph.vo: TypeNN/Basic.vo TypeNN/Surgery.vo

%.vo: %.v
	$(COQC) $(QFLAGS) $<

clean:
	rm -f TypeNN/*.vo TypeNN/*.glob TypeNN/*.vok TypeNN/*.vos TypeNN/.*.aux .lia.cache
