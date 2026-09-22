.PHONY: all clean

# Rocq >= 9 with its stdlib package (rocq-stdlib; in nixpkgs
# coqPackages.stdlib). The sources use the `From Stdlib` prefix.
COQC ?= coqc
QFLAGS = -Q TypeNN TypeNN

VO = TypeNN/Basic.vo TypeNN/Forward.vo TypeNN/Backward.vo TypeNN/Surgery.vo \
     TypeNN/Readout.vo TypeNN/Threshold.vo

all: $(VO)

TypeNN/Forward.vo TypeNN/Backward.vo TypeNN/Surgery.vo: TypeNN/Basic.vo

%.vo: %.v
	$(COQC) $(QFLAGS) $<

clean:
	rm -f TypeNN/*.vo TypeNN/*.glob TypeNN/*.vok TypeNN/*.vos TypeNN/.*.aux .lia.cache
