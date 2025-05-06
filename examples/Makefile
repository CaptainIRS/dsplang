HEXAGON = ${HEXAGON_SDK_ROOT}
HEXAGON_TOOLS = ${DEFAULT_HEXAGON_TOOLS_ROOT}
HEXAGON_ISA = v73
HEXAGON_TOOLS_VERSION = v88
LLVM_ROOT = ${PWD}/../../llvm
DSPLANG_BUILD_ROOT = ${PWD}/../build
IMPL = dsp
EXP = basic

build/$(EXP):
	mkdir -p build/$(EXP)

build/$(EXP)/$(EXP)$(IMPL).o: build/$(EXP) $(EXP)/lib.$(IMPL)
	@if [ "$(IMPL)" = "c" ]; then \
		echo "Compiling C code"; \
		$(LLVM_ROOT)/bin/hexagon-unknown-none-elf-clang \
			-fvectorize \
			-O3 \
			-c $(EXP)/lib.$(IMPL) \
			-o build/$(EXP)/$(EXP)$(IMPL).o \
			-I$(HEXAGON_TOOLS)/Tools/target/hexagon/include/; \
	else \
		echo "Compiling DSP code"; \
		$(DSPLANG_BUILD_ROOT)/bin/dsplangc \
			-o build/$(EXP)/$(EXP)$(IMPL).o \
			$(EXP)/lib.$(IMPL); \
	fi

dumpir:
	$(DSPLANG_BUILD_ROOT)/bin/dsplangc \
		-o build/$(EXP)dsp.o \
		$(EXP)/lib.dsp \
		--mlir-print-ir-after-all

build/$(EXP)/main.o: build/$(EXP) $(EXP)/main.c
	$(LLVM_ROOT)/bin/hexagon-unknown-none-elf-clang \
		-c $(EXP)/main.c \
		-o build/$(EXP)/main.o \
		-I$(HEXAGON_TOOLS)/Tools/target/hexagon/include/

build/$(EXP)/$(EXP)$(IMPL).so: build/$(EXP) build/$(EXP)/$(EXP)$(IMPL).o build/$(EXP)/main.o
	$(LLVM_ROOT)/bin/hexagon-unknown-none-elf-clang \
		-m$(HEXAGON_ISA) \
		-G0 \
		-fpic \
		-Wl,-Bsymbolic \
		-shared \
		-o build/$(EXP)/$(EXP)$(IMPL).so \
		-Wl,-soname,build/$(EXP)/$(EXP)$(IMPL).so \
		-Wl,--start-group \
		build/$(EXP)/$(EXP)$(IMPL).o \
		build/$(EXP)/main.o \
		-Wl,--end-group \
		-lc \
		-lhexagon \
		-I$(HEXAGON_TOOLS)/Tools/target/hexagon/include/ \
		-L$(HEXAGON_TOOLS)/Tools/target/hexagon/lib/$(HEXAGON_ISA)/G0/pic/ \
		-fuse-ld=eld

build/$(EXP)/libhexagon.so: build/$(EXP)
	cp \
		$(HEXAGON_TOOLS)/Tools/target/hexagon/lib/$(HEXAGON_ISA)/G0/pic/libhexagon.so \
		build/$(EXP)/libhexagon.so

build/$(EXP)/osam.cfg: build/$(EXP)
	echo "$(HEXAGON)/rtos/qurt/compute$(HEXAGON_ISA)/debugger/lnx64/qurt_model.so" > build/$(EXP)/osam.cfg

build/$(EXP)/pmu_stats.txt: build/$(EXP)
	touch build/$(EXP)/pmu_stats.txt

build/$(EXP)/q6ss.cfg: build/$(EXP)
	echo "$(HEXAGON_TOOLS)/Tools/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" > build/$(EXP)/q6ss.cfg
	echo "$(HEXAGON_TOOLS)/Tools/lib/iss/l2vic.so 32 0xFC910000" >> build/$(EXP)/q6ss.cfg

sim: build/$(EXP)/$(EXP)$(IMPL).so build/$(EXP)/libhexagon.so build/$(EXP)/pmu_stats.txt build/$(EXP)/q6ss.cfg build/$(EXP)/osam.cfg
	cd build/$(EXP); \
	$(HEXAGON_TOOLS)/Tools/bin/hexagon-sim \
		-m$(HEXAGON_ISA)na_1 \
		--simulated_returnval \
		--usefs ${PWD} \
		--pmu_statsfile pmu_stats.txt \
		--cosim_file q6ss.cfg \
		--l2tcm_base 0xd800 \
		--rtos osam.cfg \
		$(HEXAGON)/rtos/qurt/compute$(HEXAGON_ISA)/sdksim_bin/runelf.pbn \
		-- \
		$(HEXAGON)/libs/run_main_on_hexagon/ship/hexagon_tool$(HEXAGON_TOOLS_VERSION)_$(HEXAGON_ISA)/run_main_on_hexagon_sim \
		-- \
		$(EXP)$(IMPL).so

qemu: build/$(EXP)/$(EXP)$(IMPL).so build/$(EXP)/libhexagon.so build/$(EXP)/pmu_stats.txt build/$(EXP)/q6ss.cfg build/$(EXP)/osam.cfg
	cd build/$(EXP); \
	$(LLVM_ROOT)/bin/qemu-system-hexagon \
		-kernel $(HEXAGON)/rtos/qurt/compute$(HEXAGON_ISA)/sdksim_bin/runelf.pbn \
		-append "$(HEXAGON)/libs/run_main_on_hexagon/ship/hexagon_tool$(HEXAGON_TOOLS_VERSION)_$(HEXAGON_ISA)/run_main_on_hexagon_sim -- ${PWD}/build/$(EXP)/$(EXP)$(IMPL).so"

dis: build/$(EXP)/$(EXP)$(IMPL).so
	$(HEXAGON_TOOLS)/Tools/bin/hexagon-llvm-objdump \
		--disassemble-symbols=dsplanglib \
		--section=.text \
		--m$(HEXAGON_ISA) \
		--mhvx \
		--mhvx-ieee-fp \
		--disassemble \
		--demangle \
		--no-show-raw-insn \
		build/$(EXP)/$(EXP)$(IMPL).so

clean:
	rm -rf \
		build
