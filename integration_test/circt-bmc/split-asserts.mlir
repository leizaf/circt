// Without flattening, asserts inside instantiated modules cannot be moved
// into the BMC properties region and are rejected by LowerToBMC.
//  RUN: not circt-bmc %s -b 10 --module ModuleAsserts --shared-libs=%libz3 --flatten-modules=false 2>&1 | FileCheck %s
//  CHECK: error: property ops inside instantiated modules are not supported; run with --flatten-modules

// With flattening (the default), both asserts land in the properties region
// and either one being violable must be reported.
//  RUN: circt-bmc %s -b 10 --module ModuleAsserts --shared-libs=%libz3 | FileCheck %s --check-prefix=FLATTENED
//  FLATTENED: Assertion can be violated!

hw.module @OneAssert(in %in: i1) {
  verif.assert %in : i1
}

hw.module @ModuleAsserts(in %i0: i1, in %i1: i1) {
  hw.instance "a" @OneAssert(in: %i0: i1) -> ()
  hw.instance "b" @OneAssert(in: %i1: i1) -> ()
}
