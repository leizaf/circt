// REQUIRES: libz3
// REQUIRES: circt-bmc-jit

// Multiple asserts in one module: a violation of EITHER assert must be
// reported, and two holding asserts must not introduce spurious
// counterexamples.

// RUN: circt-bmc %s -b 6 --module FirstFails --shared-libs=%libz3 | FileCheck %s --check-prefix=FIRSTFAILS
// FIRSTFAILS: Assertion can be violated!
hw.module @FirstFails(in %in: i1) {
  %true = hw.constant true
  %nin = comb.xor %in, %true : i1
  %taut = comb.or %in, %nin : i1
  verif.assert %in : i1
  verif.assert %taut : i1
}

// RUN: circt-bmc %s -b 6 --module SecondFails --shared-libs=%libz3 | FileCheck %s --check-prefix=SECONDFAILS
// SECONDFAILS: Assertion can be violated!
hw.module @SecondFails(in %in: i1) {
  %true = hw.constant true
  %nin = comb.xor %in, %true : i1
  %taut = comb.or %in, %nin : i1
  verif.assert %taut : i1
  verif.assert %in : i1
}

// RUN: circt-bmc %s -b 6 --module BothHold --shared-libs=%libz3 | FileCheck %s --check-prefix=BOTHHOLD
// BOTHHOLD: Bound reached with no violations!
hw.module @BothHold(in %in: i1) {
  %true = hw.constant true
  %nin = comb.xor %in, %true : i1
  %taut = comb.or %in, %nin : i1
  %taut2 = comb.and %taut, %true : i1
  verif.assert %taut : i1
  verif.assert %taut2 : i1
}
