// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// The assume is asserted (unconditionally) inside the circuit function, so
// it persists across timesteps; nothing is ever popped. The assert is
// encoded in the loop body as a violation term (enable && !property) that
// the per-step check assumes.

// CHECK-LABEL: func.func @test_bmc_props() -> i1
// CHECK: scf.for
// CHECK:   [[CIRCUIT:%.+]]:3 = func.call @bmc_circuit(
// CHECK:   [[LEAFP:%.+]] = smt.eq [[CIRCUIT]]#1, {{%.+}} : !smt.bv<1>
// CHECK:   [[LEAFEN:%.+]] = smt.eq [[CIRCUIT]]#2, {{%.+}} : !smt.bv<1>
// CHECK:   [[NOT:%.+]] = smt.not [[LEAFP]]
// CHECK:   [[VIOL:%.+]] = smt.and [[LEAFEN]], [[NOT]]
// CHECK:   smt.check assuming([[VIOL]])
// CHECK:   func.call @bmc_loop(
// CHECK-NOT: smt.pop

// Two asserts in the region: the step's question is the disjunction of
// their violation terms, consumed by a single check.
// CHECK-LABEL: func.func @test_bmc_two_asserts() -> i1
// CHECK: scf.for
// CHECK:   [[CIRCUIT2:%.+]]:2 = func.call @bmc_circuit_0(
// CHECK:   [[EQ1:%.+]] = smt.eq [[CIRCUIT2]]#0, {{%.+}} : !smt.bv<1>
// CHECK:   [[N1:%.+]] = smt.not [[EQ1]]
// CHECK:   [[EQ2:%.+]] = smt.eq [[CIRCUIT2]]#1, {{%.+}} : !smt.bv<1>
// CHECK:   [[N2:%.+]] = smt.not [[EQ2]]
// CHECK:   [[OR:%.+]] = smt.or [[N1]], [[N2]]
// CHECK:   smt.check assuming([[OR]])

// A loop-region assume is asserted by the loop function, which runs only
// after the per-step check: it cannot constrain the current step's query.
// CHECK-LABEL: func.func @test_bmc_loop_assume() -> i1
// CHECK: scf.for
// CHECK:   smt.check assuming(
// CHECK:   func.call @bmc_loop_1(

// CHECK-LABEL: func.func @bmc_circuit
// CHECK: [[ASSUMEP:%.+]] = smt.eq
// CHECK: smt.assert [[ASSUMEP]]
// CHECK-NOT: smt.push
// CHECK: return
func.func @test_bmc_props() -> (i1) {
  %bmc = verif.bmc bound 4 num_regs 0 initial_values []
  init {
    %false = hw.constant false
    %clk = seq.to_clock %false
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    %f = seq.from_clock %clk
    %true = hw.constant true
    %n = comb.xor %f, %true : i1
    %nc = seq.to_clock %n
    verif.yield %nc : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %in: i1, %en: i1):
    %true = hw.constant true
    verif.yield %true, %in, %en : i1, i1, i1
  }
  properties {
  ^bb0(%leaf: i1, %leafEn: i1):
    verif.assert %leaf if %leafEn : i1
    verif.assume %leaf : i1
  }
  func.return %bmc : i1
}

func.func @test_bmc_two_asserts() -> (i1) {
  %bmc = verif.bmc bound 4 num_regs 0 initial_values []
  init {}
  loop {}
  circuit {
  ^bb0(%a: i1, %b: i1):
    verif.yield %a, %b : i1, i1
  }
  properties {
  ^bb0(%l1: i1, %l2: i1):
    verif.assert %l1 : i1
    verif.assert %l2 : i1
  }
  func.return %bmc : i1
}

func.func @test_bmc_loop_assume() -> (i1) {
  %bmc = verif.bmc bound 4 num_regs 0 initial_values []
  init {
    %false = hw.constant false
    %clk = seq.to_clock %false
    verif.yield %clk : !seq.clock
  }
  loop {
  ^bb0(%clk: !seq.clock):
    %true = hw.constant true
    verif.assume %true : i1
    verif.yield %clk : !seq.clock
  }
  circuit {
  ^bb0(%clk: !seq.clock, %a: i1):
    verif.yield %a : i1
  }
  properties {
  ^bb0(%l: i1):
    verif.assert %l : i1
  }
  func.return %bmc : i1
}

// CHECK-LABEL: func.func @bmc_loop_1(
// CHECK: smt.assert
