// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts -allow-unregistered-dialect | FileCheck %s

// The assume is asserted (unconditionally) inside the circuit function,
// below the property scope, so it persists across timesteps. The assert is
// encoded in the loop body, negated and conjoined with its enable, inside
// the per-step scope.

// CHECK-LABEL: func.func @test_bmc_props() -> i1
// CHECK: scf.for
// CHECK:   func.call @bmc_loop(
// CHECK:   [[CIRCUIT:%.+]]:3 = func.call @bmc_circuit(
// CHECK:   [[LEAFP:%.+]] = smt.eq [[CIRCUIT]]#1, {{%.+}} : !smt.bv<1>
// CHECK:   [[LEAFEN:%.+]] = smt.eq [[CIRCUIT]]#2, {{%.+}} : !smt.bv<1>
// CHECK:   [[NOT:%.+]] = smt.not [[LEAFP]]
// CHECK:   [[VIOL:%.+]] = smt.and [[LEAFEN]], [[NOT]]
// CHECK:   smt.assert [[VIOL]]
// CHECK:   smt.check
// CHECK:   smt.pop 1

// CHECK-LABEL: func.func @bmc_circuit
// CHECK: [[ASSUMEP:%.+]] = smt.eq
// CHECK: smt.assert [[ASSUMEP]]
// CHECK: smt.push 1
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
