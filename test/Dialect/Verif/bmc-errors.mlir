// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @below {{circuit region must yield a leaf value for each properties region leaf argument}}
verif.bmc bound 4 num_regs 0 initial_values []
init {}
loop {}
circuit {
^bb0(%a: i1):
  verif.yield
}
properties {
^bb0(%l: i1):
  verif.assert %l : i1
}

// -----

// expected-error @below {{properties region leaf argument types must match the circuit region's yielded leaf types (yielded between outputs and register next-state values)}}
verif.bmc bound 4 num_regs 0 initial_values []
init {}
loop {}
circuit {
^bb0(%a: i32):
  verif.yield %a : i32
}
properties {
^bb0(%l: i1):
  verif.assert %l : i1
}

// -----

// expected-error @below {{properties region must not yield any values}}
verif.bmc bound 4 num_regs 0 initial_values []
init {}
loop {}
circuit {
^bb0(%a: i1):
  verif.yield %a : i1
}
properties {
^bb0(%l: i1):
  verif.assert %l : i1
  verif.yield %l : i1
}
