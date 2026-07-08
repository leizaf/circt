//===- LowerToBMC.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
#define GEN_PASS_DEF_LOWERTOBMC
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

//===----------------------------------------------------------------------===//
// Convert Lower To BMC pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerToBMCPass : public circt::impl::LowerToBMCBase<LowerToBMCPass> {
  using LowerToBMCBase::LowerToBMCBase;
  void runOnOperation() override;
};
} // namespace

void LowerToBMCPass::runOnOperation() {
  Namespace names;
  // Fetch the 'hw.module' operation to model check.
  auto moduleOp = getOperation();
  auto hwModule = moduleOp.lookupSymbol<hw::HWModuleOp>(topModule);
  if (!hwModule) {
    moduleOp.emitError("hw.module named '") << topModule << "' not found";
    return signalPassFailure();
  }

  if (!sortTopologically(&hwModule.getBodyRegion().front())) {
    hwModule->emitError("could not resolve cycles in module");
    return signalPassFailure();
  }

  if (bound < ignoreAssertionsUntil) {
    hwModule->emitError(
        "number of ignored cycles must be less than or equal to bound");
    return signalPassFailure();
  }

  // Create necessary function declarations and globals
  auto *ctx = &getContext();
  OpBuilder builder(ctx);
  Location loc = moduleOp->getLoc();
  builder.setInsertionPointToEnd(moduleOp.getBody());
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  // Lookup or declare printf function.
  auto printfFunc =
      LLVM::lookupOrCreateFn(builder, moduleOp, "printf", ptrTy, voidTy, true);
  if (failed(printfFunc)) {
    moduleOp->emitError("failed to lookup or create printf");
    return signalPassFailure();
  }

  // Replace the top-module with a function performing the BMC
  auto entryFunc = func::FuncOp::create(builder, loc, topModule,
                                        builder.getFunctionType({}, {}));
  builder.createBlock(&entryFunc.getBody());

  // Save module name and port info before the module is erased
  auto *hwOutput = hwModule.getBody().front().getTerminator();
  SmallVector<std::pair<StringAttr, Value>> namedPorts;
  for (auto &port : hwModule.getPortList()) {
    Value portValue = port.isInput()
                          ? hwModule.getBody().front().getArgument(port.argNum)
                          : hwOutput->getOperand(port.argNum);
    namedPorts.push_back({builder.getStringAttr(port.getName()), portValue});
  }

  {
    OpBuilder::InsertionGuard guard(builder);
    auto *terminator = hwModule.getBody().front().getTerminator();
    builder.setInsertionPoint(terminator);
    verif::YieldOp::create(builder, loc, terminator->getOperands());
    terminator->erase();
  }

  // Property ops must live in the top module: they move into the BMC
  // properties region below, and ops inside instantiated modules cannot be
  // scoped correctly there. Modules not reachable from the top module play
  // no part in the check and are erased.

  // Collect the modules reachable from the top module through instances.
  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
  llvm::SetVector<Operation *> reachable;
  for (auto *node :
       llvm::depth_first(instanceGraph.lookup(hwModule.getModuleNameAttr())))
    reachable.insert(node->getModule().getOperation());

  // Validate the property ops and collect the top module's assert/assume
  // ops for the properties region.
  auto isPropertyOp = [](Operation *op) {
    return isa<verif::AssertOp, verif::AssumeOp, verif::CoverOp,
               verif::ClockedAssertOp, verif::ClockedAssumeOp,
               verif::ClockedCoverOp>(op);
  };
  SmallVector<Operation *> propertyOps;
  bool invalid = false;
  for (Operation *mod : reachable) {
    mod->walk([&](Operation *op) {
      if (!isPropertyOp(op))
        return;
      if (mod != hwModule) {
        // Plain boolean assumes are fine in instantiated modules: they
        // convert inside the callee and persist as facts, which is exactly
        // the lifetime they need.
        if (isa<verif::AssumeOp>(op) &&
            op->getOperand(0).getType().isSignlessInteger(1))
          return;
        op->emitError("property ops inside instantiated modules are not "
                      "supported; run with --flatten-modules");
        invalid = true;
        return;
      }
      if (!isa<verif::AssertOp, verif::AssumeOp>(op)) {
        op->emitError("unsupported property operation - only verif.assert "
                      "and verif.assume are supported");
        invalid = true;
        return;
      }
      if (!op->getOperand(0).getType().isSignlessInteger(1)) {
        op->emitError("only boolean properties are supported");
        invalid = true;
        return;
      }
      propertyOps.push_back(op);
    });
  }
  if (invalid)
    return signalPassFailure();

  // Erase the unreachable modules.
  for (auto mod : llvm::make_early_inc_range(moduleOp.getOps<hw::HWModuleOp>()))
    if (!reachable.contains(mod))
      mod->erase();

  // Double the bound given to the BMC op unless in rising clocks only mode, as
  // a clock cycle involves two negations
  verif::BoundedModelCheckingOp bmcOp;
  auto numRegs = hwModule->getAttrOfType<IntegerAttr>("num_regs");
  auto initialValues = hwModule->getAttrOfType<ArrayAttr>("initial_values");
  if (numRegs && initialValues) {
    for (auto value : initialValues) {
      if (!isa<IntegerAttr, UnitAttr>(value)) {
        hwModule->emitError("initial_values attribute must contain only "
                            "integer or unit attributes");
        return signalPassFailure();
      }
    }
    bmcOp = verif::BoundedModelCheckingOp::create(
        builder, loc, risingClocksOnly ? bound : 2 * bound,
        cast<IntegerAttr>(numRegs).getValue().getZExtValue(), initialValues);
    // Annotate the op with how many cycles to ignore - again, we may need to
    // double this to account for rising and falling edges
    if (ignoreAssertionsUntil)
      bmcOp->setAttr("ignore_asserts_until",
                     builder.getI32IntegerAttr(
                         risingClocksOnly ? ignoreAssertionsUntil
                                          : 2 * ignoreAssertionsUntil));
  } else {
    hwModule->emitOpError("no num_regs or initial_values attribute found - "
                          "please run externalize "
                          "registers pass first");
    return signalPassFailure();
  }

  // Check that there's only one clock input to the module
  // TODO: supporting multiple clocks isn't too hard, an interleaving of clock
  // toggles just needs to be generated
  bool hasClk = false;
  for (auto input : hwModule.getInputTypes()) {
    if (isa<seq::ClockType>(input)) {
      if (hasClk) {
        hwModule.emitError("designs with multiple clocks not yet supported");
        return signalPassFailure();
      }
      hasClk = true;
    }
    if (auto hwStruct = dyn_cast<hw::StructType>(input)) {
      for (auto field : hwStruct.getElements()) {
        if (isa<seq::ClockType>(field.type)) {
          if (hasClk) {
            hwModule.emitError(
                "designs with multiple clocks not yet supported");
            return signalPassFailure();
          }
          hasClk = true;
        }
      }
    }
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    // Initialize clock to 0 if it exists, otherwise just yield nothing
    // We initialize to 1 if we're in rising clocks only mode
    auto *initBlock = builder.createBlock(&bmcOp.getInit());
    builder.setInsertionPointToStart(initBlock);
    if (hasClk) {
      auto initVal = hw::ConstantOp::create(builder, loc, builder.getI1Type(),
                                            risingClocksOnly ? 1 : 0);
      auto toClk = seq::ToClockOp::create(builder, loc, initVal);
      verif::YieldOp::create(builder, loc, ValueRange{toClk});
    } else {
      verif::YieldOp::create(builder, loc, ValueRange{});
    }

    // Toggle clock in loop region if it exists, otherwise just yield nothing
    auto *loopBlock = builder.createBlock(&bmcOp.getLoop());
    builder.setInsertionPointToStart(loopBlock);
    if (hasClk) {
      loopBlock->addArgument(seq::ClockType::get(ctx), loc);
      if (risingClocksOnly) {
        // In rising clocks only mode we don't need to toggle the clock
        verif::YieldOp::create(builder, loc,
                               ValueRange{loopBlock->getArgument(0)});
      } else {
        auto fromClk =
            seq::FromClockOp::create(builder, loc, loopBlock->getArgument(0));
        auto cNeg1 =
            hw::ConstantOp::create(builder, loc, builder.getI1Type(), -1);
        auto nClk = comb::XorOp::create(builder, loc, fromClk, cNeg1);
        auto toClk = seq::ToClockOp::create(builder, loc, nClk);
        // Only yield clock value
        verif::YieldOp::create(builder, loc, ValueRange{toClk});
      }
    } else {
      verif::YieldOp::create(builder, loc, ValueRange{});
    }
  }
  auto moduleName = hwModule.getNameAttr();
  bmcOp.getCircuit().takeBody(hwModule.getBody());
  hwModule->erase();

  // Move the property ops into the properties region; the circuit instead
  // yields their conditions as 'leaf' values (inserted before the register
  // next-state values), which become the region's block arguments.
  if (!propertyOps.empty()) {
    OpBuilder::InsertionGuard guard(builder);
    auto &circuitBlock = bmcOp.getCircuit().front();
    auto *propBlock = builder.createBlock(&bmcOp.getProps());

    auto getEnable = [](Operation *op) -> Value {
      return op->getNumOperands() > 1 ? op->getOperand(1) : Value();
    };
    llvm::SetVector<Value> leaves;
    for (Operation *propOp : propertyOps) {
      leaves.insert(propOp->getOperand(0));
      if (Value enable = getEnable(propOp))
        leaves.insert(enable);
    }

    for (Value leaf : leaves)
      propBlock->addArgument(leaf.getType(), leaf.getLoc());

    for (Operation *propOp : propertyOps)
      propOp->moveBefore(propBlock, propBlock->end());

    // Rewrite leaf uses within the properties region to the block arguments.
    for (auto [leaf, arg] :
         llvm::zip_equal(leaves, propBlock->getArguments())) {
      Value leafValue = leaf;
      leafValue.replaceUsesWithIf(arg, [&](OpOperand &use) {
        return use.getOwner()->getBlock() == propBlock;
      });
    }

    builder.setInsertionPointToEnd(propBlock);
    verif::YieldOp::create(builder, loc);

    // Yield the leaves from the circuit, before the register next-states.
    auto *circuitYield = circuitBlock.getTerminator();
    SmallVector<Value> yieldOperands(circuitYield->getOperands());
    yieldOperands.insert(yieldOperands.end() - numRegs.getInt(), leaves.begin(),
                         leaves.end());
    circuitYield->setOperands(yieldOperands);
  }

  // signal names for counter-example generation.
  {
    OpBuilder::InsertionGuard guard(builder);
    auto &circuitBlock = bmcOp.getCircuit().front();
    builder.setInsertionPoint(circuitBlock.getTerminator());
    auto scope = debug::ScopeOp::create(
        builder, loc, moduleName.getValue(),
        // TODO: Hierarchy support would require walking parent InstanceOps,
        // but LowerToBMC operates on a single top-level module with no
        // instance context available at this point.
        moduleName, nullptr);
    for (auto &[name, value] : namedPorts)
      debug::VariableOp::create(builder, loc, name, value, scope);
  }
  // Define global string constants to print on success/failure
  auto createUniqueStringGlobal = [&](StringRef str) -> FailureOr<Value> {
    Location loc = moduleOp.getLoc();

    OpBuilder b = OpBuilder::atBlockEnd(moduleOp.getBody());
    auto arrayTy = LLVM::LLVMArrayType::get(b.getI8Type(), str.size() + 1);
    auto global = LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/true, LLVM::linkage::Linkage::Private,
        "resultString",
        StringAttr::get(b.getContext(), Twine(str).concat(Twine('\00'))));
    SymbolTable symTable(moduleOp);
    if (failed(symTable.renameToUnique(global, {&symTable}))) {
      return mlir::failure();
    }

    return success(
        LLVM::AddressOfOp::create(builder, loc, global)->getResult(0));
  };

  auto successStrAddr =
      createUniqueStringGlobal("Bound reached with no violations!\n");
  auto failureStrAddr =
      createUniqueStringGlobal("Assertion can be violated!\n");

  if (failed(successStrAddr) || failed(failureStrAddr)) {
    moduleOp->emitOpError("could not create result message strings");
    return signalPassFailure();
  }

  auto formatString =
      LLVM::SelectOp::create(builder, loc, bmcOp.getResult(),
                             successStrAddr.value(), failureStrAddr.value());

  LLVM::CallOp::create(builder, loc, printfFunc.value(),
                       ValueRange{formatString});
  func::ReturnOp::create(builder, loc);

  if (insertMainFunc) {
    builder.setInsertionPointToEnd(getOperation().getBody());
    Type i32Ty = builder.getI32Type();
    auto mainFunc = func::FuncOp::create(
        builder, loc, "main", builder.getFunctionType({i32Ty, ptrTy}, {i32Ty}));
    builder.createBlock(&mainFunc.getBody(), {}, {i32Ty, ptrTy}, {loc, loc});
    func::CallOp::create(builder, loc, entryFunc, ValueRange{});
    // TODO: don't use LLVM here
    Value constZero = LLVM::ConstantOp::create(builder, loc, i32Ty, 0);
    func::ReturnOp::create(builder, loc, constZero);
  }
}
