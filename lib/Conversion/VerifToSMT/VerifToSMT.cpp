//===- VerifToSMT.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/VerifToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/SMT/IR/SMTTypes.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTVERIFTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
llvm::SmallDenseMap<unsigned, StringAttr> collectDebugNames(Block &block) {
  llvm::SmallDenseMap<unsigned, StringAttr> debugNames;
  for (auto arg : block.getArguments()) {
    for (auto *user : arg.getUsers()) {
      auto varOp = dyn_cast<debug::VariableOp>(user);
      if (!varOp)
        continue;
      auto name = varOp.getNameAttr();
      if (name.getValue().empty())
        continue;
      debugNames.try_emplace(arg.getArgNumber(), name);
      break;
    }
  }
  return debugNames;
}

static void attachDebugVariables(
    OpBuilder &builder, Location loc, ArrayRef<Type> originalTypes,
    ValueRange values,
    const llvm::SmallDenseMap<unsigned, StringAttr> &debugNames) {
  for (auto [argIndex, value] : llvm::enumerate(values)) {
    if (isa<seq::ClockType>(originalTypes[argIndex]))
      continue;
    auto it = debugNames.find(argIndex);
    if (it == debugNames.end())
      continue;
    debug::VariableOp::create(builder, loc, it->second, value,
                              /*scope=*/Value{});
  }
}

//===----------------------------------------------------------------------===//
// BMC property encoding: the circuit yields 'leaf' values (between its
// outputs and register next-states) that the properties region receives as
// block arguments. Assumes become permanent facts asserted inside the
// circuit function; nothing is ever popped. Asserts are never asserted:
// each step checks `smt.check assuming(OR of enable && !property)`.
//===----------------------------------------------------------------------===//

static Value materializeBool(const TypeConverter &typeConverter,
                             OpBuilder &builder, Location loc, Value value) {
  return typeConverter.materializeTargetConversion(
      builder, loc, smt::BoolType::get(builder.getContext()), value);
}

static Value materializeEnabledProperty(const TypeConverter &typeConverter,
                                        OpBuilder &builder, Location loc,
                                        Value property, Value enable) {
  Value cond = materializeBool(typeConverter, builder, loc, property);
  if (enable) {
    Value enableCond = materializeBool(typeConverter, builder, loc, enable);
    cond = smt::ImpliesOp::create(builder, loc, enableCond, cond);
  }
  return cond;
}

/// Lower a verif::AssertOp operation with an i1 operand to a smt::AssertOp,
/// negated to check for unsatisfiability. An enabled assert can only be
/// violated while its enable holds: enable && !property.
struct VerifAssertOpConversion : OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value cond =
        materializeEnabledProperty(*typeConverter, rewriter, op.getLoc(),
                                   adaptor.getProperty(), adaptor.getEnable());
    Value notCond = smt::NotOp::create(rewriter, op.getLoc(), cond);
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, notCond);
    return success();
  }
};

/// Lower a verif::AssumeOp operation with an i1 operand to a smt::AssertOp.
/// An enabled assume only constrains while its enable holds:
/// enable -> property.
struct VerifAssumeOpConversion : OpConversionPattern<verif::AssumeOp> {
  using OpConversionPattern<verif::AssumeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value cond =
        materializeEnabledProperty(*typeConverter, rewriter, op.getLoc(),
                                   adaptor.getProperty(), adaptor.getEnable());
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, cond);
    return success();
  }
};

template <typename OpTy>
struct CircuitRelationCheckOpConversion : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

protected:
  using ConversionPattern::typeConverter;
  void
  createOutputsDifferentOps(Operation *firstOutputs, Operation *secondOutputs,
                            Location &loc, ConversionPatternRewriter &rewriter,
                            SmallVectorImpl<Value> &outputsDifferent) const {
    // Convert the yielded values back to the source type system (since
    // the operations of the inlined blocks will be converted by other patterns
    // later on and we should make sure the IR is well-typed after each pattern
    // application), and compare the output values.
    for (auto [out1, out2] :
         llvm::zip(firstOutputs->getOperands(), secondOutputs->getOperands())) {
      Value o1 = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(out1.getType()), out1);
      Value o2 = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(out1.getType()), out2);
      outputsDifferent.emplace_back(
          smt::DistinctOp::create(rewriter, loc, o1, o2));
    }
  }

  void replaceOpWithSatCheck(OpTy &op, Location &loc,
                             ConversionPatternRewriter &rewriter,
                             smt::SolverOp &solver) const {
    // If no operation uses the result of this solver, we leave our check
    // operations empty. If the result is used, we create a check operation with
    // the result type of the operation and yield the result of the check
    // operation.
    if (op.getNumResults() == 0) {
      auto checkOp = smt::CheckOp::create(rewriter, loc, TypeRange{});
      rewriter.createBlock(&checkOp.getSatRegion());
      smt::YieldOp::create(rewriter, loc);
      rewriter.createBlock(&checkOp.getUnknownRegion());
      smt::YieldOp::create(rewriter, loc);
      rewriter.createBlock(&checkOp.getUnsatRegion());
      smt::YieldOp::create(rewriter, loc);
      rewriter.setInsertionPointAfter(checkOp);
      smt::YieldOp::create(rewriter, loc);

      // Erase as operation is replaced by an operator without a return value.
      rewriter.eraseOp(op);
    } else {
      Value falseVal =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false));
      Value trueVal =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
      auto checkOp = smt::CheckOp::create(rewriter, loc, rewriter.getI1Type());
      rewriter.createBlock(&checkOp.getSatRegion());
      smt::YieldOp::create(rewriter, loc, falseVal);
      rewriter.createBlock(&checkOp.getUnknownRegion());
      smt::YieldOp::create(rewriter, loc, falseVal);
      rewriter.createBlock(&checkOp.getUnsatRegion());
      smt::YieldOp::create(rewriter, loc, trueVal);
      rewriter.setInsertionPointAfter(checkOp);
      smt::YieldOp::create(rewriter, loc, checkOp->getResults());

      rewriter.replaceOp(op, solver->getResults());
    }
  }
};

/// Lower a verif::LecOp operation to a miter circuit encoded in SMT.
/// More information on miter circuits can be found, e.g., in this paper:
/// Brand, D., 1993, November. Verification of large synthesized designs. In
/// Proceedings of 1993 International Conference on Computer Aided Design
/// (ICCAD) (pp. 534-537). IEEE.
struct LogicEquivalenceCheckingOpConversion
    : CircuitRelationCheckOpConversion<verif::LogicEquivalenceCheckingOp> {
  using CircuitRelationCheckOpConversion<
      verif::LogicEquivalenceCheckingOp>::CircuitRelationCheckOpConversion;

  LogicalResult
  matchAndRewrite(verif::LogicEquivalenceCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *firstOutputs = adaptor.getFirstCircuit().front().getTerminator();
    auto *secondOutputs = adaptor.getSecondCircuit().front().getTerminator();

    auto hasNoResult = op.getNumResults() == 0;

    if (firstOutputs->getNumOperands() == 0) {
      // Trivially equivalent
      if (hasNoResult) {
        rewriter.eraseOp(op);
      } else {
        Value trueVal = arith::ConstantOp::create(rewriter, loc,
                                                  rewriter.getBoolAttr(true));
        rewriter.replaceOp(op, trueVal);
      }
      return success();
    }

    // Solver will only return a result when it is used to check the returned
    // value.
    smt::SolverOp solver;
    if (hasNoResult)
      solver = smt::SolverOp::create(rewriter, loc, TypeRange{}, ValueRange{});
    else
      solver = smt::SolverOp::create(rewriter, loc, rewriter.getI1Type(),
                                     ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    // First, convert the block arguments of the miter bodies.
    if (failed(rewriter.convertRegionTypes(&adaptor.getFirstCircuit(),
                                           *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&adaptor.getSecondCircuit(),
                                           *typeConverter)))
      return failure();

    // Second, create the symbolic values we replace the block arguments with
    SmallVector<Value> inputs;
    for (auto arg : adaptor.getFirstCircuit().getArguments())
      inputs.push_back(smt::DeclareFunOp::create(rewriter, loc, arg.getType()));

    // Third, inline the blocks
    // Note: the argument value replacement does not happen immediately, but
    // only after all the operations are already legalized.
    // Also, it has to be ensured that the original argument type and the type
    // of the value with which is is to be replaced match. The value is looked
    // up (transitively) in the replacement map at the time the replacement
    // pattern is committed.
    rewriter.mergeBlocks(&adaptor.getFirstCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.mergeBlocks(&adaptor.getSecondCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.setInsertionPointToEnd(solver.getBody());

    // Fourth, build the assertion.
    SmallVector<Value> outputsDifferent;
    createOutputsDifferentOps(firstOutputs, secondOutputs, loc, rewriter,
                              outputsDifferent);

    rewriter.eraseOp(firstOutputs);
    rewriter.eraseOp(secondOutputs);

    Value toAssert;
    if (outputsDifferent.size() == 1)
      toAssert = outputsDifferent[0];
    else
      toAssert = smt::OrOp::create(rewriter, loc, outputsDifferent);

    smt::AssertOp::create(rewriter, loc, toAssert);

    // Fifth, check for satisfiablility and report the result back.
    replaceOpWithSatCheck(op, loc, rewriter, solver);
    return success();
  }
};

struct RefinementCheckingOpConversion
    : CircuitRelationCheckOpConversion<verif::RefinementCheckingOp> {
  using CircuitRelationCheckOpConversion<
      verif::RefinementCheckingOp>::CircuitRelationCheckOpConversion;

  LogicalResult
  matchAndRewrite(verif::RefinementCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Find non-deterministic values (free variables) in the source circuit.
    // For now, only support quantification over 'primitive' types.
    SmallVector<Value> srcNonDetValues;
    bool canBind = true;
    for (auto ndOp : op.getFirstCircuit().getOps<smt::DeclareFunOp>()) {
      if (!isa<smt::IntType, smt::BoolType, smt::BitVectorType>(
              ndOp.getType())) {
        ndOp.emitError("Uninterpreted function of non-primitive type cannot be "
                       "converted.");
        canBind = false;
      }
      srcNonDetValues.push_back(ndOp.getResult());
    }
    if (!canBind)
      return failure();

    if (srcNonDetValues.empty()) {
      // If there is no non-determinism in the source circuit, the
      // refinement check becomes an equivalence check, which does not
      // need quantified expressions.
      auto eqOp = verif::LogicEquivalenceCheckingOp::create(
          rewriter, op.getLoc(), op.getNumResults() != 0);
      rewriter.moveBlockBefore(&op.getFirstCircuit().front(),
                               &eqOp.getFirstCircuit(),
                               eqOp.getFirstCircuit().end());
      rewriter.moveBlockBefore(&op.getSecondCircuit().front(),
                               &eqOp.getSecondCircuit(),
                               eqOp.getSecondCircuit().end());
      rewriter.replaceOp(op, eqOp);
      return success();
    }

    Location loc = op.getLoc();
    auto *firstOutputs = adaptor.getFirstCircuit().front().getTerminator();
    auto *secondOutputs = adaptor.getSecondCircuit().front().getTerminator();

    auto hasNoResult = op.getNumResults() == 0;

    if (firstOutputs->getNumOperands() == 0) {
      // Trivially equivalent
      if (hasNoResult) {
        rewriter.eraseOp(op);
      } else {
        Value trueVal = arith::ConstantOp::create(rewriter, loc,
                                                  rewriter.getBoolAttr(true));
        rewriter.replaceOp(op, trueVal);
      }
      return success();
    }

    // Solver will only return a result when it is used to check the returned
    // value.
    smt::SolverOp solver;
    if (hasNoResult)
      solver = smt::SolverOp::create(rewriter, loc, TypeRange{}, ValueRange{});
    else
      solver = smt::SolverOp::create(rewriter, loc, rewriter.getI1Type(),
                                     ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    // Convert the block arguments of the miter bodies.
    if (failed(rewriter.convertRegionTypes(&adaptor.getFirstCircuit(),
                                           *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&adaptor.getSecondCircuit(),
                                           *typeConverter)))
      return failure();

    // Create the symbolic values we replace the block arguments with
    SmallVector<Value> inputs;
    for (auto arg : adaptor.getFirstCircuit().getArguments())
      inputs.push_back(smt::DeclareFunOp::create(rewriter, loc, arg.getType()));

    // Inline the target circuit. Free variables remain free variables.
    rewriter.mergeBlocks(&adaptor.getSecondCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.setInsertionPointToEnd(solver.getBody());

    // Create the universally quantified expression containing the source
    // circuit. Free variables in the circuit's body become bound variables.
    auto forallOp = smt::ForallOp::create(
        rewriter, op.getLoc(), TypeRange(srcNonDetValues),
        [&](OpBuilder &builder, auto, ValueRange args) -> Value {
          // Inline the source circuit
          Block *body = builder.getBlock();
          rewriter.mergeBlocks(&adaptor.getFirstCircuit().front(), body,
                               inputs);

          // Replace non-deterministic values with the quantifier's bound
          // variables
          for (auto [freeVar, boundVar] : llvm::zip(srcNonDetValues, args))
            rewriter.replaceOp(freeVar.getDefiningOp(), boundVar);

          // Compare the output values
          rewriter.setInsertionPointToEnd(body);
          SmallVector<Value> outputsDifferent;
          createOutputsDifferentOps(firstOutputs, secondOutputs, loc, rewriter,
                                    outputsDifferent);
          if (outputsDifferent.size() == 1)
            return outputsDifferent[0];
          else
            return rewriter.createOrFold<smt::OrOp>(loc, outputsDifferent);
        });

    rewriter.eraseOp(firstOutputs);
    rewriter.eraseOp(secondOutputs);

    // Assert the quantified expression
    rewriter.setInsertionPointAfter(forallOp);
    smt::AssertOp::create(rewriter, op.getLoc(), forallOp.getResult());

    // Check for satisfiability and report the result back.
    replaceOpWithSatCheck(op, loc, rewriter, solver);
    return success();
  }
};

/// Lower a verif::BMCOp operation to an MLIR program that performs the bounded
/// model check
struct VerifBoundedModelCheckingOpConversion
    : OpConversionPattern<verif::BoundedModelCheckingOp> {
  using OpConversionPattern<verif::BoundedModelCheckingOp>::OpConversionPattern;

  VerifBoundedModelCheckingOpConversion(
      TypeConverter &converter, MLIRContext *context, Namespace &names,
      bool risingClocksOnly, SmallVectorImpl<Operation *> &propertylessBMCOps)
      : OpConversionPattern(converter, context), names(names),
        risingClocksOnly(risingClocksOnly),
        propertylessBMCOps(propertylessBMCOps) {}
  LogicalResult
  matchAndRewrite(verif::BoundedModelCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (std::find(propertylessBMCOps.begin(), propertylessBMCOps.end(), op) !=
        propertylessBMCOps.end()) {
      // Nothing to check; return true directly instead of asking the
      // solver, which would report SAT.
      Value trueVal =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
      rewriter.replaceOp(op, trueVal);
      return success();
    }

    // Per the verifier (plus isolation and dominance), the properties region
    // holds boolean assert/assume ops over its own leaf block arguments.
    Block &propsBlock = op.getProps().front();
    unsigned numLeaves = propsBlock.getNumArguments();

    SmallVector<Type> oldLoopInputTy(op.getLoop().getArgumentTypes());
    SmallVector<Type> oldCircuitInputTy(op.getCircuit().getArgumentTypes());
    // TODO: the init and loop regions should be able to be concrete instead of
    // symbolic which is probably preferable - just need to convert back and
    // forth
    SmallVector<Type> loopInputTy, circuitInputTy, initOutputTy,
        circuitOutputTy;
    if (failed(typeConverter->convertTypes(oldLoopInputTy, loopInputTy)))
      return failure();
    if (failed(typeConverter->convertTypes(oldCircuitInputTy, circuitInputTy)))
      return failure();
    if (failed(typeConverter->convertTypes(
            op.getInit().front().back().getOperandTypes(), initOutputTy)))
      return failure();
    if (failed(typeConverter->convertTypes(
            op.getCircuit().front().back().getOperandTypes(), circuitOutputTy)))
      return failure();
    auto debugNames = collectDebugNames(op.getCircuit().front());
    if (failed(rewriter.convertRegionTypes(&op.getInit(), *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&op.getLoop(), *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&op.getCircuit(), *typeConverter)))
      return failure();

    unsigned numRegs = op.getNumRegs();
    auto initialValues = op.getInitialValues();

    // Encodes the requested kind of property ops with the leaf block
    // arguments bound to this step's circuit values.
    auto encodeProps = [&](OpBuilder &builder, ValueRange circuitVals,
                           bool wantAsserts,
                           function_ref<void(Value, Value)> encodeProp) {
      auto leafVals = circuitVals.drop_back(numRegs).take_back(numLeaves);
      SmallVector<Value> boolCache(numLeaves);
      auto toBool = [&](Value v) -> Value {
        unsigned index = cast<BlockArgument>(v).getArgNumber();
        Value &cached = boolCache[index];
        if (!cached)
          cached =
              materializeBool(*typeConverter, builder, loc, leafVals[index]);
        return cached;
      };
      for (Operation &propOp : propsBlock) {
        if (!isa<verif::AssertOp, verif::AssumeOp>(propOp) ||
            isa<verif::AssertOp>(propOp) != wantAsserts)
          continue;
        Value property = toBool(propOp.getOperand(0));
        Value enable = propOp.getNumOperands() > 1
                           ? toBool(propOp.getOperand(1))
                           : Value();
        encodeProp(property, enable);
      }
    };

    auto initFuncTy = rewriter.getFunctionType({}, initOutputTy);
    // Loop and init output types are necessarily the same, so just use init
    // output types
    auto loopFuncTy = rewriter.getFunctionType(loopInputTy, initOutputTy);
    auto circuitFuncTy =
        rewriter.getFunctionType(circuitInputTy, circuitOutputTy);

    func::FuncOp initFuncOp, loopFuncOp, circuitFuncOp;

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(
          op->getParentOfType<ModuleOp>().getBody());
      initFuncOp = func::FuncOp::create(rewriter, loc,
                                        names.newName("bmc_init"), initFuncTy);
      rewriter.inlineRegionBefore(op.getInit(), initFuncOp.getFunctionBody(),
                                  initFuncOp.end());
      loopFuncOp = func::FuncOp::create(rewriter, loc,
                                        names.newName("bmc_loop"), loopFuncTy);
      rewriter.inlineRegionBefore(op.getLoop(), loopFuncOp.getFunctionBody(),
                                  loopFuncOp.end());
      circuitFuncOp = func::FuncOp::create(
          rewriter, loc, names.newName("bmc_circuit"), circuitFuncTy);
      rewriter.inlineRegionBefore(op.getCircuit(),
                                  circuitFuncOp.getFunctionBody(),
                                  circuitFuncOp.end());
      auto funcOps = {&initFuncOp, &loopFuncOp, &circuitFuncOp};
      // initOutputTy is the same as loop output types
      auto outputTys = {initOutputTy, initOutputTy, circuitOutputTy};
      for (auto [funcOp, outputTy] : llvm::zip(funcOps, outputTys)) {
        auto operands = funcOp->getBody().front().back().getOperands();
        rewriter.eraseOp(&funcOp->getFunctionBody().front().back());
        rewriter.setInsertionPointToEnd(&funcOp->getBody().front());
        SmallVector<Value> toReturn;
        for (unsigned i = 0; i < outputTy.size(); ++i)
          toReturn.push_back(typeConverter->materializeTargetConversion(
              rewriter, loc, outputTy[i], operands[i]));
        func::ReturnOp::create(rewriter, loc, toReturn);
      }

      // Assumes become permanent facts inside the circuit function;
      // nothing is ever popped, so they hold across timesteps.
      auto *ret = circuitFuncOp.getBody().front().getTerminator();
      rewriter.setInsertionPoint(ret);
      encodeProps(rewriter, ret->getOperands(), /*wantAsserts=*/false,
                  [&](Value property, Value enable) {
                    if (enable)
                      property = smt::ImpliesOp::create(rewriter, loc, enable,
                                                        property);
                    smt::AssertOp::create(rewriter, loc, property);
                  });
    }

    auto solver = smt::SolverOp::create(rewriter, loc, rewriter.getI1Type(),
                                        ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    // Call init func to get initial clock values
    ValueRange initVals =
        func::CallOp::create(rewriter, loc, initFuncOp)->getResults();

    // InputDecls order should be <circuit arguments> <state arguments>
    // <wasViolated>
    // Get list of clock indexes in circuit args
    size_t initIndex = 0;
    size_t regStartIdx = oldCircuitInputTy.size() - numRegs;
    SmallVector<Value> inputDecls;
    SmallVector<int> clockIndexes;
    auto getNameAttr = [&](unsigned argIndex, bool isReg) {
      if (auto it = debugNames.find(argIndex); it != debugNames.end())
        return it->second;
      auto fallback = isReg ? ("reg_" + Twine(argIndex - regStartIdx)).str()
                            : ("input_" + Twine(argIndex)).str();
      return rewriter.getStringAttr(fallback);
    };
    for (auto [curIndex, oldTy, newTy] :
         llvm::enumerate(oldCircuitInputTy, circuitInputTy)) {
      if (isa<seq::ClockType>(oldTy)) {
        inputDecls.push_back(initVals[initIndex++]);
        clockIndexes.push_back(curIndex);
        continue;
      }
      if (curIndex >= regStartIdx) {
        auto initVal = initialValues[curIndex - regStartIdx];
        if (auto initIntAttr = dyn_cast<IntegerAttr>(initVal)) {
          const auto &cstInt = initIntAttr.getValue();
          assert(cstInt.getBitWidth() ==
                     cast<smt::BitVectorType>(newTy).getWidth() &&
                 "Width mismatch between initial value and target type");
          inputDecls.push_back(
              smt::BVConstantOp::create(rewriter, loc, cstInt));
          continue;
        }
      }
      inputDecls.push_back(smt::DeclareFunOp::create(
          rewriter, loc, newTy,
          getNameAttr(curIndex, curIndex >= regStartIdx)));
    }

    auto numStateArgs = initVals.size() - initIndex;
    // Add the rest of the init vals (state args)
    for (; initIndex < initVals.size(); ++initIndex)
      inputDecls.push_back(initVals[initIndex]);

    attachDebugVariables(
        rewriter, loc, oldCircuitInputTy,
        ValueRange(inputDecls).take_front(circuitFuncOp.getNumArguments()),
        debugNames);

    Value lowerBound =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    Value step =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
    Value upperBound =
        arith::ConstantOp::create(rewriter, loc, adaptor.getBoundAttr());
    Value constFalse =
        arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false));
    Value constTrue =
        arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(true));
    inputDecls.push_back(constFalse); // wasViolated?

    // TODO: swapping to a whileOp here would allow early exit once the property
    // is violated
    // Perform model check up to the provided bound
    auto forOp = scf::ForOp::create(
        rewriter, loc, lowerBound, upperBound, step, inputDecls,
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          attachDebugVariables(
              builder, loc, oldCircuitInputTy,
              iterArgs.take_front(circuitFuncOp.getNumArguments()), debugNames);

          // Execute the circuit
          ValueRange circuitCallOuts =
              func::CallOp::create(
                  builder, loc, circuitFuncOp,
                  iterArgs.take_front(circuitFuncOp.getNumArguments()))
                  ->getResults();

          // The step's question: the disjunction of the (enabled) assert
          // violations. Nothing is asserted.
          SmallVector<Value> violations;
          encodeProps(builder, circuitCallOuts, /*wantAsserts=*/true,
                      [&](Value property, Value enable) {
                        Value violation =
                            smt::NotOp::create(builder, loc, property);
                        if (enable)
                          violation = smt::AndOp::create(builder, loc, enable,
                                                         violation);
                        violations.push_back(violation);
                      });
          Value stepViolation =
              violations.size() == 1
                  ? violations.front()
                  : smt::OrOp::create(builder, loc, violations);

          // If we have a cycle up to which we ignore assertions, we need an
          // IfOp to track this
          // First, save the insertion point so we can safely enter the IfOp

          auto insideForPoint = builder.saveInsertionPoint();
          // We need to still have the yielded result of the op in scope after
          // we've built the check
          Value yieldedValue;
          auto ignoreAssertionsUntil =
              op->getAttrOfType<IntegerAttr>("ignore_asserts_until");
          if (ignoreAssertionsUntil) {
            auto ignoreUntilConstant = arith::ConstantOp::create(
                builder, loc,
                rewriter.getI32IntegerAttr(
                    ignoreAssertionsUntil.getValue().getZExtValue()));
            auto shouldIgnore =
                arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                                      i, ignoreUntilConstant);
            auto ifShouldIgnore = scf::IfOp::create(
                builder, loc, builder.getI1Type(), shouldIgnore, true);
            // If we should ignore, yield the existing value
            builder.setInsertionPointToEnd(
                &ifShouldIgnore.getThenRegion().front());
            scf::YieldOp::create(builder, loc, ValueRange(iterArgs.back()));
            builder.setInsertionPointToEnd(
                &ifShouldIgnore.getElseRegion().front());
            yieldedValue = ifShouldIgnore.getResult(0);
          }

          SmallVector<Value> checkAssumptions;
          if (stepViolation)
            checkAssumptions.push_back(stepViolation);
          auto checkOp = smt::CheckOp::create(
              rewriter, loc, builder.getI1Type(), checkAssumptions);
          {
            OpBuilder::InsertionGuard guard(builder);
            builder.createBlock(&checkOp.getSatRegion());
            smt::YieldOp::create(builder, loc, constTrue);
            builder.createBlock(&checkOp.getUnknownRegion());
            smt::YieldOp::create(builder, loc, constTrue);
            builder.createBlock(&checkOp.getUnsatRegion());
            smt::YieldOp::create(builder, loc, constFalse);
          }

          Value violated = arith::OrIOp::create(
              builder, loc, checkOp.getResult(0), iterArgs.back());

          // If we've packaged everything in an IfOp, we need to yield the
          // new violated value
          if (ignoreAssertionsUntil) {
            scf::YieldOp::create(builder, loc, violated);
            // Replace the variable with the yielded value
            violated = yieldedValue;
          }

          // If we created an IfOp, make sure we start inserting after it again
          builder.restoreInsertionPoint(insideForPoint);

          // Update clock and state values; runs after the check so loop
          // effects cannot constrain the current step's query.
          SmallVector<Value> loopCallInputs;
          for (int index : clockIndexes)
            loopCallInputs.push_back(iterArgs[index]);
          llvm::append_range(loopCallInputs,
                             iterArgs.drop_back().take_back(numStateArgs));
          ValueRange loopVals =
              func::CallOp::create(builder, loc, loopFuncOp, loopCallInputs)
                  ->getResults();

          size_t loopIndex = 0;
          // Collect decls to yield at end of iteration
          SmallVector<Value> newDecls;
          for (auto [inputIdx, oldTy, newTy] :
               llvm::enumerate(TypeRange(oldCircuitInputTy).drop_back(numRegs),
                               TypeRange(circuitInputTy).drop_back(numRegs))) {
            if (isa<seq::ClockType>(oldTy)) {
              newDecls.push_back(loopVals[loopIndex++]);
            } else {
              newDecls.push_back(smt::DeclareFunOp::create(
                  builder, loc, newTy, getNameAttr(inputIdx, false)));
            }
          }

          // Only update the registers on a clock posedge unless in rising
          // clocks only mode
          // TODO: this will also need changing with multiple clocks - currently
          // it only accounts for the one clock case.
          if (clockIndexes.size() == 1) {
            SmallVector<Value> regInputs = circuitCallOuts.take_back(numRegs);
            if (risingClocksOnly) {
              // In rising clocks only mode we don't need to worry about whether
              // there was a posedge
              newDecls.append(regInputs);
            } else {
              auto clockIndex = clockIndexes[0];
              auto oldClock = iterArgs[clockIndex];
              // The clock is necessarily the first value returned by the loop
              // region
              auto newClock = loopVals[0];
              auto oldClockLow = smt::BVNotOp::create(builder, loc, oldClock);
              auto isPosedgeBV =
                  smt::BVAndOp::create(builder, loc, oldClockLow, newClock);
              // Convert posedge bv<1> to bool
              auto trueBV = smt::BVConstantOp::create(builder, loc, 1, 1);
              auto isPosedge =
                  smt::EqOp::create(builder, loc, isPosedgeBV, trueBV);
              auto regStates =
                  iterArgs.take_front(circuitFuncOp.getNumArguments())
                      .take_back(numRegs);
              SmallVector<Value> nextRegStates;
              for (auto [regState, regInput] :
                   llvm::zip(regStates, regInputs)) {
                // Create an ITE to calculate the next reg state
                // TODO: we create a lot of ITEs here that will slow things down
                // - these could be avoided by making init/loop regions concrete
                nextRegStates.push_back(smt::IteOp::create(
                    builder, loc, isPosedge, regInput, regState));
              }
              newDecls.append(nextRegStates);
            }
          }

          // Add the rest of the loop state args
          for (; loopIndex < loopVals.size(); ++loopIndex)
            newDecls.push_back(loopVals[loopIndex]);

          attachDebugVariables(
              builder, loc, oldCircuitInputTy,
              ValueRange(newDecls).take_front(circuitFuncOp.getNumArguments()),
              debugNames);

          newDecls.push_back(violated);

          scf::YieldOp::create(builder, loc, newDecls);
        });

    Value res = arith::XOrIOp::create(rewriter, loc, forOp->getResults().back(),
                                      constTrue);
    smt::YieldOp::create(rewriter, loc, res);
    rewriter.replaceOp(op, solver.getResults());
    return success();
  }

  Namespace &names;
  bool risingClocksOnly;
  SmallVectorImpl<Operation *> &propertylessBMCOps;
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Verif to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertVerifToSMTPass
    : public circt::impl::ConvertVerifToSMTBase<ConvertVerifToSMTPass> {
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

void circt::populateVerifToSMTConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns, Namespace &names,
    bool risingClocksOnly, SmallVectorImpl<Operation *> &propertylessBMCOps) {
  patterns.add<VerifAssertOpConversion, VerifAssumeOpConversion,
               LogicEquivalenceCheckingOpConversion,
               RefinementCheckingOpConversion>(converter,
                                               patterns.getContext());
  patterns.add<VerifBoundedModelCheckingOpConversion>(
      converter, patterns.getContext(), names, risingClocksOnly,
      propertylessBMCOps);
}

//===----------------------------------------------------------------------===//
// BMC validation
//===----------------------------------------------------------------------===//

/// Initial values are only supported on integer-typed registers.
static LogicalResult
checkRegisterInitialValues(verif::BoundedModelCheckingOp bmcOp) {
  auto regTypes = TypeRange(bmcOp.getCircuit().getArgumentTypes())
                      .take_back(bmcOp.getNumRegs());
  for (auto [regType, initVal] :
       llvm::zip(regTypes, bmcOp.getInitialValues())) {
    if (isa<UnitAttr>(initVal))
      continue;
    if (!isa<IntegerType>(regType))
      return bmcOp.emitError("initial values are currently only supported "
                             "for registers with integer types");
    auto tyAttr = dyn_cast<TypedAttr>(initVal);
    if (!tyAttr || tyAttr.getType() != regType)
      return bmcOp.emitError("type of initial value does not match type of "
                             "initialized register");
  }
  return success();
}

/// The BMC unrolling supports at most one clock.
/// TODO: remove once reg ins/outs can be associated with clocks.
static LogicalResult checkAtMostOneClock(verif::BoundedModelCheckingOp bmcOp) {
  int numClocks =
      llvm::count_if(bmcOp.getCircuit().getArgumentTypes(),
                     [](Type ty) { return isa<seq::ClockType>(ty); });
  if (numClocks > 1)
    return bmcOp.emitError(
        "only modules with one or zero clocks are currently supported");
  return success();
}

/// Asserts outside the properties region would convert into permanent
/// facts and mask later violations; assumes are fine anywhere, since
/// persisting is exactly the lifetime facts need.
static LogicalResult
checkNoMisplacedPropertyOps(verif::BoundedModelCheckingOp bmcOp,
                            SymbolTable &symbolTable) {
  // Unsupported property ops in the op's own regions would otherwise be
  // silently erased by the propertyless shortcut.
  bool invalid = false;
  for (Region *region :
       {&bmcOp.getCircuit(), &bmcOp.getInit(), &bmcOp.getLoop()})
    region->walk([&](Operation *op) {
      if (isa<verif::AssertOp>(op)) {
        op->emitError("assertions are not supported inside the circuit, "
                      "init, or loop regions of a verif.bmc op - they must "
                      "live in the op's properties region (the circt-bmc "
                      "pipeline moves them there)");
        invalid = true;
      } else if (isa<verif::CoverOp, verif::ClockedAssertOp,
                     verif::ClockedAssumeOp, verif::ClockedCoverOp>(op)) {
        op->emitError("unsupported property operation inside a verif.bmc "
                      "region - only boolean verif.assume is supported here");
        invalid = true;
      }
    });
  if (invalid)
    return failure();

  // Reject asserts in functions and modules reachable through calls and
  // instances.
  SetVector<Operation *> reachable;
  auto enqueueCallees = [&](auto &&root) {
    root.walk([&](Operation *op) {
      Operation *callee = nullptr;
      if (auto inst = dyn_cast<InstanceOp>(op))
        callee = symbolTable.lookup(inst.getModuleName());
      else if (auto call = dyn_cast<func::CallOp>(op))
        callee = symbolTable.lookup(call.getCallee());
      if (callee)
        reachable.insert(callee);
    });
  };
  for (Region *region :
       {&bmcOp.getCircuit(), &bmcOp.getInit(), &bmcOp.getLoop()})
    enqueueCallees(*region);
  for (unsigned i = 0; i < reachable.size(); ++i)
    enqueueCallees(*reachable[i]);
  for (Operation *callee : reachable)
    if (callee->walk([&](verif::AssertOp) { return WalkResult::interrupt(); })
            .wasInterrupted())
      return bmcOp.emitError(
          "assertions inside instantiated modules or called functions are "
          "not supported - inline them into the top module first (e.g. with "
          "--flatten-modules)");
  return success();
}

/// Validates every verif.bmc op and collects the propertyless ones, which
/// short-circuit to a trivial pass.
static LogicalResult
validateBMCOps(ModuleOp module, SymbolTable &symbolTable,
               SmallVectorImpl<Operation *> &propertylessBMCOps) {
  WalkResult result = module.walk([&](verif::BoundedModelCheckingOp bmcOp) {
    if (failed(checkRegisterInitialValues(bmcOp)) ||
        failed(checkAtMostOneClock(bmcOp)) ||
        failed(checkNoMisplacedPropertyOps(bmcOp, symbolTable)))
      return WalkResult::interrupt();
    bool hasAsserts = !bmcOp.getProps().empty() &&
                      llvm::any_of(bmcOp.getProps().front(), [](Operation &op) {
                        return isa<verif::AssertOp>(op);
                      });
    if (!hasAsserts) {
      bmcOp.emitWarning("no property provided to check in module - will "
                        "trivially find no violations.");
      propertylessBMCOps.push_back(bmcOp);
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void ConvertVerifToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<verif::VerifDialect>();
  target.addLegalDialect<debug::DebugDialect, smt::SMTDialect,
                         arith::ArithDialect, scf::SCFDialect,
                         func::FuncDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  // Ops in the properties region of a verif.bmc op are encoded (and erased)
  // by the BMC op's own conversion, not by their own patterns.
  auto inPropsRegion = [](Operation *op) {
    auto bmcOp =
        dyn_cast_or_null<verif::BoundedModelCheckingOp>(op->getParentOp());
    return bmcOp && op->getParentRegion() == &bmcOp.getProps();
  };
  target.addDynamicallyLegalOp<verif::AssertOp, verif::AssumeOp>(inPropsRegion);

  SymbolTable symbolTable(getOperation());
  SmallVector<Operation *> propertylessBMCOps;
  if (failed(validateBMCOps(getOperation(), symbolTable, propertylessBMCOps)))
    return signalPassFailure();
  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter);

  SymbolCache symCache;
  symCache.addDefinitions(getOperation());
  Namespace names;
  names.add(symCache);

  populateVerifToSMTConversionPatterns(converter, patterns, names,
                                       risingClocksOnly, propertylessBMCOps);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
