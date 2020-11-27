/*
 ***********************************************************************************************************************
 *
 *  Copyright (c) 2020 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *
 **********************************************************************************************************************/
/**
 ***********************************************************************************************************************
 * @file  PatchReadFirstLane.cpp
 * @brief LLPC source file: contains declaration and implementation of class lgc::PatchReadFirstLane.
 ***********************************************************************************************************************
 */
#include "lgc/Builder.h"
#include "lgc/patch/Patch.h"
#include "lgc/state/PipelineState.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include <deque>

#define DEBUG_TYPE "lgc-patch-read-first-lane"

using namespace lgc;
using namespace llvm;

namespace {
class PatchReadFirstLane final : public FunctionPass {
public:
  PatchReadFirstLane();

  virtual bool runOnFunction(Function &function) override;
  void getAnalysisUsage(AnalysisUsage &analysisUsage) const override;

  static char ID; // ID of this pass

private:
  PatchReadFirstLane(const PatchReadFirstLane &) = delete;
  PatchReadFirstLane &operator=(const PatchReadFirstLane &) = delete;

  bool liftReadFirstLane(Function &function);
  void collectAssumeUniforms(BasicBlock *block, const SmallVectorImpl<Instruction *> &initialReadFirstLanes);
  void findBestInsertLocation(const SmallVectorImpl<Instruction *> &initialReadFirstLanes);
  bool isAllUsersAssumedUniform(Instruction *inst);
  void applyReadFirstLane(Instruction *inst, BuilderBase &builder);

  // We only support to apply amdgcn_readfirstlane on float or int type
  // TODO: Support various types when backend work is ready
  bool isSupportedType(Instruction *inst) { return inst->getType()->isFloatTy() || inst->getType()->isIntegerTy(32); }

  LegacyDivergenceAnalysis *m_divergenceAnalysis; // The divergence analysis
  TargetTransformInfo *m_targetTransformInfo;     // The target tranform info to determine stop propagation
  DenseMap<Instruction *, SmallVector<Instruction *, 2>>
      m_canAssumeUniformDivergentUsesMap; // The map key is an instruction `I` that can be assumed uniform. The map
                                          // value is a vector of instructions that we can apply readfirstlane on them.
                                          // An empty vector means the instruction `I` is an insert location.
  DenseSet<Instruction *> m_insertLocations; // The insert locations of readfirstlane
};

} // anonymous namespace

// =====================================================================================================================
// Initializes static members.
char PatchReadFirstLane::ID = 0;

// =====================================================================================================================
// Pass creator, creates the pass of LLVM patching operations for readfirstlane optimizations.
FunctionPass *lgc::createPatchReadFirstLane() {
  return new PatchReadFirstLane();
}

// =====================================================================================================================
// Returns true if all users of the given instruction defined in the given block.
//
// @param inst : The given instruction
// @param block : The given block
static bool isAllUsersDefinedInBlock(Instruction *inst, BasicBlock *block) {
  for (auto user : inst->users()) {
    if (auto userInst = dyn_cast<Instruction>(user))
      if (userInst->getParent() != block)
        return false;
  }
  return true;
}

// =====================================================================================================================
// Returns true if all users of the given instruction are already readfirstlane
//
// @param inst : The given instruction
static bool areAllUserReadFirstLane(Instruction *inst) {
  for (auto user : inst->users()) {
    if (isa<DbgInfoIntrinsic>(user))
      continue;
    auto intrinsic = dyn_cast<IntrinsicInst>(user);
    if (!intrinsic || intrinsic->getIntrinsicID() != Intrinsic::amdgcn_readfirstlane)
      return false;
  }
  return true;
}

PatchReadFirstLane::PatchReadFirstLane() : FunctionPass(ID), m_targetTransformInfo(nullptr) {
}

// =====================================================================================================================
// Executes this LLVM pass on the specified LLVM function.
//
// @param [in,out] function : LLVM function to be run on.
bool PatchReadFirstLane::runOnFunction(Function &function) {
  LLVM_DEBUG(dbgs() << "Run the pass Patch-Read-First-Lane\n");

  m_divergenceAnalysis = &getAnalysis<LegacyDivergenceAnalysis>();
  auto *targetTransformInfoWrapperPass = getAnalysisIfAvailable<TargetTransformInfoWrapperPass>();
  if (targetTransformInfoWrapperPass)
    m_targetTransformInfo = &targetTransformInfoWrapperPass->getTTI(function);
  assert(m_targetTransformInfo);

  const bool changed = liftReadFirstLane(function);

  return changed;
}

// =====================================================================================================================
// Specify what analysis passes this pass depends on.
//
// @param [in,out] analysisUsage : The place to record our analysis pass usage requirements.
void PatchReadFirstLane::getAnalysisUsage(AnalysisUsage &analysisUsage) const {
  analysisUsage.addRequired<LegacyDivergenceAnalysis>();
  analysisUsage.setPreservesCFG();
}

// =====================================================================================================================
// Lift readfirstlanes in relevant basic blocks to transform VGPR instructions into SGPR instructions as many as
// possible.
//
// @param [in,out] function : LLVM function to be run for readfirstlane optimization.
bool PatchReadFirstLane::liftReadFirstLane(Function &function) {
  // Collect the basic blocks with amdgcn_readfirstlane
  // Build the map between initial readfirstlanes and their corrensponding blocks
  DenseMap<BasicBlock *, SmallVector<Instruction *, 2>> blockInitialReadFirstLanesMap;
  Module *module = function.getParent();
  for (auto &func : *module) {
    if (func.getIntrinsicID() == Intrinsic::amdgcn_readfirstlane) {
      for (User *user : func.users()) {
        Instruction *inst = cast<Instruction>(user);
        if (inst->getFunction() != &function)
          continue;
        blockInitialReadFirstLanesMap[inst->getParent()].push_back(inst);
      }
      break;
    }
  }
  bool changed = false;

  // Lift readfirstlanes in each relevant basic block
  for (auto blockInitialReadFirstLanes : blockInitialReadFirstLanesMap) {
    BasicBlock *curBb = blockInitialReadFirstLanes.first;

    // Step 1: Collect all instructions that "can be assumed uniform" with its divergent uses in a map
    // (m_canAssumeUniformDivergentUsesMap)
    collectAssumeUniforms(curBb, blockInitialReadFirstLanes.second);

    // Step 2: Determine the best places to insert readfirstlane according to a heuristic
    findBestInsertLocation(blockInitialReadFirstLanes.second);

    // Step 3: Apply readFirstLane on all determined locations
    assert(m_insertLocations.size() <= blockInitialReadFirstLanes.second.size());
    BuilderBase builder(curBb->getContext());
    for (auto inst : m_insertLocations) {
      // Avoid to insert reduncant readfirstlane
      if (auto intrinsic = dyn_cast<IntrinsicInst>(inst))
        if (intrinsic->getIntrinsicID() == Intrinsic::amdgcn_readfirstlane)
          continue;
      if (areAllUserReadFirstLane(inst))
        continue;

      applyReadFirstLane(inst, builder);
      changed = true;
    }
  }
  return changed;
}

// =====================================================================================================================
// Collect the instructions that can be "assumed uniform" if the value itself isn't known to be uniform (by
// DivergenceAnalysis), but we can replace all of its uses by readfirstlane(V) and still be correct
//
// @param block : The processing basic block
// @param initialReadFirstLanes : The initial amdgcn_readfirstlane vector
void PatchReadFirstLane::collectAssumeUniforms(BasicBlock *block,
                                               const SmallVectorImpl<Instruction *> &initialReadFirstLanes) {
  auto instructionOrder = [](Instruction* lhs, Instruction* rhs) { return lhs->comesBefore(rhs); };
  SmallVector<Instruction *, 16> candidates;

  auto insertCandidate = [&](Instruction *candidate) {
    auto insertPos = llvm::lower_bound(candidates, candidate, instructionOrder);
    if (insertPos == candidates.end() || *insertPos != candidate)
      candidates.insert(insertPos, candidate);
  };

  // The given instruction can be assumed to have a uniform result, i.e., replacing its uses by a use of a
  // readfirstlane of it would be correct. This helper function:
  //  1. Records this fact and
  //  2. Determines whether the assumption of a uniform result could be propagated to the candidate's operands.
  auto tryPropagate = [&](Instruction *candidate) {
    // Are there reasons inherent to the candidate instruction itself why lifting the readfirstlane even further
    // isn't possible?
    bool cannotPropagate =
        m_targetTransformInfo->isSourceOfDivergence(candidate) ||
        isa<PHINode>(candidate);

    SmallVector<Instruction *, 3> operandInsts;
    if (!cannotPropagate) {
      for (Use &use : candidate->operands()) {
        if (!m_divergenceAnalysis->isDivergentUse(&use))
          continue; // already known to be uniform -- no need to consider this operand

        auto operandInst = dyn_cast<Instruction>(use.get());
        if (!operandInst) {
          // Known to be divergent, but not an instruction. Further propagation is currently not implemented.
          assert(isa<Argument>(operandInst));
          cannotPropagate = true;
          break;
        }

        if (operandInst->getParent() != block || !isAllUsersDefinedInBlock(operandInst, block)) {
          // Further propagation is currently not implemented. Theoretically, we could insert a readfirstlane
          // instruction dedicated for users in this basic block, but it's not clear whether that would be a win.
          cannotPropagate = true;
          break;
        }

        operandInsts.push_back(operandInst);
      }

      if (cannotPropagate)
        operandInsts.clear();
    }

    assert(m_canAssumeUniformDivergentUsesMap.count(candidate) == 0);
    m_canAssumeUniformDivergentUsesMap.try_emplace(candidate, std::move(operandInsts));

    for (Instruction *operandInst : operandInsts)
      insertCandidate(operandInst);
  };

  for (auto readfirstlane : initialReadFirstLanes)
    tryPropagate(readfirstlane);

  while (!candidates.empty()) {
    Instruction *candidate = candidates.pop_back_val();

    if (isAllUsersAssumedUniform(candidate))
      tryPropagate(candidate);
  }
}

// =====================================================================================================================
// Find the best insert locations according to the heuristic.
// The heuristic is: if an instruction that can be assumed to be uniform has multiple divergent operands, then you take
// the definition of the divergent operand that is earliest in basic block order (call it "the earliest divergent
// operand") and propagate up to that instruction; if it turns out that that instruction can be assumed to be uniform,
// then we can just insert the readfirstlane there (or propagate).
//
// @param readFirstLaneCount : The initial amdgcn_readfirstlane vector
void PatchReadFirstLane::findBestInsertLocation(const SmallVectorImpl<Instruction *> &initialReadFirstLanes) {
  // Each key of m_canAssumeUniformDivergentUsesMap is candidated for applying readfirstlane. The principle is to lift
  // readfirstlane to a best insert location without introducing extra readfirstlane.

  // For single divergent operand case, we can easily find %a as a valid insert location starting from %d.
  // %a = call...
  // %b = add i32 %a, 10.
  // %c = mul i32 %b, 3.
  // %d = readfirstlane i32 %c.
  // Search chain d->c->b->a: key %a is related to an empty value so that %a is a valid insert location.

  // For multiple divergent operands case that the search chain involves `nonEarliestDivergentUses` to help determine
  // whether multiple divergent uses directly or indirectly depend on %a. The search chain starts from %g and goes up to
  // each earliest come operand.
  // %a = call ...
  // %b = add i32 %a, 10.
  // %c = mul i32 %a, 3.
  // %d = add i32 %b, %c.
  // %e = sub i32 %c,4.
  // %f = add i32 %d, %e.
  // %g = readfirstlane i32 %f.
  // Search chain g->f->d->b->a: nonEarliestDivergentUses {e,c} - stop search up because key %a is related to an empty
  // value. {e,c} required insert location is %a. So we can lift readfirstlane on %a.

  // Set of instructions from m_canAssumeUniformDivergentUsesMap which will be forced to become uniform by the
  // instructions we already plan to insert so far. Allows us to break out of searches that would be redundant.
  DenseSet<Instruction *> enforcedUniform;
  SmallVector<Instruction *, 8> enforcedUniformTracker;

  m_insertLocations.clear();

  for (auto &initialReadFirstLane : initialReadFirstLanes) {
    // Find a best insert location for a lifted readfirstlane to obsolete the existing, initial readfirstlane.
    // Conceptually, we trace backwards through the induced data dependency graph (or "cone") of
    // divergent-but-can-assume-uniform instructions feeding into the initialReadFirstLane.
    // Each iteration of the middle loop jumps to the next "bottleneck" in this DAG, that is, `current` always points
    // at a bottleneck where we could insert a single readfirstlane (depending on the type).
    Instruction *bestInsertLocation = nullptr;
    unsigned bestInsertLocationDepth = 0;

    Instruction *current = initialReadFirstLane;

    for (;;) {
      const auto &divergentOperands = m_canAssumeUniformDivergentUsesMap.find(current)->second;
      if (divergentOperands.empty())
        break; // no further propagation possible

      if (divergentOperands.size() == 1) {
        // There is only a single operand, we can jump to it directly.
        current = divergentOperands[0];
      } else {
        // There are multiple operands. Since we don't want to increase the number of readfirstlanes, try to find
        // an earlier bottleneck in the data dependency graph.
        //
        // The search proceeds backwards by instruction order in the basic block, maintaining a sorted queue of
        // instructions that remain to be explored. We use two heuristics to limit the cost of the search:
        //  - We never explore beyond the earliest operand of `current`.
        //  - We limit both the depth and the breadth (i.e., maximum queue size) of the search.
        //
        // We maintain the queue as a vector because it will always be short, and inserting into a short sorted
        // vector is very fast.
        constexpr unsigned MaxSearchBreadth = 4;
        constexpr unsigned MaxSearchDepth = 10;
        auto instructionOrder = [](Instruction* lhs, Instruction* rhs) { return lhs->comesBefore(rhs); };

        if (divergentOperands.size() > MaxSearchBreadth)
          break;

        SmallVector<Instruction *, MaxSearchBreadth> queue;
        queue.insert(queue.begin(), divergentOperands.begin(), divergentOperands.end());
        llvm::sort(queue, instructionOrder);

        bool searchAborted = false;
        unsigned depth = 0;
        do {
          Instruction* candidate = queue.back();
          if (enforcedUniform.count(candidate)) {
            // Candidate is already enforced to be uniform by a previous decision to insert a readfirstlane.
            // We can just skip it.
            queue.pop_back();
            continue;
          }

          const auto &candidateOperands = m_canAssumeUniformDivergentUsesMap.find(candidate)->second;
          if (candidateOperands.empty())
            break; // no further propagation possible, need to abort the search
          queue.pop_back();

          enforcedUniformTracker.push_back(candidate);

          // Add the operands to the queue if they aren't already contained in it.
          for (Instruction* operand : candidateOperands) {
            auto insertPos = llvm::lower_bound(queue, operand, instructionOrder);
            if (insertPos == queue.end() || *insertPos != operand) {
              // Abort if the search becomes too "wide" or moves beyond the earliest operand of `current`.
              if (queue.size() >= MaxSearchBreadth || insertPos == queue.begin()) {
                searchAborted = true;
                break;
              }
              queue.insert(insertPos, operand);
            }
          }

          if (++depth > MaxSearchDepth)
            break;
        } while (queue.size() >= 2 && !searchAborted);

        if (queue.size() >= 2)
          break; // didn't find a next bottleneck in the data dependency graph, bail out

        current = queue[0]; // move to the found bottleneck
      }

      if (enforcedUniform.count(current)) {
        // Already enforced to be uniform, no need to continue the search or even consider inserting a new
        // readfirstlane.
        bestInsertLocation = nullptr;
        break;
      }

      enforcedUniformTracker.push_back(current);

      if (isSupportedType(current)) {
        bestInsertLocation = current;
        bestInsertLocationDepth = enforcedUniformTracker.size();
      }
    }

    // Record the best (read: earliest) bottleneck that we were able to find in the graph/
    if (bestInsertLocation) {
      m_insertLocations.insert(bestInsertLocation);

      for (unsigned idx = 0; idx < bestInsertLocationDepth; ++idx)
        enforcedUniform.insert(enforcedUniformTracker[idx]);
    }

    enforcedUniformTracker.clear();
  }
}

// =====================================================================================================================
// Return true if all users of the given instruction are "assumed uniform"
//
// @param inst : The instruction to be checked
bool PatchReadFirstLane::isAllUsersAssumedUniform(Instruction *inst) {
  for (auto user : inst->users()) {
    auto userInst = dyn_cast<Instruction>(user);
    if (m_canAssumeUniformDivergentUsesMap.count(userInst) == 0)
      return false;
  }
  return true;
}

// =====================================================================================================================
// Try to apply readfirstlane on the given instruction
//
// @param inst : The instruction to be applied readfirstlane on
// @param builder : BuildBase to use
void PatchReadFirstLane::applyReadFirstLane(Instruction *inst, BuilderBase &builder) {
  // Guarantee the insert position is behind all PhiNodes
  Instruction *insertPos = inst->getNextNonDebugInstruction();
  while (dyn_cast<PHINode>(insertPos))
    insertPos = insertPos->getNextNonDebugInstruction();
  builder.SetInsertPoint(insertPos);

  Type *instTy = inst->getType();
  const bool isFloat = instTy->isFloatTy();
  assert(isFloat || instTy->isIntegerTy(32));
  Value *newInst = inst;
  if (isFloat)
    newInst = builder.CreateBitCast(inst, builder.getInt32Ty());

  Value *readFirstLane = builder.CreateIntrinsic(Intrinsic::amdgcn_readfirstlane, {}, newInst);

  Value *replaceInst = nullptr;
  if (isFloat) {
    replaceInst = builder.CreateBitCast(readFirstLane, instTy);
  } else {
    newInst = readFirstLane;
    replaceInst = readFirstLane;
  }
  inst->replaceUsesWithIf(replaceInst, [newInst](Use &U) { return U.getUser() != newInst; });
}

// =====================================================================================================================
// Initializes the pass of LLVM patching operations for readfirstlane optimizations.
INITIALIZE_PASS_BEGIN(PatchReadFirstLane, DEBUG_TYPE, "Patch LLVM for readfirstlane optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(PatchReadFirstLane, DEBUG_TYPE, "Patch LLVM for readfirstlane optimizations", false, false)
