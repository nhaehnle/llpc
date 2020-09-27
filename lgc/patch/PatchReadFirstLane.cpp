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
  bool isAllUsesDependCandidate(Instruction *candidateInsertLocatin,
                                const DenseSet<Instruction *> &nonEarliestDivergentUses,
                                DenseSet<Instruction *> &visitedInsts);
  bool isAllUsersAssumedUniform(Instruction *inst);
  void applyReadFirstLane(Instruction *inst, BuilderBase &builder);
  void addLocation(Instruction *inst);

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
  for (auto readfirstlane : initialReadFirstLanes)
    m_canAssumeUniformDivergentUsesMap.insert({readfirstlane, {}});

  for (Instruction &inst : llvm::reverse(*block)) {
    // Skip readfirstlane-irrelevant instructions or a source of divergence which isn't clear to determine its operands
    if (m_canAssumeUniformDivergentUsesMap.count(&inst) == 0 || (m_targetTransformInfo->isSourceOfDivergence(&inst)))
      continue;

    // For each operand, see if it can be assumed uniform to be inserted as the key of
    // m_canAssumeUniformDivergentUsesMap. For all operands, see if they all can be readfirstlane-ified and then collect
    // them as the value of m_canAssumeUniformDivergentUsesMap Readfirstlane-ified means that the operand is a key of
    // the map and not fat pointer.
    bool canCollectOperands = true;
    auto loadInst = dyn_cast<LoadInst>(&inst);
    SmallVector<Instruction *, 3> operandInsts;
    for (Use &use : inst.operands()) {
      if (!m_divergenceAnalysis->isDivergentUse(&use))
        continue;
      auto operandInst = dyn_cast<Instruction>(use.get());
      if (!operandInst)
        continue;
      // Skip the operandInst that is already in the map
      if (m_canAssumeUniformDivergentUsesMap.count(operandInst) == 1)
        continue;
      if (operandInst->getParent() != block || !isAllUsersDefinedInBlock(operandInst, block))
        continue;

      if (isAllUsersAssumedUniform(operandInst)) {
        // The operand can be assumed uniform becasue all users are assumed uniform
        m_canAssumeUniformDivergentUsesMap[operandInst] = {};
        // NOTE: Stop propagation for a loadInst whose address space is fat pointer because it has uncertain size
        if (loadInst && loadInst->getPointerAddressSpace() == ADDR_SPACE_BUFFER_FAT_POINTER)
          canCollectOperands = false;
      } else {
        canCollectOperands = false;
      }
      operandInsts.push_back(operandInst);
    }
    if (canCollectOperands)
      m_canAssumeUniformDivergentUsesMap[&inst] = std::move(operandInsts);
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

  m_insertLocations.clear();
  // Avoid to process the same instruction mulitple times.
  DenseSet<Instruction *> visitedInsts;

  // Find a best insert location for each initial readfirstlane.
  for (auto &initialReadFirstLane : initialReadFirstLanes) {
    // Collect the divergent operands of the `candidateInst` exclude the earliest come operand in basic bloc order.
    DenseSet<Instruction *> nonEarliestDivergentUses;
    // Record the best insert location if it is valid.
    Instruction *insertLocation = nullptr;
    // Mark whether the search chain involves multiple divergent operands.
    bool hasMultipleDivergentOperands = false;
    // The earliest come operands are candidate for best location.
    // Record the search chain of candidate instructions and start from the initial readfirstlane intrinsic.
    std::deque<Instruction *> candidateInstDeque;
    candidateInstDeque.push_back(initialReadFirstLane);

    // The search isn't stopped until the instruction has no divergent operand.
    while (!candidateInstDeque.empty()) {
      auto candidateInst = candidateInstDeque.front();
      candidateInstDeque.pop_front();

      // Skip the processed candidate.
      if (visitedInsts.count(candidateInst) == 1)
        continue;
      visitedInsts.insert(candidateInst);

      // The value of m_canAssumeUniformDivergentUsesMap is empty means we cannot lift readfirstlane over
      // `candidateInst`. `candidateInst` can be a valid insert location if the search chain doen't involve multiple
      // operands or all uses directly or indirectly depend on `candidateInst`.
      auto &divergentOperandsOfCandidate = m_canAssumeUniformDivergentUsesMap[candidateInst];
      const unsigned divergentOperandCount = divergentOperandsOfCandidate.size();
      if (divergentOperandCount == 0) {
        bool isValidInsertLocation = true;
        if (hasMultipleDivergentOperands)
          isValidInsertLocation = isAllUsesDependCandidate(candidateInst, nonEarliestDivergentUses, visitedInsts);
        if (isValidInsertLocation)
          insertLocation = candidateInst;
        break;
      }

      if (divergentOperandCount > 1)
        hasMultipleDivergentOperands = true;

      // `earliestDivergentDef` represents the instruction that comes first in basic block order among all the operands
      Instruction *earliestDivergentDef = divergentOperandsOfCandidate[0];
      for (unsigned idx = 1; idx < divergentOperandCount; ++idx) {
        auto divergentDef = divergentOperandsOfCandidate[idx];
        if (divergentDef->comesBefore(earliestDivergentDef))
          earliestDivergentDef = divergentDef;
      }
      candidateInstDeque.push_back(earliestDivergentDef);

      if (hasMultipleDivergentOperands) {
        // Collect the divergent operands (exclude earliestDivergentDef) of `candidateInst`.
        for (auto operand : divergentOperandsOfCandidate) {
          if (operand != earliestDivergentDef)
            nonEarliestDivergentUses.insert(operand);
        }
      }
    }
    if (insertLocation)
      addLocation(insertLocation);
  }
}

// =====================================================================================================================
// Use a queue to track back each non earliest divergent use. Stop search if the use is non-uniform (i.e., not found in
// m_canAssumeUniformDivergentUsesMap) or it is a key but corresponding an empty vector. Return true if there is no
// non-uniform and the stopped location is the candidate insert location.
//
// @param candidateInsertLocation : The candidate insert location.
// @param nonEarliestDivergentUses : The set of divergent uses exclude all earliest come operands.
// @param visitedInsts : The visited instruction set.
bool PatchReadFirstLane::isAllUsesDependCandidate(Instruction *candidateInsertLocation,
                                                  const DenseSet<Instruction *> &nonEarliestDivergentUses,
                                                  DenseSet<Instruction *> &visitedInsts) {
  DenseSet<Instruction *> extraInsertLocations;
  // Each instruction on the search chain should be found in the map otherwise the uses cannot depend on one candidate
  bool hasNonUniform = false;
  for (auto divergentUse : nonEarliestDivergentUses) {
    std::deque<Instruction *> searchChain;
    searchChain.push_back(divergentUse);

    while (!searchChain.empty()) {
      Instruction *curInst = searchChain.front();
      searchChain.pop_front();

      if (visitedInsts.count(curInst) == 1)
        continue;
      visitedInsts.insert(curInst);

      for (Use &operand : curInst->operands()) {
        if (auto operandInst = dyn_cast<Instruction>(operand)) {
          if (m_canAssumeUniformDivergentUsesMap.count(operandInst) == 0) {
            hasNonUniform = true;
            break;
          }
          const auto &divergentOperandsOfCandidate = m_canAssumeUniformDivergentUsesMap[operandInst];
          for (auto divergentOperand : divergentOperandsOfCandidate) {
            if (m_canAssumeUniformDivergentUsesMap[divergentOperand].empty()) {
              // The operand should be an insert location because it has no divergent use
              extraInsertLocations.insert(divergentOperand);
            } else {
              // Continue to search up
              searchChain.push_back(divergentOperand);
            }
          }
        }
      }
      if (hasNonUniform)
        break;
    }
    if (hasNonUniform)
      break;
  }

  bool isValidInsertLocation = false;
  if (!hasNonUniform) {
    const unsigned extraInsertLocationCount = extraInsertLocations.size();
    if (extraInsertLocationCount == 0 ||
        (extraInsertLocationCount == 1 && (candidateInsertLocation == *extraInsertLocations.begin())))
      isValidInsertLocation = true;
  }
  return isValidInsertLocation;
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
//
// @param inst : The instruction to be as a best insert location
void PatchReadFirstLane::addLocation(Instruction *inst) {
  if (isSupportedType(inst))
    m_insertLocations.insert(inst);
}

// =====================================================================================================================
// Initializes the pass of LLVM patching operations for readfirstlane optimizations.
INITIALIZE_PASS_BEGIN(PatchReadFirstLane, DEBUG_TYPE, "Patch LLVM for readfirstlane optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(PatchReadFirstLane, DEBUG_TYPE, "Patch LLVM for readfirstlane optimizations", false, false)
