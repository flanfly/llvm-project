//===- PanopticonInstrEmitter.cpp - Generate a Instruction Set Desc. --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#include "CodeGenDAGPatterns.h"
#include "CodeGenInstruction.h"
#include "CodeGenSchedule.h"
#include "CodeGenTarget.h"
#include "PredicateExpander.h"
#include "SequenceToOffsetTable.h"
#include "TableGenBackends.h"
#include "X86DisassemblerTables.h"
#include "X86RecognizableInstr.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cassert>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <exception>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <sstream>

using namespace llvm;

namespace {

class PanopticonInstrEmitter {
  RecordKeeper &Records;
  CodeGenDAGPatterns CDP;
  const CodeGenSchedModels &SchedModels;

public:
  PanopticonInstrEmitter(RecordKeeper &R):
    Records(R), CDP(R), SchedModels(CDP.getTargetInfo().getSchedModels()) {}

  // run - Output the instruction set description.
  void run(raw_ostream &OS);
  void emitSemantics(raw_ostream &OS);
  void emitInstructionEnum(raw_ostream &OS);

private:
  typedef std::map<std::vector<std::string>, unsigned> OperandInfoMapTy;

  /// The keys of this map are maps which have OpName enum values as their keys
  /// and instruction operand indices as their values.  The values of this map
  /// are lists of instruction names.
  typedef std::map<std::map<unsigned, unsigned>,
                   std::vector<std::string>> OpNameMapTy;
  typedef std::map<std::string, unsigned>::iterator StrUintMapIter;

  /// Generate member functions in the target-specific GenInstrInfo class.
  ///
  /// This method is used to custom expand TIIPredicate definitions.
  /// See file llvm/Target/TargetInstPredicates.td for a description of what is
  /// a TIIPredicate and how to use it.
  void emitTIIHelperMethods(raw_ostream &OS, StringRef TargetName,
                            bool ExpandDefinition = true);

  /// Expand TIIPredicate definitions to functions that accept a const MCInst
  /// reference.
  void emitMCIIHelperMethods(raw_ostream &OS, StringRef TargetName);
  void emitOperandTypesEnum(raw_ostream &OS, const CodeGenTarget &Target);
  void initOperandMapData(
            ArrayRef<const CodeGenInstruction *> NumberedInstructions,
            StringRef Namespace,
            std::map<std::string, unsigned> &Operands,
            OpNameMapTy &OperandMap);
  void emitOperandNameMappings(raw_ostream &OS, const CodeGenTarget &Target,
            ArrayRef<const CodeGenInstruction*> NumberedInstructions);

  // Operand information.
  void EmitOperandInfo(raw_ostream &OS, OperandInfoMapTy &OperandInfoIDs);
  std::vector<std::string> GetOperandInfo(const CodeGenInstruction &Inst);
};

} // end anonymous namespace

static void PrintDefList(const std::vector<Record*> &Uses,
                         unsigned Num, raw_ostream &OS) {
  OS << "static const MCPhysReg ImplicitList" << Num << "[] = { ";
  for (Record *U : Uses)
    OS << getQualifiedName(U) << ", ";
  OS << "0 };\n";
}

//===----------------------------------------------------------------------===//
// Operand Info Emission.
//===----------------------------------------------------------------------===//

std::vector<std::string>
PanopticonInstrEmitter::GetOperandInfo(const CodeGenInstruction &Inst) {
  std::vector<std::string> Result;

  for (auto &Op : Inst.Operands) {
    // Handle aggregate operands and normal operands the same way by expanding
    // either case into a list of operands for this op.
    std::vector<CGIOperandList::OperandInfo> OperandList;

    // This might be a multiple operand thing.  Targets like X86 have
    // registers in their multi-operand operands.  It may also be an anonymous
    // operand, which has a single operand, but no declared class for the
    // operand.
    DagInit *MIOI = Op.MIOperandInfo;

    if (!MIOI || MIOI->getNumArgs() == 0) {
      // Single, anonymous, operand.
      OperandList.push_back(Op);
    } else {
      for (unsigned j = 0, e = Op.MINumOperands; j != e; ++j) {
        OperandList.push_back(Op);

        auto *OpR = cast<DefInit>(MIOI->getArg(j))->getDef();
        OperandList.back().Rec = OpR;
      }
    }

    for (unsigned j = 0, e = OperandList.size(); j != e; ++j) {
      Record *OpR = OperandList[j].Rec;
      std::string Res;

      if (OpR->isSubClassOf("RegisterOperand"))
        OpR = OpR->getValueAsDef("RegClass");
      if (OpR->isSubClassOf("RegisterClass"))
        Res += getQualifiedName(OpR) + "RegClassID, ";
      else if (OpR->isSubClassOf("PointerLikeRegClass"))
        Res += utostr(OpR->getValueAsInt("RegClassKind")) + ", ";
      else
        // -1 means the operand does not have a fixed register class.
        Res += "-1, ";

      // Fill in applicable flags.
      Res += "0";

      // Ptr value whose register class is resolved via callback.
      if (OpR->isSubClassOf("PointerLikeRegClass"))
        Res += "|(1<<MCOI::LookupPtrRegClass)";

      // Predicate operands.  Check to see if the original unexpanded operand
      // was of type PredicateOp.
      if (Op.Rec->isSubClassOf("PredicateOp"))
        Res += "|(1<<MCOI::Predicate)";

      // Optional def operands.  Check to see if the original unexpanded operand
      // was of type OptionalDefOperand.
      if (Op.Rec->isSubClassOf("OptionalDefOperand"))
        Res += "|(1<<MCOI::OptionalDef)";

      // Fill in operand type.
      Res += ", ";
      assert(!Op.OperandType.empty() && "Invalid operand type.");
      Res += Op.OperandType;

      // Fill in constraint info.
      Res += ", ";

      const CGIOperandList::ConstraintInfo &Constraint =
        Op.Constraints[j];
      if (Constraint.isNone())
        Res += "0";
      else if (Constraint.isEarlyClobber())
        Res += "(1 << MCOI::EARLY_CLOBBER)";
      else {
        assert(Constraint.isTied());
        Res += "((" + utostr(Constraint.getTiedOperand()) +
                    " << 16) | (1 << MCOI::TIED_TO))";
      }

      Result.push_back(Res);
    }
  }

  return Result;
}

void PanopticonInstrEmitter::EmitOperandInfo(raw_ostream &OS,
                                       OperandInfoMapTy &OperandInfoIDs) {
  // ID #0 is for no operand info.
  unsigned OperandListNum = 0;
  OperandInfoIDs[std::vector<std::string>()] = ++OperandListNum;

  OS << "\n";
  const CodeGenTarget &Target = CDP.getTargetInfo();
  for (const CodeGenInstruction *Inst : Target.getInstructionsByEnumValue()) {
    std::vector<std::string> OperandInfo = GetOperandInfo(*Inst);
    unsigned &N = OperandInfoIDs[OperandInfo];
    if (N != 0) continue;

    N = ++OperandListNum;
    OS << "static const MCOperandInfo OperandInfo" << N << "[] = { ";
    for (const std::string &Info : OperandInfo)
      OS << "{ " << Info << " }, ";
    OS << "};\n";
  }
}

/// Initialize data structures for generating operand name mappings.
/// 
/// \param Operands [out] A map used to generate the OpName enum with operand
///        names as its keys and operand enum values as its values.
/// \param OperandMap [out] A map for representing the operand name mappings for
///        each instructions.  This is used to generate the OperandMap table as
///        well as the getNamedOperandIdx() function.
void PanopticonInstrEmitter::initOperandMapData(
        ArrayRef<const CodeGenInstruction *> NumberedInstructions,
        StringRef Namespace,
        std::map<std::string, unsigned> &Operands,
        OpNameMapTy &OperandMap) {
  unsigned NumOperands = 0;
  for (const CodeGenInstruction *Inst : NumberedInstructions) {
    if (!Inst->TheDef->getValueAsBit("UseNamedOperandTable"))
      continue;
    std::map<unsigned, unsigned> OpList;
    for (const auto &Info : Inst->Operands) {
      StrUintMapIter I = Operands.find(Info.Name);

      if (I == Operands.end()) {
        I = Operands.insert(Operands.begin(),
                    std::pair<std::string, unsigned>(Info.Name, NumOperands++));
      }
      OpList[I->second] = Info.MIOperandNo;
    }
    OperandMap[OpList].push_back(Namespace.str() + "::" +
                                 Inst->TheDef->getName().str());
  }
}

/// Generate a table and function for looking up the indices of operands by
/// name.
///
/// This code generates:
/// - An enum in the llvm::TargetNamespace::OpName namespace, with one entry
///   for each operand name.
/// - A 2-dimensional table called OperandMap for mapping OpName enum values to
///   operand indices.
/// - A function called getNamedOperandIdx(uint16_t Opcode, uint16_t NamedIdx)
///   for looking up the operand index for an instruction, given a value from
///   OpName enum
void PanopticonInstrEmitter::emitOperandNameMappings(raw_ostream &OS,
           const CodeGenTarget &Target,
           ArrayRef<const CodeGenInstruction*> NumberedInstructions) {
  StringRef Namespace = Target.getInstNamespace();
  std::string OpNameNS = "OpName";
  // Map of operand names to their enumeration value.  This will be used to
  // generate the OpName enum.
  std::map<std::string, unsigned> Operands;
  OpNameMapTy OperandMap;

  initOperandMapData(NumberedInstructions, Namespace, Operands, OperandMap);

  OS << "#ifdef GET_INSTRINFO_OPERAND_ENUM\n";
  OS << "#undef GET_INSTRINFO_OPERAND_ENUM\n";
  OS << "namespace llvm {\n";
  OS << "namespace " << Namespace << " {\n";
  OS << "namespace " << OpNameNS << " {\n";
  OS << "enum {\n";
  for (const auto &Op : Operands)
    OS << "  " << Op.first << " = " << Op.second << ",\n";

  OS << "OPERAND_LAST";
  OS << "\n};\n";
  OS << "} // end namespace OpName\n";
  OS << "} // end namespace " << Namespace << "\n";
  OS << "} // end namespace llvm\n";
  OS << "#endif //GET_INSTRINFO_OPERAND_ENUM\n\n";

  OS << "#ifdef GET_INSTRINFO_NAMED_OPS\n";
  OS << "#undef GET_INSTRINFO_NAMED_OPS\n";
  OS << "namespace llvm {\n";
  OS << "namespace " << Namespace << " {\n";
  OS << "LLVM_READONLY\n";
  OS << "int16_t getNamedOperandIdx(uint16_t Opcode, uint16_t NamedIdx) {\n";
  if (!Operands.empty()) {
    OS << "  static const int16_t OperandMap [][" << Operands.size()
       << "] = {\n";
    for (const auto &Entry : OperandMap) {
      const std::map<unsigned, unsigned> &OpList = Entry.first;
      OS << "{";

      // Emit a row of the OperandMap table
      for (unsigned i = 0, e = Operands.size(); i != e; ++i)
        OS << (OpList.count(i) == 0 ? -1 : (int)OpList.find(i)->second) << ", ";

      OS << "},\n";
    }
    OS << "};\n";

    OS << "  switch(Opcode) {\n";
    unsigned TableIndex = 0;
    for (const auto &Entry : OperandMap) {
      for (const std::string &Name : Entry.second)
        OS << "  case " << Name << ":\n";

      OS << "    return OperandMap[" << TableIndex++ << "][NamedIdx];\n";
    }
    OS << "    default: return -1;\n";
    OS << "  }\n";
  } else {
    // There are no operands, so no need to emit anything
    OS << "  return -1;\n";
  }
  OS << "}\n";
  OS << "} // end namespace " << Namespace << "\n";
  OS << "} // end namespace llvm\n";
  OS << "#endif //GET_INSTRINFO_NAMED_OPS\n\n";
}

/// Generate an enum for all the operand types for this target, under the
/// llvm::TargetNamespace::OpTypes namespace.
/// Operand types are all definitions derived of the Operand Target.td class.
void PanopticonInstrEmitter::emitOperandTypesEnum(raw_ostream &OS,
                                            const CodeGenTarget &Target) {

  StringRef Namespace = Target.getInstNamespace();
  std::vector<Record *> Operands = Records.getAllDerivedDefinitions("Operand");

  OS << "#ifdef GET_INSTRINFO_OPERAND_TYPES_ENUM\n";
  OS << "#undef GET_INSTRINFO_OPERAND_TYPES_ENUM\n";
  OS << "namespace llvm {\n";
  OS << "namespace " << Namespace << " {\n";
  OS << "namespace OpTypes {\n";
  OS << "enum OperandType {\n";

  unsigned EnumVal = 0;
  for (const Record *Op : Operands) {
    if (!Op->isAnonymous())
      OS << "  " << Op->getName() << " = " << EnumVal << ",\n";
    ++EnumVal;
  }

  OS << "  OPERAND_TYPE_LIST_END" << "\n};\n";
  OS << "} // end namespace OpTypes\n";
  OS << "} // end namespace " << Namespace << "\n";
  OS << "} // end namespace llvm\n";
  OS << "#endif // GET_INSTRINFO_OPERAND_TYPES_ENUM\n\n";
}

void PanopticonInstrEmitter::emitMCIIHelperMethods(raw_ostream &OS,
                                             StringRef TargetName) {
  RecVec TIIPredicates = Records.getAllDerivedDefinitions("TIIPredicate");
  if (TIIPredicates.empty())
    return;

  OS << "#ifdef GET_INSTRINFO_MC_HELPER_DECLS\n";
  OS << "#undef GET_INSTRINFO_MC_HELPER_DECLS\n\n";

  OS << "namespace llvm {\n";
  OS << "class MCInst;\n\n";

  OS << "namespace " << TargetName << "_MC {\n\n";

  for (const Record *Rec : TIIPredicates) {
    OS << "bool " << Rec->getValueAsString("FunctionName")
        << "(const MCInst &MI);\n";
  }

  OS << "\n} // end " << TargetName << "_MC namespace\n";
  OS << "} // end llvm namespace\n\n";

  OS << "#endif // GET_INSTRINFO_MC_HELPER_DECLS\n\n";

  OS << "#ifdef GET_INSTRINFO_MC_HELPERS\n";
  OS << "#undef GET_INSTRINFO_MC_HELPERS\n\n";

  OS << "namespace llvm {\n";
  OS << "namespace " << TargetName << "_MC {\n\n";

  PredicateExpander PE(TargetName);
  PE.setExpandForMC(true);

  for (const Record *Rec : TIIPredicates) {
    OS << "bool " << Rec->getValueAsString("FunctionName");
    OS << "(const MCInst &MI) {\n";

    OS.indent(PE.getIndentLevel() * 2);
    PE.expandStatement(OS, Rec->getValueAsDef("Body"));
    OS << "\n}\n\n";
  }

  OS << "} // end " << TargetName << "_MC namespace\n";
  OS << "} // end llvm namespace\n\n";

  OS << "#endif // GET_GENISTRINFO_MC_HELPERS\n";
}

void PanopticonInstrEmitter::emitTIIHelperMethods(raw_ostream &OS,
                                            StringRef TargetName,
                                            bool ExpandDefinition) {
  RecVec TIIPredicates = Records.getAllDerivedDefinitions("TIIPredicate");
  if (TIIPredicates.empty())
    return;

  PredicateExpander PE(TargetName);
  PE.setExpandForMC(false);

  for (const Record *Rec : TIIPredicates) {
    OS << (ExpandDefinition ? "" : "static ") << "bool ";
    if (ExpandDefinition)
      OS << TargetName << "InstrInfo::";
    OS << Rec->getValueAsString("FunctionName");
    OS << "(const MachineInstr &MI)";
    if (!ExpandDefinition) {
      OS << ";\n";
      continue;
    }

    OS << " {\n";
    OS.indent(PE.getIndentLevel() * 2);
    PE.expandStatement(OS, Rec->getValueAsDef("Body"));
    OS << "\n}\n\n";
  }
}

//===----------------------------------------------------------------------===//
// Main Output.
//===----------------------------------------------------------------------===//

/// hasNullFragReference - Return true if the DAG has any reference to the
/// null_frag operator.
static bool hasNullFragReference(DagInit *DI) {
  DefInit *OpDef = dyn_cast<DefInit>(DI->getOperator());
  if (!OpDef) return false;
  Record *Operator = OpDef->getDef();

  // If this is the null fragment, return true.
  if (Operator->getName() == "null_frag") return true;
  // If any of the arguments reference the null fragment, return true.
  for (unsigned i = 0, e = DI->getNumArgs(); i != e; ++i) {
    DagInit *Arg = dyn_cast<DagInit>(DI->getArg(i));
    if (Arg && hasNullFragReference(Arg))
      return true;
  }

  return false;
}

/// hasNullFragReference - Return true if any DAG in the list references
/// the null_frag operator.
static bool hasNullFragReference(ListInit *LI) {
  for (Init *I : LI->getValues()) {
    DagInit *DI = dyn_cast<DagInit>(I);
    assert(DI && "non-dag in an instruction Pattern list?!");
    if (hasNullFragReference(DI))
      return true;
  }
  return false;
}

struct arg {
  unsigned bits;
  unsigned pointed_bits;

  arg() : bits(0), pointed_bits(0) {}
  arg(unsigned b) : bits(b), pointed_bits(0) {}
  arg(unsigned b, unsigned p) : bits(b), pointed_bits(p) {}
  ~arg() {}
};

std::string nextTemp(unsigned bits, std::unordered_map<std::string, arg> &args, unsigned &next_temp) {
  std::stringstream ss;

  ss << "t" << next_temp++;
  args[ss.str()] = bits;

  return ss.str();
}

std::string processPattern(raw_ostream &OS, DagInit *dag, std::unordered_map<std::string, arg> &args, unsigned &next_temp, bool &fallthru);

std::string processRecord(raw_ostream &OS, Init *rec, std::string const& name, std::unordered_map<std::string, arg> &args, unsigned &next_temp, bool &fallthru) {
  if (!rec) {
    llvm_unreachable("processRecord: rec is NULL");
  }

  auto def = dyn_cast<DefInit>(rec);
  if (def) {
    if (name == "") {
      return def->getAsString();
    } else {
      return name;
    }
  }

  auto dag = dyn_cast<DagInit>(rec);
  if (dag) {
    return processPattern(OS, dag, args, next_temp, fallthru);
  }

  auto val = dyn_cast<IntInit>(rec);
  if (val) {
    std::stringstream ss;

    if (val->getValue() < 0) {
      ss << "[" << 0xffffffffffffffff - val->getValue() << "]";
    } else {
      ss << "[" << val->getValue() << "]";
    }

    return ss.str();
  }

  llvm_unreachable("processRecord fell-thru");
}

std::string processPattern(raw_ostream &OS, DagInit *dag, std::unordered_map<std::string, arg> &args, unsigned &next_temp, bool &fallthru) {
  auto op = dyn_cast<DefInit>(dag->getOperator());
  auto opnam = op->getAsString();

  if (opnam == "set") {
    auto dst = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto src = processRecord(OS, dag->getArg(dag->getNumArgs() - 1), dag->getArgNameStr(dag->getNumArgs() - 1).str(), args, next_temp, fallthru);

    if (dst != "EFLAGS" && args[dst].bits > 0 && args[src].bits > 0) {
      OS << "\til.extend(rreil!{\n";
      OS << "\t\tmov " << dst << ":" << args[dst].bits << ", " << src << ":" << args[src].bits << "\n";
      OS << "\t}?);\n";
    }

    return dst;
  } else if (opnam == "store") {
    auto src = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto addr = processRecord(OS, dag->getArg(1), dag->getArgNameStr(1).str(), args, next_temp, fallthru);

    OS << "\til.extend(rreil!{\n";
    OS << "\t\tstore/RAM/le/" << args[addr].pointed_bits << " " << addr << ":" << args[addr].bits << ", " << src << ":" << args[src].bits << "\n";
    OS << "\t}?);\n";

    return src;
  } else if (opnam == "i8" || opnam == "i16" || opnam == "i32" || opnam == "i64" || opnam == "f64" || opnam == "v2f64" || opnam == "v4f32") {
    auto t = nextTemp(1, args, next_temp);
    auto imm = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);

    OS << "\til.extend(rreil!{\n";
    OS << "\t\tmov " << t << ":" << args[t].bits << ", " << imm << ":" << args[t].bits << "\n";
    OS << "\t}?);\n";

    return t;
  } else if (opnam == "load" || opnam == "loadi8" || opnam == "loadi16" || opnam == "loadi32" || opnam == "loadi64" || opnam == "loadf64" || opnam == "memop" || opnam == "memopv2f64" || opnam == "memopv4f32") {
    // load addr
    auto addr = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto t = nextTemp(args[addr].pointed_bits, args, next_temp);

    OS << "\til.extend(rreil!{\n";
    OS << "\t\tload/RAM/le/" << args[addr].pointed_bits << " " << t << ":" << args[t].bits
      << ", " << addr << ":" << args[addr].bits << "\n";
    OS << "\t}?);\n";

    return t;
  } else if (opnam == "implicit") {
    return nextTemp(0, args, next_temp);
  } else if (opnam == "X86call") {
    return nextTemp(0, args, next_temp);
  } else if (opnam == "ret") {
    return nextTemp(0, args, next_temp);
  } else if (opnam == "reti") {
    return nextTemp(0, args, next_temp);
  } else if (opnam == "push") {
    return nextTemp(0, args, next_temp);
  } else if (opnam == "pop") {
    return nextTemp(0, args, next_temp);
  } else if (opnam == "X86testpat") {
    return nextTemp(0, args, next_temp);
  } else if (opnam == "X86cmp") {
auto dst = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto src = processRecord(OS, dag->getArg(1), dag->getArgNameStr(1).str(), args, next_temp, fallthru);
    auto b = args[dst].bits;
    auto res = nextTemp(b, args, next_temp);

    OS << "// X86cmp_flag\n";
    OS << "\til.extend(rreil!{\n";
    OS << "\tsub res:" << b << ", " << dst << ":" << b ", " << src << ":" << b "\n";
    OS << "\tcmplts SF:1, res:" << b << ", [0]:" << b << "\n";
    OS << "\tcmpeq ZF:1, res:" << b << ", [0]:" << b << "\n";

    OS << "\t\tmov " << res << ":" << b << ", " << dst << ":" << b << "\n";
    OS << "\t}?);\n\n";

    OS << "\tset_sub_carry_flag(rreil_var!(" << res << ":" << b << "), rreil_val!(" << dst << ":" << b << "), rreil_val!(" << src << ":" << args[src].bits << "), " << b <<", il)?;\n";
    OS << "\tset_sub_adj_flag(rreil_var!(" << res << ":" << b << "), rreil_val!(" << dst << ":" << b << "), rreil_val!(" << src << ":" << args[src].bits << "), " << b <<", il)?;\n";
    OS << "\tset_sub_overflow_flag(rreil_var!(" << res << ":" << b << "), rreil_val!(" << dst << ":" << b << "), rreil_val!(" << src << ":" << args[src].bits << "), " << b <<", il)?;\n";
    OS << "\tset_parity_flag(rreil_var!(" << res << ":" << b << "), il)?;\n";


    return nextTemp(0, args, next_temp);
  } else if (opnam == "X86cmov") {
    return nextTemp(0, args, next_temp);
  } else if (opnam == "X86smul_flag") {
    return nextTemp(0, args, next_temp);
  } else if (opnam == "X86sub_flag" || opnam == "sub") {
auto dst = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto src = processRecord(OS, dag->getArg(1), dag->getArgNameStr(1).str(), args, next_temp, fallthru);
    auto b = args[dst].bits;
    auto res = nextTemp(b, args, next_temp);

    OS << "// X86sub_flag\n";
    OS << "\til.extend(rreil!{\n";
    OS << "\t\tsub res:" << b << ", " << dst << ":" << b ", " << src << ":" << b "\n";
    OS << "\t\tcmplts SF:1, res:" << b << ", [0]:" << b << "\n";
    OS << "\t\tcmpeq ZF:1, res:" << b << ", [0]:" << b << "\n";

    OS << "\t\tmov " << res << ":" << b << ", " << dst << ":" << b << "\n";
    OS << "\t}?);\n\n";

    OS << "\tset_sub_carry_flag(rreil_var!(" << res << ":" << b << "), rreil_val!(" << dst << ":" << b << "), rreil_val!(" << src << ":" << args[src].bits << "), " << b <<", il)?;\n";
    OS << "\tset_sub_adj_flag(rreil_var!(" << res << ":" << b << "), rreil_val!(" << dst << ":" << b << "), rreil_val!(" << src << ":" << args[src].bits << "), " << b <<", il)?;\n";
    OS << "\tset_sub_overflow_flag(rreil_var!(" << res << ":" << b << "), rreil_val!(" << dst << ":" << b << "), rreil_val!(" << src << ":" << args[src].bits << "), " << b <<", il)?;\n";
    OS << "\tset_parity_flag(rreil_var!(" << res << ":" << b << "), il)?;\n";


    return nextTemp(0, args, next_temp);
  } else if (opnam == "X86sbb_flag") {
    // X86sbb_flag dst, src, EFLAGS
    auto dst = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto src = processRecord(OS, dag->getArg(1), dag->getArgNameStr(1).str(), args, next_temp, fallthru);
    auto b = args[dst].bits;
    auto res = nextTemp(b, args, next_temp);

    OS << "// X86sbb_flag\n";
    OS << "\til.extend(rreil!{\n";
    OS << "\t\tsub " << res << ":" << b << ", " << dst << ", " << src << "\n";
    OS << "\t\tzext/" << b << " cf:" << b << ", CF:1\n";
    OS << "\t\tsub " << res << ":" << b << ", " << res << ":" << b << ", cf:" << b << "\n";
    OS << "\t\tcmplts SF:1, " << res << ":" << b << ", [0]:" << b << "\n";
    OS << "\t\tcmpeq ZF:1, " << res << ":" << b << ", [0]:" << b << "\n";

    OS << "\t\tmov a:4, " << dst << ":" << b << "\n";
    OS << "\t\tcmpeq af1:1, " << res << ":4, a:4\n";
    OS << "\t\tcmpltu af2:1, a:4, " << res << ":4\n";
    OS << "\t\tand af1:1, af1:1, CF:1\n";
    OS << "\t\tor AF:1, af1:1, af2:1\n";

    OS << "\t\tmov " << res << ":" << b << ", " << dst << ":" << b << "\n";
    OS << "\t}?);\n\n";

    OS << "\tset_sub_carry_flag(rreil_var!(" << res << ":" << b << "), rreil_val!(" << dst << ":" << b << "), rreil_val!(" << src << ":" << args[src].bits << "), " << b <<", il)?;\n";
    OS << "\tset_sub_overflow_flag(rreil_var!(" << res << ":" << b << "), rreil_val!(" << dst << ":" << b << "), rreil_val!(" << src << ":" << args[src].bits << "), " << b <<", il)?;\n";
    OS << "\tset_parity_flag(rreil_var!(" << res << ":" << b << "), il)?;\n";

    return dst;
  } else if (opnam == "X86brcond") {
    auto tgt = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto b = args[tgt].bits;
    auto cc = dag->getArg(1)->getAsString();

    if (cc == "X86_COND_A") {
      OS << "\til.extend(rreil!{\n";
      OS << "\t\tcmpeq ZFnull:1, ZF:1, [0]:1\n";
      OS << "\t\tcmpeq CFnull:1, CF:1, [0]:1\n";
      OS << "\t\tand ZFandCFnull:1, CFnull:1, ZFnull:1\n";
      OS << "\t}?);\n";

      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(ZFandCFnull:1), expected: true }))\n}\n\n";
    } else if (cc == "X86_COND_AE") {
      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(CF:1), expected: false }))\n}\n\n";
    } else if (cc == "X86_COND_B") {
      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(CF:1), expected: true }))\n}\n\n";
    } else if (cc == "X86_COND_BE") {
      OS << "\til.extend(rreil!{\n";
      OS << "\t\tand ZForCF:1, CF:1, ZF:1\n";
      OS << "\t}?);\n";

      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(CForZF:1), expected: true }))\n}\n\n";
    } else if (cc == "X86_COND_E") {
      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(ZF:1), expected: true }))\n}\n\n";
    } else if (cc == "X86_COND_G") {
      OS << "\til.extend(rreil!{\n";
      OS << "\t\tcmpeq SFisOF:1, SF:1, OF:1\n";
      OS << "\t\tcmpeq ZFnull:1, ZF:1, [0]:1\n";
      OS << "\t\tand ZFandGreater:1, ZFnull:1, SFisOF:1\n";
      OS << "\t}?);\n";

      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(ZFandGreater:1), expected: true }))\n}\n\n";
    } else if (cc == "X86_COND_GE") {
      OS << "\til.extend(rreil!{\n";
      OS << "\t\tcmpeq SFisOF:1, SF:1, OF:1\n";
      OS << "\t}?);\n";

      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(SFisOF:1), expected: true }))\n}\n\n";
    } else if (cc == "X86_COND_L") {
      OS << "\til.extend(rreil!{\n";
      OS << "\t\txor SFxorOF:1, SF:1, OF:1\n";
      OS << "\t}?);\n";

      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(SFxorOF:1), expected: true }))\n}\n\n";
    } else if (cc == "X86_COND_LE") {
      OS << "\til.extend(rreil!{\n";
      OS << "\t\txor SFxorOF:1, SF:1, OF:1\n";
      OS << "\t\tor ZFofLess:1, ZF:1, SFxorOF:1\n";
      OS << "\t}?);\n";

      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(ZFofLess:1), expected: true }))\n}\n\n";
    } else if (cc == "X86_COND_NO") {
      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(OF:1), expected: false }))\n}\n\n";
    } else if (cc == "X86_COND_NE") {
      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(ZF:1), expected: false }))\n}\n\n";
    } else if (cc == "X86_COND_NP") {
      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(PF:1), expected: false }))\n}\n\n";
    } else if (cc == "X86_COND_NS") {
      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(SF:1), expected: false }))\n}\n\n";
    } else if (cc == "X86_COND_O") {
      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(OF:1), expected: true }))\n}\n\n";
    } else if (cc == "X86_COND_P") {
      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(PF:1), expected: true }))\n}\n\n";
    } else if (cc == "X86_COND_S") {
      OS << "\tOk(JumpSpec::Branch(rreil_val!(" << tgt << ":" << b
        << "), Guard::Predicate{ flag: rreil_var!(SF:1), expected: true }))\n}\n\n";
    } else {
      llvm_unreachable("unknown CC");
    }

    fallthru = false;

    return tgt;
  } else if (opnam == "brind") {
    auto tgt = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto b = args[tgt].bits;

    OS << "\tOk(JumpSpec::Jump(rreil_val!(" << tgt << ":" << b << ")))\n}\n\n";
    fallthru = false;

    return tgt;
  } else if (opnam == "not") {
    auto val = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto b = args[val].bits;

    OS << "\til.extend(rreil!{\n";
    OS << "\t\txor " << val << ":" << b << ", " << val << ":" << b << ", [0xffffffffffffffff]:" << b << "\n";
    OS << "\t}?);\n\n";

    return val;
  } else if (opnam == "X86adc_flag" || opnam == "adc") {
    // X86adc_flag dst, src, EFLAGS
    auto dst = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto src = processRecord(OS, dag->getArg(1), dag->getArgNameStr(1).str(), args, next_temp, fallthru);
    auto b = args[dst].bits;
    auto res = nextTemp(b, args, next_temp);

    OS << "// X86adc_flag\n";
    OS << "\til.extend(rreil!{\n";
    OS << "\t\tadd " << res << ":" << b << ", " << dst << ":" << b << ", " << src << ":" << args[src].bits << "\n";
    OS << "\t\tzext/" << b << " cf:" << b << ", CF:1\n";
    OS << "\t\tadd " << res << ":" << b << ", " << dst << ":" << b << ", cf:" << b << "\n";

    OS << "\t\tcmplts SF:1, " << res << ":" << b << ", [0]:" << b << "\n";

    OS << "\t\tcmpeq ZF:1, " << res << ":" << b << ", [0]:" << b << "\n\n";

    OS << "\t\tmov a:4, " << src << ":" << args[src].bits << "\n";
    OS << "\t\tcmpeq af1:1, " << dst << ":4, a:4\n";
    OS << "\t\tcmpltu af2:1, " << dst << ":4, a:4\n";
    OS << "\t\tand af1:1, af1:1, CF:1\n";
    OS << "\t\tor AF:1, af1:1, af2:1\n";

    OS << "\t\tmov " << res << ":" << b << ", " << dst << ":" << b << "\n";
    OS << "\t}?);\n\n";

    OS << "\tset_carry_flag(rreil_var!(" << res << ":" << b << "), rreil_val!(" << dst << ":" << b << "), rreil_val!(" << src << ":" << args[src].bits << "), " << b <<", il)?;\n";
    OS << "\tset_overflow_flag(rreil_var!(" << res << ":" << b << "), rreil_val!(" << dst << ":" << b << "), rreil_val!(" << src << ":" << args[src].bits << "), " << b <<", il)?;\n";
    OS << "\tset_parity_flag(rreil_var!(" << res << ":" << b << "), il)?;\n";

    return dst;
  } else if (opnam == "fadd" || opnam == "X86Addsub") {
    assert(dag->getNumArgs() == 2);
    auto dst = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto src = processRecord(OS, dag->getArg(1), dag->getArgNameStr(1).str(), args, next_temp, fallthru);
    auto b = args[dst].bits;

    OS << "\til.extend(rreil!{\n";
    OS << "\t\tmov " << dst << ":" << b << ", ?\n";
    OS << "\t}?);\n";

    return dst;
  } else if (opnam == "X86and_flag" || opnam == "and") {
    // X86and_flag dst, src, EFLAGS
    auto dst = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto src = processRecord(OS, dag->getArg(1), dag->getArgNameStr(1).str(), args, next_temp, fallthru);
    auto b = args[dst].bits;

    OS << "\til.extend(rreil!{\n";
    OS << "\t\tand " << dst << ":" << b << ", " << dst << ":" << b << ", " << src << ":" << b << "\n";
    OS << "\t\tcmplts SF:1, " << dst << ":" << b << ", [0]:" << b << "\n";
    OS << "\t\tcmpeq ZF:1, " << dst << ":" << b << ", [0]:" << b << "\n";
    OS << "\t\tmov CF:1, [0]:1\n";
    OS << "\t\tmov OF:1, [0]:1\n";
    OS << "\t\tmov AF:1, ?\n";
    OS << "\t}?);\n\n";

    OS << "\tset_parity_flag(rreil_var!(" << dst << ":" << b << "), il)?;\n";

    return dst;
  } else if (opnam == "X86or_flag" || opnam == "or") {
    // X86or_flag dst, src, EFLAGS
    auto dst = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto src = processRecord(OS, dag->getArg(1), dag->getArgNameStr(1).str(), args, next_temp, fallthru);
    auto b = args[dst].bits;

    OS << "\til.extend(rreil!{\n";
    OS << "\t\tor " << dst << ":" << b << ", " << dst << ":" << b << ", " << src << ":" << b << "\n";
    OS << "\t\tcmplts SF:1, " << dst << ":" << b << ", [0]:" << b << "\n";
    OS << "\t\tcmpeq ZF:1, " << dst << ":" << b << ", [0]:" << b << "\n";
    OS << "\t\tmov CF:1, [0]:1\n";
    OS << "\t\tmov OF:1, [0]:1\n";
    OS << "\t\tmov AF:1, ?\n";
    OS << "\t}?);\n\n";

    OS << "\tset_parity_flag(rreil_var!(" << dst << ":" << b << "), il)?;\n";

    return dst;
  } else if (opnam == "X86bt") {
    auto src1 = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto src2 = processRecord(OS, dag->getArg(1), dag->getArgNameStr(1).str(), args, next_temp, fallthru);
    auto b = args[src1].bits;
    auto t = nextTemp(b, args, next_temp);

    OS << "\til.extend(rreil!{\n";
    OS << "\t\tmov " << t << ":" << args[t].bits << ", ?\n";
    OS << "\t\tmov CF:1, ?\n";
    OS << "\t}?);\n";

    return t;
  } else if (opnam == "X86xor_flag" || opnam == "xor") {
    // X86xor_flag dst, src, EFLAGS
    auto dst = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto src = processRecord(OS, dag->getArg(1), dag->getArgNameStr(1).str(), args, next_temp, fallthru);
    auto b = args[dst].bits;

    OS << "\til.extend(rreil!{\n";
    OS << "\t\txor " << dst << ":" << b << ", " << dst << ":" << b << ", " << src << ":" << b << "\n";
    OS << "\t\tcmplts SF:1, " << dst << ":" << b << ", [0]:" << b << "\n";
    OS << "\t\tcmpeq ZF:1, " << dst << ":" << b << ", [0]:" << b << "\n";
    OS << "\t\tmov CF:1, [0]:1\n";
    OS << "\t\tmov OF:1, [0]:1\n";
    OS << "\t\tmov AF:1, ?\n";
    OS << "\t}?);\n\n";

    OS << "\tset_parity_flag(rreil_var!(" << dst << ":" << b << "), il)?;\n";

    return dst;
  } else if (opnam == "X86add_flag" || opnam == "add") {
    // X86add_flag dst, src, EFLAGS
    auto dst = processRecord(OS, dag->getArg(0), dag->getArgNameStr(0).str(), args, next_temp, fallthru);
    auto src = processRecord(OS, dag->getArg(1), dag->getArgNameStr(1).str(), args, next_temp, fallthru);
    auto b = args[dst].bits;
    auto res = nextTemp(b, args, next_temp);

    OS << "\t// X86add_flag\n";
    OS << "\til.extend(rreil!{\n";
    OS << "\t\tadd " << res << ":" << b << ", " << dst << ":" << b << ", " << src << ":" << b << "\n";

    OS << "\t\tcmplts SF:1, " << res << ":" << b << ", [0]:" << b << "\n";

    OS << "\t\tcmpeq ZF:1, " << res << ":" << b << ", [0]:" << b << "\n\n";

    OS << "\t\tmov a:4, " << src << ":" << b << "\n";
    OS << "\t\tcmpeq af1:1, " << res << ":4, a:4\n";
    OS << "\t\tcmpltu af2:1, " << res << ":4, a:4\n";
    OS << "\t\tor AF:1, af1:1, af2:1\n";

    OS << "\t\tmov " << res << ":" << b << ", " << dst << ":" << b << "\n";
    OS << "\t}?);\n\n";

    OS << "\tset_carry_flag(rreil_var!(" << res << ":" << b << "), rreil_val!(" << dst << ":" << b << "), rreil_val!(" << src << ":" << b << "), " << b <<", il)?;\n";
    OS << "\tset_overflow_flag(rreil_var!(" << res << ":" << b << "), rreil_val!(" << dst << ":" << b << "), rreil_val!(" << src << ":" << b << "), " << b <<", il)?;\n";
    OS << "\tset_parity_flag(rreil_var!(" << res << ":" << b << "), il)?;\n";

    return dst;
  } else {
    std::vector<std::string> ops;

    for (unsigned i = 0; i < dag->getNumArgs(); i += 1) {
      ops.push_back(processRecord(OS, dag->getArg(i), dag->getArgNameStr(i).str(), args, next_temp, fallthru));
    }

    unsigned b = 32;

    if (args.count("dst")) {
      b = args["dst"].bits;
    } else if (ops.size() > 0) {
      b = args[ops[0]].bits;
    }

    auto t = nextTemp(b, args, next_temp);

    OS << "\t// undefined: " << opnam << "\n";
    OS << "\til.extend(rreil!{\n";
    OS << "\t\tmov " << t << ":" << args[t].bits << ", ?\n";
    OS << "\t}?);\n";

    return t;
  }
}

// run - Emit the main instruction description records for the target...
void PanopticonInstrEmitter::run(raw_ostream &OS) {
  //emitInstructionEnum(OS);
  emitSemantics(OS);
}

void PanopticonInstrEmitter::emitInstructionEnum(raw_ostream &OS) {
  CodeGenTarget &Target = CDP.getTargetInfo();
  unsigned Num = 0;
  OS << "#[derive(PartialEq, Debug, Clone, Copy)]\n";
  OS << "pub enum Opcode {\n";
  for (const CodeGenInstruction *Inst : Target.getInstructionsByEnumValue())
    OS << "    " << Inst->TheDef->getName() << "\t= " << Num++ << ",\n";
  OS << "}\n\n";
}

void PanopticonInstrEmitter::emitSemantics(raw_ostream &OS) {
  CodeGenTarget &Target = CDP.getTargetInfo();
  ArrayRef<const CodeGenInstruction*> NumberedInstructions =
    Target.getInstructionsByEnumValue();
  X86Disassembler::DisassemblerTables Tables;

  OS << "#![allow(non_snake_case)]\n\n";
  OS << "use crate::common::{JumpSpec, set_carry_flag, set_parity_flag, set_overflow_flag};\n";
  OS << "use crate::decoder::{Instruction, decode_operand};\n";
  OS << "\n";
  OS << "use p8n_types::{Value, Statement, Guard, Result};\n";
  OS << "use p8n_rreil_macro::{rreil_var, rreil_val, rreil};\n";
  OS << "\n";

  for (auto cgi: NumberedInstructions) {
    auto def = cgi->TheDef;

    if(def->getValueAsBit("isCodeGenOnly") || def->getValueAsBit("isPseudo") || cgi->AsmString == "")
      continue;

    auto insn = CDP.getInstruction(def);
    auto src = insn.getSrcPattern();
    ListInit *LI = nullptr;

    if (isa<ListInit>(def->getValueInit("Pattern")))
      LI = def->getValueAsListInit("Pattern");

        OS << "// " << cgi->AsmString << "\n";
    OS << "pub fn sem_"
      << cgi->TheDef->getName()
      << "(insn: &mut Instruction, il: &mut Vec<Statement>) -> Result<JumpSpec> {\n";

    std::unordered_map<std::string, arg> args;
    std::vector<std::string> ops;

    for (unsigned i = 0; i < cgi->Operands.size(); i += 1) { 
      auto opi = cgi->Operands[i];

      ops.push_back(opi.Name);

      if (opi.Rec->isSubClassOf("RegisterClass") || opi.Rec->isSubClassOf("RegisterOperand")) {
        std::string cls;

        if (opi.Rec->isSubClassOf("RegisterOperand")) {
          cls = opi.Rec->getValueAsDef("RegClass")->getName();
        } else {
          cls = opi.Rec->getName();
        }

        // XXX: assumes sizeof(ptr) == 4

        if (cls == "VK1") {
          args[opi.Name] = 1;
        } else if (cls == "GR8" || cls == "VK1WM" || cls == "VK2" || cls == "VK4" || cls == "VK8" || cls == "VK2WM" || cls == "VK4WM" || cls == "VK8WM") {
          args[opi.Name] = 8;
        } else if (cls == "GR16" || cls == "VK16" || cls == "VK16WM" || cls == "SEGMENT_REG") {
          args[opi.Name] = 16;
        } else if (cls == "GR32" || cls == "VK32" || cls == "VK32WM" || cls == "FR32X" || cls == "FR32" || cls == "RFP32" || cls == "CONTROL_REG" || cls == "DEBUG_REG") {
          args[opi.Name] = 32;
        } else if (cls == "GR64" || cls == "VK64" || cls == "VK64WM" || cls == "RFP64" || cls == "VR64" || cls == "FR64" || cls == "FR64X") {
          args[opi.Name] = 64;
        } else if (cls == "RFP80" || cls == "RST") {
          args[opi.Name] = 80;
        } else if (cls == "VR128" || cls == "VR128X" || cls == "BNDR") {
          args[opi.Name] = 128;
        } else if (cls == "VR256" || cls == "VR256X") {
          args[opi.Name] = 256;
        } else if (cls == "VR512") {
          args[opi.Name] = 512;
        } else {
          OS << "-- RegClass: " << cls << "\n";
          llvm_unreachable("unknown register class");
        }
      } else {
        auto opty = opi.Rec->getValueAsString("OperandType");

        if (opty == "OPERAND_MEMORY") {
          auto cls = opi.Rec->getName();

          // XXX: assumes sizeof(ptr) == 4

          if (cls == "i8mem" || cls == "opaquemem" || cls == "anymem" || cls ==  "dstidx8" || cls == "srcidx8" || cls == "offset32_8") {
            args[opi.Name] = arg(32, 8);
          } else if (cls == "i16mem" || cls ==  "dstidx16" || cls == "srcidx16" || cls == "offset32_16") {
            args[opi.Name] = arg(32, 16);
          } else if (cls == "i32mem" || cls == "f32mem" || cls ==  "dstidx32" || cls == "srcidx32" || cls == "offset32_32") {
            args[opi.Name] = arg(32, 32);
          } else if (cls == "i64mem" || cls == "f64mem" || cls == "vx64mem" || cls == "vx64xmem" || cls ==  "dstidx64" || cls == "srcidx64" || cls == "offset32_64") {
            args[opi.Name] = arg(32, 64);
          } else if (cls == "f80mem") {
            args[opi.Name] = arg(32, 80);
          } else if (cls == "i128mem" || cls == "f128mem" || cls == "sdmem" || cls == "ssmem" || cls == "vx128xmem" || cls == "vx128mem"  || cls == "vy128mem" || cls == "vy128xmem") {
            args[opi.Name] = arg(32, 128);
          } else if (cls == "i256mem" || cls == "f256mem" || cls == "vx256mem" || cls == "vx256xmem" || cls == "vy256mem" || cls == "vy256xmem" || cls == "vz256mem") {
            args[opi.Name] = arg(32, 256);
          } else if (cls == "i512mem" || cls == "f512mem" || cls == "v512mem" || cls == "vy512xmem" || cls == "vz512mem") {
            args[opi.Name] = arg(32, 512);
          } else if (cls == "offset16_8") {
            args[opi.Name] = arg(16, 8);
          } else if (cls == "offset16_16") {
            args[opi.Name] = arg(16, 16);
          } else if (cls == "offset16_32") {
            args[opi.Name] = arg(16, 32);
          } else if (cls == "offset16_64") {
            args[opi.Name] = arg(16, 64);
          } else if (cls == "offset64_8") {
            args[opi.Name] = arg(64, 8);
          } else if (cls == "offset64_16") {
            args[opi.Name] = arg(64, 16);
          } else if (cls == "offset64_32") {
            args[opi.Name] = arg(64, 32);
          } else if (cls == "offset64_64") {
            args[opi.Name] = arg(64, 64);
           } else {
             OS << "-- Mem: " << cls << "\n";
             llvm_unreachable("unknown memory ref");
          }
        } else if (opty == "OPERAND_IMMEDIATE") {
          //OS << "-- Imm\n";
          auto ty = opi.Rec->getValueAsDef("Type")->getName();
          auto cls = opi.Rec->getValueAsDef("ParserMatchClass")->getName();

          if (cls == "ImmAsmOperand") {
            ;
          } else if (cls == "ImmSExtAsmOperandClass") {
            ;
          } else if (cls == "ImmSExti16i8AsmOperand") {
            ;
          } else if (cls == "ImmSExti32i8AsmOperand") {
            ;
          } else if (cls == "ImmSExti64i8AsmOperand") {
            ;
          } else if (cls == "ImmSExti64i32AsmOperand") {
            ;
          } else if (cls == "ImmUnsignedi8AsmOperand") {
            ;
          } else if (cls == "AVX512RCOperand") {
            ;
          } else {
            llvm_unreachable("unknown imm class");
          }

          if (ty == "i8") {
            args[opi.Name] = 8;
          } else if (ty == "i16") {
            args[opi.Name] = 16;
          } else if (ty == "i32") {
            args[opi.Name] = 32;
          } else if (ty == "i64") {
            args[opi.Name] = 64;
          } else {
            llvm_unreachable("unknown imm value type");
          }
        } else if (opty == "OPERAND_PCREL") {
          //OS << "-- PC-Rel\n";
          //OS << *opi.Rec << "\n";
          args[opi.Name] = 32;
        } else {
          //OS << "-- Unk: " << opi.Rec->getValueAsString("OperandType") << "\n";
          //for (auto p: opi.Rec->getSuperClasses()) {
          //  OS << "--- " << p.first->getName() << "\n";
          //}
          //OS << *opi.Rec << "\n";
        }
      }

      assert (opi.Name != "");
    }


    unsigned idx = 0;

    OS << "\til.extend(rreil!{\n";
    for (auto o: ops) {
      unsigned bits = args[o].bits;
      
      if (bits == 0 && cgi->TheDef->getName() == "LEA64r") {
        bits = 64;
      } else if (bits == 0 && (cgi->TheDef->getName() == "LEA64_32r" || cgi->TheDef->getName() == "LEA32r")) {
        bits = 32;
      }

      OS << "\t\tmov " << o << ":" << bits << ", (decode_operand(insn, insn.operands.unwrap()[" << idx << "].clone(), " << bits << ")?.1)\n";
      idx += 1;
    }
    OS << "\t}?);\n";

    bool fallthru = true;
    unsigned next_tmp = 0;

    if (!LI || LI->empty() || hasNullFragReference(LI)) {
      if (cgi->TheDef->getName().substr(0, 6) == "PUSH16") {
        OS << "\t// XXX: push word\n";
      } else if (cgi->TheDef->getName().substr(0, 6) == "PUSH32") {
        OS << "\t// XXX: push dword\n";
      } else if (cgi->TheDef->getName().substr(0, 6) == "PUSH64") {
        OS << "\t// XXX: push qword\n";
      } else if (cgi->TheDef->getName().substr(0, 3) == "LEA") {
        OS << "\t// XXX: lea\n";
      } else if (cgi->TheDef->getName().substr(0, 4) == "CALL") {
        OS << "\t// XXX: call\n";
      } else if (cgi->TheDef->getName().substr(0, 7) == "FARCALL") {
        OS << "\t// XXX: farcall\n";
      } else if (cgi->TheDef->getName().substr(0, 3) == "RET") {
        OS << "\t// XXX: ret\n";
      } else {
        OS << "\t// no sematics defined\n";
      }
    } else {
      for (auto initdag: *LI) {
        auto dag = dyn_cast<DagInit>(initdag);
        processPattern(OS, dag, args, next_tmp, fallthru);
      }
    }

    if (fallthru) {
      OS << "\tOk(JumpSpec::FallThru)\n}\n\n";
    }
  }

  OS << "pub const SEMANTICS: &'static [Option<fn(&mut Instruction, &mut Vec<Statement>) -> Result<JumpSpec>>] = &[\n";
  for (auto cgi: NumberedInstructions) {
    auto def = cgi->TheDef;

    if(def->getValueAsBit("isCodeGenOnly") || def->getValueAsBit("isPseudo") || cgi->AsmString == "") {
      OS << "\tNone, // " << def->getName() << "\n";
    } else {
      OS << "\tSome(sem_" << def->getName() << "),\n";
    }
  }
  OS << "];\n";
}

namespace llvm {

void EmitPanopticonInstrs(RecordKeeper &RK, raw_ostream &OS) {
  PanopticonInstrEmitter(RK).run(OS);
}

} // end llvm namespace
