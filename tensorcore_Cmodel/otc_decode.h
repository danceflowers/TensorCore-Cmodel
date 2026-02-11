#pragma once
#include "otc_types.h"

// ============================================================================
// OpenTensorCore ISA Decode Framework
// ============================================================================
// Flexible instruction decode layer. Opcode/funct3/unit_id values are NOT
// hardcoded — they are loaded from a configurable table so the ISA can be
// adjusted after performance simulation without touching decode logic.
// ============================================================================

// --- Instruction type enum (semantic, not encoding) -------------------------
enum class OTC_OpType : uint8_t {
    NOP = 0,

    // Tensor-core compute
    TCU_WMMA,       // Matrix multiply-accumulate  D = A×B + C
    TCU_SP,         // Declare single-precision float format
    TCU_INT,        // Declare integer format
    TCU_DP,         // Declare double-precision float format
    TCU_SFU,        // Special function unit operation

    // Memory
    TCU_LOAD,       // Load matrix tile from memory → TCU registers
    TCU_STORE,      // Store TCU result → memory
    LOAD,           // General-purpose register load
    STORE,          // General-purpose register store

    // Synchronisation
    TCU_BARRIER,    // Memory ordering / fence

    OP_COUNT        // sentinel — number of op types
};

inline const char* optype_name(OTC_OpType op) {
    switch (op) {
        case OTC_OpType::NOP:          return "NOP";
        case OTC_OpType::TCU_WMMA:     return "TCU_WMMA";
        case OTC_OpType::TCU_SP:       return "TCU_SP";
        case OTC_OpType::TCU_INT:      return "TCU_INT";
        case OTC_OpType::TCU_DP:       return "TCU_DP";
        case OTC_OpType::TCU_SFU:      return "TCU_SFU";
        case OTC_OpType::TCU_LOAD:     return "TCU_LOAD";
        case OTC_OpType::TCU_STORE:    return "TCU_STORE";
        case OTC_OpType::LOAD:         return "LOAD";
        case OTC_OpType::STORE:        return "STORE";
        case OTC_OpType::TCU_BARRIER:  return "TCU_BARRIER";
        default:                       return "UNKNOWN";
    }
}

// --- Execution unit target --------------------------------------------------
enum class ExecUnit : uint8_t {
    NONE = 0,
    TCU  = 1,       // Tensor Core Unit
    LSU  = 2,       // Load/Store Unit (general purpose)
    SYNC = 3,       // Synchronisation / barrier unit
    SFU  = 4,       // Special Function Unit
};

// --- Decoded instruction packet ---------------------------------------------
// This is the output of the decoder — everything downstream needs to execute.
struct DecodedInst {
    // Identification
    OTC_OpType  op      = OTC_OpType::NOP;
    ExecUnit    unit    = ExecUnit::NONE;
    uint32_t    raw     = 0;          // original 32-bit instruction word

    // Register operands (semantic, filled by decoder)
    uint8_t     rd      = 0;          // destination register index
    uint8_t     rs1     = 0;          // source register 1
    uint8_t     rs2     = 0;          // source register 2
    uint8_t     rs3     = 0;          // source register 3 (for FMA-style)

    // Immediate / config fields
    int32_t     imm     = 0;          // sign-extended immediate
    uint8_t     funct3  = 0;          // sub-function code
    uint8_t     funct7  = 0;          // extended function code

    // Matrix-specific operand info (populated for TCU_WMMA / TCU_LOAD / TCU_STORE)
    uint8_t     mat_m   = 0;          // M dimension hint
    uint8_t     mat_k   = 0;          // K dimension hint
    uint8_t     mat_n   = 0;          // N dimension hint
    uint8_t     dtype   = 0;          // data-type selector (maps to TYPE_FP4 etc.)
    uint8_t     dtype_sub = 0;        // sub-type (E5M2 vs E4M3)

    // Control flags
    bool        valid   = false;      // decoder successfully matched
    bool        is_mem  = false;      // touches memory
    bool        is_tcu  = false;      // goes to tensor core pipeline
    bool        is_sync = false;      // barrier / fence

    void clear() { *this = DecodedInst{}; }

    void dump(std::ostream& os) const {
        os << "Inst[" << optype_name(op) << "] raw=0x"
           << std::hex << std::setw(8) << std::setfill('0') << raw << std::dec
           << " rd=" << (int)rd << " rs1=" << (int)rs1 << " rs2=" << (int)rs2
           << " imm=" << imm << " funct3=" << (int)funct3
           << " valid=" << valid << "\n";
    }
};

// --- ISA encoding table entry -----------------------------------------------
// One row per instruction type.  The actual binary values live HERE, not in
// the decode switch-case, so ISA tweaks only touch the table.
struct ISA_Entry {
    OTC_OpType  op;
    uint8_t     opcode;       // 7-bit opcode field
    uint8_t     unit_id;      // 3-bit unit selector
    uint8_t     funct3;       // 3-bit function code
    ExecUnit    target;       // which execution unit handles it

    // Masks: which fields are "don't care" when matching (0 = must match)
    uint8_t     funct3_mask;  // 0x07 = must match all 3 bits, 0 = ignore
};

// --- Decoder class ----------------------------------------------------------
class OTC_Decoder {
public:
    // Initialise with default ISA table.  Call load_isa_table() to override.
    void init();

    // Replace the ISA table (e.g. from a config file after perf-sim tuning).
    void load_isa_table(const std::vector<ISA_Entry>& table);

    // Main decode entry point: 32-bit instruction word → DecodedInst
    DecodedInst decode(uint32_t inst) const;

    // Convenience: decode and route to the appropriate execution-unit queue.
    // Returns the target ExecUnit so the caller can dispatch.
    ExecUnit decode_and_route(uint32_t inst, DecodedInst& out) const;

    // Query helpers
    const std::vector<ISA_Entry>& isa_table() const { return table_; }
    int table_size() const { return (int)table_.size(); }

private:
    std::vector<ISA_Entry> table_;

    // Extract fixed bit-fields from a 32-bit instruction word.
    // These follow a RISC-V-like layout but can be remapped.
    static uint8_t  extract_opcode(uint32_t inst) { return inst & 0x7F; }
    static uint8_t  extract_rd    (uint32_t inst) { return (inst >> 7) & 0x1F; }
    static uint8_t  extract_funct3(uint32_t inst) { return (inst >> 12) & 0x07; }
    static uint8_t  extract_rs1   (uint32_t inst) { return (inst >> 15) & 0x1F; }
    static uint8_t  extract_rs2   (uint32_t inst) { return (inst >> 20) & 0x1F; }
    static uint8_t  extract_funct7(uint32_t inst) { return (inst >> 25) & 0x7F; }
    static uint8_t  extract_unit_id(uint32_t inst);

    // Build sign-extended immediate for different instruction formats
    static int32_t  extract_imm_I(uint32_t inst);
    static int32_t  extract_imm_S(uint32_t inst);

    // Lookup inst in ISA table, return index or -1
    int match(uint32_t inst) const;
};
