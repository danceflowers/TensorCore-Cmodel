#include "otc_decode.h"

// ============================================================================
// Default ISA table — matches the OpenTC binary encoding specification.
// Values are centralised here; the decode logic only uses table lookups.
// After performance simulation, update ONLY this table (or load a new one).
// ============================================================================
static const std::vector<ISA_Entry> DEFAULT_ISA_TABLE = {
    //  op                  opcode  unit_id funct3  target          f3_mask
    { OTC_OpType::TCU_WMMA,    0x21,   0x01,  0x01, ExecUnit::TCU,   0x07 },
    { OTC_OpType::TCU_LOAD,    0x23,   0x01,  0x01, ExecUnit::TCU,   0x07 },
    { OTC_OpType::TCU_STORE,   0x27,   0x01,  0x01, ExecUnit::TCU,   0x07 },
    { OTC_OpType::LOAD,        0x03,   0x02,  0x02, ExecUnit::LSU,   0x07 },
    { OTC_OpType::STORE,       0x23,   0x02,  0x02, ExecUnit::LSU,   0x07 },
    { OTC_OpType::TCU_BARRIER, 0x33,   0x03,  0x01, ExecUnit::SYNC,  0x07 },
    { OTC_OpType::TCU_SP,      0x43,   0x04,  0x01, ExecUnit::TCU,   0x07 },
    { OTC_OpType::TCU_INT,     0x53,   0x05,  0x00, ExecUnit::TCU,   0x07 },
    { OTC_OpType::TCU_DP,      0x63,   0x06,  0x01, ExecUnit::TCU,   0x07 },
    { OTC_OpType::TCU_SFU,     0x73,   0x07,  0x01, ExecUnit::SFU,   0x07 },
};

// ============================================================================
// Bit-field extraction helpers
// ============================================================================

uint8_t OTC_Decoder::extract_unit_id(uint32_t inst) {
    // unit_id sits in bits [9:7] in the current encoding (overlaps rd field).
    // This mapping can change — keeping it in one place.
    return (inst >> 7) & 0x07;
}

int32_t OTC_Decoder::extract_imm_I(uint32_t inst) {
    // I-type immediate: inst[31:20]
    int32_t imm = (int32_t)(inst & 0xFFF00000) >> 20;
    return imm;
}

int32_t OTC_Decoder::extract_imm_S(uint32_t inst) {
    // S-type immediate: inst[31:25] | inst[11:7]
    int32_t hi = (int32_t)(inst & 0xFE000000) >> 20;
    int32_t lo = (inst >> 7) & 0x1F;
    return hi | lo;
}

// ============================================================================
// Table management
// ============================================================================

void OTC_Decoder::init() {
    table_ = DEFAULT_ISA_TABLE;
}

void OTC_Decoder::load_isa_table(const std::vector<ISA_Entry>& table) {
    table_ = table;
}

// ============================================================================
// Core decode logic
// ============================================================================

int OTC_Decoder::match(uint32_t inst) const {
    uint8_t opc = extract_opcode(inst);
    uint8_t f3  = extract_funct3(inst);

    for (int i = 0; i < (int)table_.size(); i++) {
        const auto& e = table_[i];
        if (opc != e.opcode) continue;
        if ((f3 & e.funct3_mask) != (e.funct3 & e.funct3_mask)) continue;
        return i;
    }
    return -1;
}

DecodedInst OTC_Decoder::decode(uint32_t inst) const {
    DecodedInst d;
    d.raw    = inst;
    d.funct3 = extract_funct3(inst);
    d.funct7 = extract_funct7(inst);
    d.rd     = extract_rd(inst);
    d.rs1    = extract_rs1(inst);
    d.rs2    = extract_rs2(inst);

    int idx = match(inst);
    if (idx < 0) {
        d.valid = false;
        d.op    = OTC_OpType::NOP;
        d.unit  = ExecUnit::NONE;
        return d;
    }

    const auto& e = table_[idx];
    d.op    = e.op;
    d.unit  = e.target;
    d.valid = true;

    // Classify control flags
    d.is_tcu  = (e.target == ExecUnit::TCU || e.target == ExecUnit::SFU);
    d.is_mem  = (e.op == OTC_OpType::LOAD || e.op == OTC_OpType::STORE ||
                 e.op == OTC_OpType::TCU_LOAD || e.op == OTC_OpType::TCU_STORE);
    d.is_sync = (e.op == OTC_OpType::TCU_BARRIER);

    // Operand extraction depends on instruction class
    switch (d.op) {
        case OTC_OpType::TCU_WMMA:
            // rs3 for the C accumulator register could be encoded in funct7
            d.rs3 = (d.funct7 >> 2) & 0x1F;
            // dtype selector from bits within immediate or funct7
            d.dtype = d.funct7 & 0x03;
            break;

        case OTC_OpType::TCU_LOAD:
        case OTC_OpType::LOAD:
            d.imm = extract_imm_I(inst);
            break;

        case OTC_OpType::TCU_STORE:
        case OTC_OpType::STORE:
            d.imm = extract_imm_S(inst);
            break;

        case OTC_OpType::TCU_SP:
        case OTC_OpType::TCU_INT:
        case OTC_OpType::TCU_DP:
            // Data-format configuration instructions
            d.dtype = d.rs2;   // encode format selector in rs2 field
            break;

        case OTC_OpType::TCU_SFU:
            // SFU operation type in funct7
            break;

        case OTC_OpType::TCU_BARRIER:
            // No extra operands needed
            break;

        default:
            break;
    }

    return d;
}

ExecUnit OTC_Decoder::decode_and_route(uint32_t inst, DecodedInst& out) const {
    out = decode(inst);
    return out.unit;
}
