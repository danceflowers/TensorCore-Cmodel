

5.4 存储模块
Amem与Bmem模块
 
    Amem：该模块用于存储经过to_fp9_con转换后的计算结果（fp9格式的A数据）
•	                功能：
•	                按照规定的格式缓存从输入矩阵数据到定制存储器（数据精度为fp9）
•	                为后续的 MAC 计算阵列提供低延迟的数据访问。
•	                设计细节：
•	                每个寄存器Tile分32个子块，每个字块占4个bit单元（4byte）共16byte。（待确认大小）
•	                bit单元为最小存储单元必须存放连续的数据块。
o        加载数据平均分配到32个子快中，如有剩余空间，通过texture拷贝补齐以达到均衡负载的目的。补齐运算通过输入地址映射模块完成，不会产生额外的开销。
•	                支持多路并行访问，以满足 MAC 计算阵列的高吞吐需求。
 
          考虑到后续的MAC阵列主要对矩阵A进行行操作，因此在存储A的元素时，尽可能地把同一行的元素分发到不同的子块（bank），这样同时提取一整行时可以实现并行操作。注意到一共有32个子块，一共有32*4*4*8=4096个比特的空间，而一个8*8fp9的大小为8*8*9=576比特。不过考虑到一个比特单元（cell）（大小为4byte=32bit）中必须存储连续的数据，因此实际的空间利用率为27/32 = 0.84.（这是否意味着一个寄存器tile只有27*4*4*8=3456bit的空间可用？）
          按照这样的说法，每个tile理论上可以存3456/576 = 6 个矩阵。
          以每个tile存4个矩阵为例（因为此时刚好可以分为每个矩阵占8个子块）。基本的Amem模块代码结构如下：
1.	module Amem #(
    parameter MATRICES = 4,
    parameter BANKS = 32,
    parameter BANKS_PER_MATRIX = 8,
    parameter FP9_BITS = 9,
    parameter CELL_BITS = 32
)(
    input wire clk,
    input wire rst_n,
 
// Bank分配：4个矩阵，每个8个bank
localparam BANK_MAP [0:3] = '{
    '{0, 1, 2, 3, 4, 5, 6, 7},      // 矩阵0
    '{8, 9, 10, 11, 12, 13, 14, 15}, // 矩阵1
    '{16, 17, 18, 19, 20, 21, 22, 23}, // 矩阵2  
    '{24, 25, 26, 27, 28, 29, 30, 31}  // 矩阵3
};
 
          对四个矩阵进行完子块分配后，开始根据不同矩阵的不同位置为具体的元素分配空间，首先是bank映射；
       这里的bank映射用到了哈希（散列）。
function automatic [2:0] amem_bank_mapping;
    input [1:0] matrix_id;
    input [2:0] row, col;
    reg [2:0] internal_bank;
    reg [4:0] global_bank;
    begin
        // 在矩阵内的8个bank中分布，确保同一行不同列到不同bank
internal_bank = (row * 3 + col * 5) % BANKS_PER_MATRIX;
        
      // 转换为全局bank编号
        global_bank = BANK_MAP[matrix_id][internal_bank];
        
        amem_bank_mapping = global_bank;
    end
endfunction
 
          其次是比特cell映射（子块中的哪个比特单元），和cell offset（比特单元中的偏移）。
function automatic void amem_cell_mapping;
    input [2:0] row, col;
    output [1:0] cell_index;
    output [1:0] element_slot;
    output [4:0] bit_offset;
    
    reg [5:0] linear_index;
    begin
1.	linear_index = row * 8 + col;
        
        // 每3个元素一个cell
1.	cell_index = (linear_index / 3) % 4;
        
        // cell内的槽位
        element_slot = linear_index % 3;
        
        // 位偏移
        case (element_slot)
            2'd0: bit_offset = 0;   // 位0-8
            2'd1: bit_offset = 9;   // 位9-17
            2'd2: bit_offset = 18;  // 位18-26
        endcase
    end
endfunction
 
        同理，Bmem的模块结构与Amem类似，只不过在矩阵乘法中更多要求提取B的一列（与A提取行对应），因此我们尽量使B中同一列的元素分散到不同的子块中。方法与Amem基本相同，在这里不过多赘述。部分代码（bank映射部分）：
function automatic [2:0] bmem_bank_mapping;
    input [1:0] matrix_id;
    input [2:0] row, col;
    input wire transpose;
    reg [2:0] internal_bank;
    reg [4:0] global_bank;
    begin
        if (transpose) begin
            // 转置时：行列交换，确保转置后的列分布均匀
            internal_bank = (col * 7 + row * 11) % BANKS_PER_MATRIX;
        end else begin
            // 正常模式：确保同一列不同行到不同bank
            internal_bank = (col * 5 + row * 13) % BANKS_PER_MATRIX;
        end
        
        global_bank = BANK_MAP[matrix_id][internal_bank];
        bmem_bank_mapping = global_bank;
    end
endfunction
（基本上与A唯一的区别就是添加了一个transpose判断）。
         
在Amem模块的基础之上我们可以把它扩展为一个写入控制器，其中用到了bank映射、cell映射和偏置设定（amem中的函数）。（下为写入控制器的模块接口）
module amem_write_controller #(
    parameter MATRICES = 4,
    parameter BANKS = 32
)(
    input wire clk,
    input wire rst_n,
    
    // 写入接口
    input wire [8:0] write_data [0:7],
    input wire [2:0] write_row [0:7],
    input wire [2:0] write_col [0:7],
    input wire [1:0] write_matrix [0:7],
    input wire write_valid,
    output wire write_ready,
    
    // Bank写入接口
    output reg [BANKS-1:0] bank_wr_en,
    output reg [1:0] bank_cell_sel [0:BANKS-1],
    output reg [31:0] bank_wr_data [0:BANKS-1],
    output reg [31:0] bank_wr_mask [0:BANKS-1]
);
 
        读取数据时则也是根据具体的矩阵编号、行列位置来计算并访问这个元素所处的子块、比特单元、偏置位置。（下为读取控制器模块接口）
module amem_read_controller #(
    parameter BANKS = 32
)(
    input wire clk,
    input wire rst_n,
    
    // 读取接口
    output reg [8:0] read_data [0:7],
    input wire [2:0] read_row,
    input wire [1:0] read_matrix,
    input wire read_en,
    output reg read_valid,
    
    // Bank读取接口
    input wire [31:0] bank_rd_data [0:BANKS-1],
    output reg [BANKS-1:0] bank_rd_en,
    output reg [1:0] bank_rd_cell_sel [0:BANKS-1]
);
 
 
 
Cmem模块
 
Cmem模块相对特殊，这个模块主要储存经过to_next_con转换后的计算结果，即fp4-fp22格式的C数据。因此如何根据具体的精度进行高效的内存分配时C模块关注的重点。与Cmem模块相关的两个有意思的关键模块是动态打包方案和地址映射策略。
 
计算每个cell能存储的元素数量（动态打包）
function automatic [2:0] calculate_elements_per_cell;
    input [4:0] precision;
    begin
        case (precision)
            4:  calculate_elements_per_cell = 8;  // 32/4 = 8
            5:  calculate_elements_per_cell = 6;  // 32/5 = 6.4 → 6
            6:  calculate_elements_per_cell = 5;  // 32/6 = 5.3 → 5
            7:  calculate_elements_per_cell = 4;  // 32/7 = 4.5 → 4
            8:  calculate_elements_per_cell = 4;  // 32/8 = 4
            9:  calculate_elements_per_cell = 3;  // 32/9 = 3.5 → 3
            10: calculate_elements_per_cell = 3;  // 32/10 = 3.2 → 3
            11: calculate_elements_per_cell = 2;  // 32/11 = 2.9 → 2
            12: calculate_elements_per_cell = 2;  // 32/12 = 2.6 → 2
            13: calculate_elements_per_cell = 2;  // 32/13 = 2.4 → 2
            14: calculate_elements_per_cell = 2;  // 32/14 = 2.2 → 2
            15: calculate_elements_per_cell = 2;  // 32/15 = 2.1 → 2
            16: calculate_elements_per_cell = 2;  // 32/16 = 2
            // 17-22: 每个cell只能存1个元素
            default: calculate_elements_per_cell = 1;
        endcase
    end
endfunction
根据精度生成打包布局
FP4: |元素7|元素6|...|元素0| (每个4位)
FP8: |空闲|元素3|元素2|元素1|元素0|
FP9-FP10: |空闲|元素2|元素1|元素0|
FP11-FP16: |空闲|元素1|元素0|
FP17-FP22: |空闲|元素0|
地址映射策略
 
// Cmem专用地址映射 - 优化整个8×8矩阵的访问
module cmem_address_mapper #(
    parameter BANKS = 32,
    parameter MATRICES = 4
)(
    input wire [2:0] row, col,
    input wire [1:0] matrix_id,
    input wire [4:0] precision_bits,
    output reg [4:0] bank_sel,
    output reg [1:0] cell_sel,
    output reg [4:0] bit_offset,
    output reg [2:0] elements_per_cell
);
 
// 计算每个cell存储的元素数量
function [2:0] get_epc;
    input [4:0] precision;
    begin
        if (precision <= 4) get_epc = 8;
        else if (precision <= 8) get_epc = 4;
        else if (precision <= 10) get_epc = 3;
        else if (precision <= 16) get_epc = 2;
        else get_epc = 1;
    end
endfunction
 
always @(*) begin
    elements_per_cell = get_epc(precision_bits);
    
    // 线性索引
    linear_index = row * 8 + col;
    
    // Bank选择：确保同一矩阵的连续元素分布到不同bank
    bank_sel = (matrix_id * 8 + (linear_index % 8) * 3 + (linear_index / 8) * 5) % BANKS;
    
    // Cell选择：基于元素数量和精度
    cell_sel = (linear_index / elements_per_cell) % 4;
    
    // 位偏移计算
    element_in_cell = linear_index % elements_per_cell;
    bit_offset = element_in_cell * precision_bits;
end
 
endmodule
 
PS：思考与小问题
Texture拷贝补齐以均衡负载是什么意思？
——是用重复的数据来填充bank为了不浪费空间吗？
   ——但是我们存储的时候如果可以优化出更好的映射（更巧妙的哈希函数等），理论上应该数据存储本身就是相对均衡的。
   ——是不是动态bank映射？（deepseek猜测）
// 动态调整映射以避免热点
function automatic [4:0] adaptive_mapping;
    input [2:0] row, col;
    input [1:0] pattern;
    begin
        // 基础映射
        base_bank = (row * 13 + col * 7) % BANKS;
        
        // 如果检测到该bank负载过高，选择替代bank
        if (access_count[base_bank] > 128) begin // 阈值
            // 选择负载较低的替代bank
            for (int i = 1; i < BANKS; i++) begin
                alternative = (base_bank + i) % BANKS;
                if (access_count[alternative] < 64) begin
                    adaptive_mapping = alternative;
                    return;
                end
            end
        end
        
        adaptive_mapping = base_bank;
    end
endfunction
 
或是当检测到负载不均衡时，重新计算映射表（哈希函数）
always @(posedge clk) begin
    // 计算负载不均衡度
load_imbalance = calculate_imbalance(bank_access_count);
    
    if (load_imbalance > THRESHOLD) begin
        remapping_enable <= 1'b1;
        new_bank_mapping <= compute_balanced_mapping(bank_access_count);
    end
end
 
 
3.	Md_data 模块
（1）模块概述
 md_data模块是OpenTensorCore架构中的结果累加存储核心，专门负责存储和管理矩阵乘加运算的最终计算结果。作为计算流水线的终端存储单元，它承担着数据累加、精度转换和结果输出的关键职责。
在整个TensorCore架构中，md_data作为整个计算流水线的数据汇聚点，该模块负责接收来自mm_mul_add模块的计算结果，并以多精度格式（fp4-fp22）进行高效存储和管理。
 
（2）核心功能
md_data模块在整个TensorCore数据流中处于承上启下的关键位置：
数据汇聚中心：汇集所有计算单元的输出结果
累加操作支持：为多轮矩阵运算提供累加存储支持
精度转换桥梁：在不同精度模式间提供数据缓存和格式转换支持
 
   （3）存储层次
存储层次结构
  
存储容量配置
总体容量：32子块 × 32字节/子块 = 1024字节
子块数量：32个独立子块
子块容量：每个子块8个寄存器单元 × 4字节/单元 = 32字节
最小单元：4字节寄存器单元，确保数据连续性
（4）关键技术特性
多精度数据支持
模块支持从fp4到fp22的多种精度格式存储：
fp4：1位符号，2位指数，1位尾数
fp8：E4M3（1+4+3）和E5M2（1+5+2）变体
fp16：1位符号，5位指数，10位尾数
fp22：扩展精度用于高精度累加
 
并行访问机制
为满足MAC计算阵列的高吞吐需求：
多端口设计：支持同时多个读写端口
bank分组：32个子块可独立并行访问
冲突检测：硬件级访问冲突检测和解决
负载均衡：数据均匀分布避免热点
累加操作支持
初始结果写入 → 2. 后续累加更新 → 3. 最终结果输出
 
（5）应用场景分析
深度学习训练
混合精度训练支持：
前向传播: FP16计算 → FP22累加 → 激活输出
反向传播: 梯度计算 → 高精度累加 → 参数更新
模型优势: 内存效率 + 数值稳定性
科学计算
大规模矩阵运算：支持分块矩阵计算，高精度累加保障数值稳定性，并行访问优化提升吞吐量。

