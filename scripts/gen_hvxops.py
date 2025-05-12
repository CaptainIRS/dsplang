from dataclasses import dataclass
import os
import re
import urllib.request

basefile = '/tmp/IntrinsicsHexagon.td'
llvm_version = '20.x'
if not os.path.exists(basefile):
    url = f'https://raw.githubusercontent.com/llvm/llvm-project/refs/heads/release/{llvm_version}/llvm/include/llvm/IR/IntrinsicsHexagon.td'
    urllib.request.urlretrieve(url, basefile)

depfile = '/tmp/IntrinsicsHexagonDep.td'
if not os.path.exists(depfile):
    url = f'https://raw.githubusercontent.com/llvm/llvm-project/refs/heads/release/{llvm_version}/llvm/include/llvm/IR/IntrinsicsHexagonDep.td'
    urllib.request.urlretrieve(url, depfile)

headerfile = '/tmp/hvx_hexagon_protos.h'
if not os.path.exists(headerfile):
    url = f'https://raw.githubusercontent.com/llvm/llvm-project/refs/heads/release/{llvm_version}/clang/lib/Headers/hvx_hexagon_protos.h'
    urllib.request.urlretrieve(url, headerfile)
headerfile_contents = open(headerfile).read()

@dataclass
class DataType:
    def llvmir(self):
        raise NotImplementedError
    def mlir(self):
        raise NotImplementedError

@dataclass
class DataTypePtr(DataType):
    def llvmir(self):
        return '!llvm.ptr'
    def mlir(self):
        return 'LLVM_AnyPointer'

@dataclass
class DataTypeI(DataType):
    type: str
    def llvmir(self):
        return f'{self.type}'
    def mlir(self):
        return f'{self.type.upper()}'

@dataclass
class DataTypeV(DataType):
    length: int
    type: DataTypeI
    def llvmir(self):
        return f'vector<{self.length} x {self.type.llvmir()}>'
    def mlir(self):
        return f'VectorOfLengthAndType<[{self.length}], [{self.type.mlir()}]>'

@dataclass
class Intrinsic:
    name: str
    return_types: list[DataType]
    args: list[DataType]
    imm_index: int | None = None

    def __str__(self):
        return f'Intrinsic("{self.name}", {self.return_types}, {self.args})'

contents = []

hvx_intrinsics_src = open(basefile).readlines() + open(depfile).readlines()
hvx_intrinsics_src = hvx_intrinsics_src[:hvx_intrinsics_src.index('// V79 HVX Instructions.\n')]
for i in range(len(hvx_intrinsics_src)):
    if hvx_intrinsics_src[i].endswith('_Intrinsic_128B;\n'):
        name = hvx_intrinsics_src[i - 1][len('def int_hexagon_V6_'):][:-3]
        line = hvx_intrinsics_src[i].replace('_custom', '').replace('_128B;\n', '')
        hvx_intrinsics_src[i] = f'{line}<"HEXAGON_V6_{name}">;\n'
        print(f'Line: {hvx_intrinsics_src[i]}')
    if 'Hexagon_pred_vload_imm_128B;' in hvx_intrinsics_src[i]:
        name = hvx_intrinsics_src[i].split(':')[0][len('def int_hexagon_V6_'):]
        hvx_intrinsics_src[i] = f'Hexagon_v32i32_i1ptri32_Intrinsic_128B<"HEXAGON_V6_{name}", [ImmArg<ArgIndex<2>>]>;\n'
    if 'Hexagom_pred_vload_upd_128B<1>;' in hvx_intrinsics_src[i]:
        name = hvx_intrinsics_src[i].split(':')[0][len('def int_hexagon_V6_'):]
        hvx_intrinsics_src[i] = f'Hexagon_v32i32ptr_i1ptri32_Intrinsic_128B<"HEXAGON_V6_{name}", [ImmArg<ArgIndex<2>>]>;\n'
    if 'Hexagom_pred_vload_upd_128B<0>;' in hvx_intrinsics_src[i]:
        name = hvx_intrinsics_src[i].split(':')[0][len('def int_hexagon_V6_'):]
        hvx_intrinsics_src[i] = f'Hexagon_v32i32ptr_i1ptri32_Intrinsic_128B<"HEXAGON_V6_{name}">;\n'
    if 'Hexagon_pred_vstore_imm_128B;' in hvx_intrinsics_src[i]:
        name = hvx_intrinsics_src[i].split(':')[0][len('def int_hexagon_V6_'):]
        hvx_intrinsics_src[i] = f'Hexagon_i1ptri32v32i32_Intrinsic_128B<"HEXAGON_V6_{name}", [ImmArg<ArgIndex<2>>]>;\n'
    if 'Hexagon_pred_vstore_upd_128B<1>;' in hvx_intrinsics_src[i]:
        name = hvx_intrinsics_src[i].split(':')[0][len('def int_hexagon_V6_'):]
        hvx_intrinsics_src[i] = f'Hexagon_ptr_i1ptri32v32i32_Intrinsic_128B<"HEXAGON_V6_{name}", [ImmArg<ArgIndex<2>>]>;\n'
    if 'Hexagon_pred_vstore_upd_128B<0>;' in hvx_intrinsics_src[i]:
        name = hvx_intrinsics_src[i].split(':')[0][len('def int_hexagon_V6_'):]
        hvx_intrinsics_src[i] = f'Hexagon_ptr_i1ptri32v32i32_Intrinsic_128B<"HEXAGON_V6_{name}">;\n'
    if 'Hexagon_custom_vms_Intrinsic_128B;' in hvx_intrinsics_src[i]:
        name = hvx_intrinsics_src[i].split(':')[0][len('def int_hexagon_V6_'):]
        hvx_intrinsics_src[i] = f'Hexagon_v128i1ptrv32i32_Intrinsic_128B<"HEXAGON_V6_{name}">;\n'

hvx_intrinsics_src = [line for line in hvx_intrinsics_src if line.startswith('Hexagon_') and '128B' in line]

intrinsic_map = {}
def parse_args(args):
    args_list = []
    while args:
        if args.startswith('ptr'):
            args_list.append(DataTypePtr())
            args = args[3:]
        elif args.startswith('v'):
            match = re.match(r'v\d+i\d+', args)
            if match:
                arg = args[:match.end()]
                length = re.findall(r'v\d+', arg)[0][1:]
                dtype = re.findall(r'i\d+', arg)[0]
                args_list.append(DataTypeV(int(length), DataTypeI(dtype)))
                args = args[match.end():]
            else:
                print(f"Error: {args}")
                break
        elif args.startswith('i'):
            # extract i\d+
            match = re.match(r'i\d+', args)
            if match:
                dtype = re.match(r'i\d+', args).group()
                args_list.append(DataTypeI(dtype))
                args = args[match.end():]
            else:
                print(f"Error: {args}")
                break
    return args_list
def parse_line(line):
    line = line.strip()
    name = line.split('"')[1][len('HEXAGON_V6_'):]
    if 'ImmArg' in line:
        imm_index = int(re.findall(r'ImmArg<ArgIndex<(\d+)>>', line)[0])
    else:
        imm_index = None
    return_args = line.split('Hexagon_')[1].split('_Intrinsic')[0]
    if '_' in return_args:
        return_types = return_args.split('_')[0]
        args = return_args.split('_')[1]
    else:
        return_types = None
        args = return_args
    args_list = parse_args(args)
    if return_types:
        return_types = parse_args(return_types)
    intrinsic = Intrinsic(name, return_types, args_list, imm_index)
    intrinsic_map[name] = intrinsic
    return intrinsic

ll_tablegen_contents = '// GENERATED BY scripts/gen_hvxops.py - DO NOT EDIT MANUALLY\n\n'
for line in hvx_intrinsics_src:
    intrinsic = parse_line(line)
    # tablegen_contents += f'// {intrinsic}\n'
    op_name = intrinsic.name.split('_')
    op_name = ''.join([name.capitalize() for name in op_name])
    op_name = op_name[:-1] + op_name[-1].upper()
    ll_tablegen_contents += f'def {op_name} :\n'
    if intrinsic.return_types:
        if len(intrinsic.return_types) == 1:
            ll_tablegen_contents += f'    LLHVX_OneResultIntrOp<\n'
        else:
            ll_tablegen_contents += f'    LLHVX_TwoResultIntrOp<\n'
    else:
        ll_tablegen_contents += f'    LLHVX_ZeroResultIntrOp<\n'
    ll_tablegen_contents += f'        "{intrinsic.name}"'
    # if intrinsic.return_types:
    #     if len(intrinsic.return_types) == 1:
    #     # if intrinsic.return_types:
    #         ll_tablegen_contents += f',\n'
    #         ll_tablegen_contents += f'        [TypeIs<"res", {intrinsic.return_types[0].mlir()}>]\n'
    #     else:
    #         ll_tablegen_contents += f',\n        ['
    #         for i, return_type in enumerate(intrinsic.return_types):
    #             ll_tablegen_contents += f'TypeIs<"res", {return_type.mlir()}>'
    #             if i != len(intrinsic.return_types) - 1:
    #                 ll_tablegen_contents += f', '
    #         ll_tablegen_contents += f']\n'
    # else:
    ll_tablegen_contents += '\n'
    ll_tablegen_contents += f'    >'
    if intrinsic.args:
        ll_tablegen_contents += ',\n'
        ll_tablegen_contents += f'    Arguments<(ins\n'
        for i, arg in enumerate(intrinsic.args):
            if i == len(intrinsic.args) - 1:
                ll_tablegen_contents += f'        {arg.mlir()}:$arg{i}\n'
            else:
                ll_tablegen_contents += f'        {arg.mlir()}:$arg{i},\n'
        ll_tablegen_contents += f'    )>;\n\n'
    else:
        ll_tablegen_contents += ';\n\n'

with open('include/dsplang/Dialect/LLHVX/IR/LLHVXIntrOpsGen.td', 'w') as f:
    f.write(ll_tablegen_contents)

ll_test_contents = '// RUN: llhvx-translate %s -mlir-to-llvmir -split-input-file | FileCheck %s\n\n'
for line in hvx_intrinsics_src:
    intrinsic = parse_line(line)
    op_name = intrinsic.name.replace('_', '.')
    ariths = []
    if intrinsic.return_types:
        if len(intrinsic.return_types) == 1:
            return_type = intrinsic.return_types[0]
            match return_type:
                case DataTypeV(128, DataTypeI('i1')):
                    ll_test_contents += f'// CHECK-LABEL: define <32 x i32> @{intrinsic.name}\n'
                case _:
                    ll_test_contents += f'// CHECK-LABEL: define {return_type.llvmir().replace("vector", "").replace("!llvm.", "")} @{intrinsic.name}\n'
        else:
            # ll_test_contents += f'// CHECK-LABEL: define {{ {", ".join([return_type.llvmir().replace("vector", "").replace("!llvm.", "") for return_type in intrinsic.return_types])} }} @{intrinsic.name}\n'
            ll_test_contents += f'// CHECK-LABEL: define {intrinsic.return_types[0].llvmir().replace("vector", "").replace("!llvm.", "")} @{intrinsic.name}\n'
    else:
        ll_test_contents += f'// CHECK-LABEL: define void @{intrinsic.name}\n'
    ll_test_contents += f'func.func @{intrinsic.name}('
    for i, arg in enumerate(intrinsic.args):
        if arg == DataTypeI('i1') or (intrinsic.imm_index is not None and i == intrinsic.imm_index):
            if i == len(intrinsic.args) - 1:
                ll_test_contents = ll_test_contents[:-2]
            ariths.append((i, arg))
            continue
        ll_test_contents += f'%{chr(ord('A') + i)} : {arg.llvmir()}'
        if i != len(intrinsic.args) - 1:
            ll_test_contents += f', '
    if intrinsic.return_types:
        return_type = intrinsic.return_types[0]
        match return_type:
            case DataTypeV(128, DataTypeI('i1')):
                for i in range(len(intrinsic.args), len(intrinsic.args) + 2):
                    ll_test_contents += f', %{chr(ord('A') + i)} : vector<32xi32>'
                ll_test_contents += ') -> vector<32xi32> {\n'
            case _:
                ll_test_contents += f') -> {return_type.llvmir()}' + ' {\n'
    else:
        ll_test_contents += f')' + ' {\n'
    if intrinsic.args:
        if intrinsic.return_types:
            if len(intrinsic.return_types) == 1:
                ll_test_contents += f'    // CHECK: call {intrinsic.return_types[0].llvmir().replace("vector", "").replace("!llvm.", "")} @llvm.hexagon.V6.{op_name}(\n'
            else:
                ll_test_contents += f'    // CHECK: call {{ {", ".join([return_type.llvmir().replace("vector", "").replace("!llvm.", "") for return_type in intrinsic.return_types])} }} @llvm.hexagon.V6.{op_name}(\n'
        else:
            ll_test_contents += f'    // CHECK: call void @llvm.hexagon.V6.{op_name}(\n'
        for i, arg in enumerate(intrinsic.args):
            if arg == DataTypeI('i1') or (intrinsic.imm_index is not None and i == intrinsic.imm_index):
                if arg == DataTypeI('i1'):
                    ll_test_contents += f'    // CHECK-SAME: {arg.llvmir().replace("vector", "").replace("!llvm.", "")} true'
                else:
                    ll_test_contents += f'    // CHECK-SAME: {arg.llvmir().replace("vector", "").replace("!llvm.", "")} {{{{[0-9]+}}}}'
            else:
                ll_test_contents += f'    // CHECK-SAME: {arg.llvmir().replace("vector", "").replace("!llvm.", "")} %{{{{[0-9]+}}}}'
            if i == len(intrinsic.args) - 1:
                ll_test_contents += f')\n'
            else:
                ll_test_contents += f',\n'
    else:
        if intrinsic.return_types:
            if len(intrinsic.return_types) == 1:
                return_type = intrinsic.return_types[0]
                match return_type:
                    case DataTypeV(128, DataTypeI('i1')):
                        ll_test_contents += f'    // CHECK: call vector<32xi32> @llvm.hexagon.V6.{op_name}()\n'
                    case _:
                        ll_test_contents += f'    // CHECK: call {return_type.llvmir().replace("vector", "").replace("!llvm.", "")} @llvm.hexagon.V6.{op_name}()\n'
        else:
            ll_test_contents += f'    // CHECK: call void @llvm.hexagon.V6.{op_name}()\n'
    for c in ariths:
        if c[1] == DataTypeI('i1'):
            ll_test_contents += f'    %{chr(ord("A") + c[0])} = arith.constant 1 : i1\n'
        else:
            ll_test_contents += f'    %{chr(ord("A") + c[0])} = arith.constant 1 : i32\n'
    if intrinsic.return_types:
        ll_test_contents += f'    %0 = "llhvx.intr.{op_name.replace(".", "_")}"('
    else:
        ll_test_contents += f'    "llhvx.intr.{op_name.replace(".", "_")}"('
    current_arith = 0
    for i, arg in enumerate(intrinsic.args):
        if arg == DataTypeI('i1') or (intrinsic.imm_index is not None and i == intrinsic.imm_index):
            ll_test_contents += f'%{chr(ord('A') + ariths[current_arith][0])}'
            current_arith += 1
        else:
            ll_test_contents += f'%{chr(ord('A') + i)}'
        if i != len(intrinsic.args) - 1:
            ll_test_contents += f', '
    if intrinsic.return_types:
        if len(intrinsic.return_types) == 1:
            ll_test_contents += f') : ({", ".join([arg.llvmir() for arg in intrinsic.args])}) -> {intrinsic.return_types[0].llvmir()}\n'
        else:
            ll_test_contents += f') : ({", ".join([arg.llvmir() for arg in intrinsic.args])}) -> !llvm.struct<({", ".join([return_type.llvmir() for return_type in intrinsic.return_types])})>\n'
    else:
        ll_test_contents += f') : ({", ".join([arg.llvmir() for arg in intrinsic.args])}) -> ()\n'
    if intrinsic.return_types:
        if len(intrinsic.return_types) == 1:
            return_type = intrinsic.return_types[0]
            match return_type:
                case DataTypeV(128, DataTypeI('i1')):
                    ll_test_contents += f'    %1 = "llhvx.intr.vaddbq_128B"(%0, %{chr(ord('A') + len(intrinsic.args))}, %{chr(ord('A') + len(intrinsic.args) + 1)}) : (vector<128xi1>, vector<32xi32>, vector<32xi32>) -> vector<32xi32>\n'
                    ll_test_contents += f'    return %1 : vector<32xi32>\n'
                case _:
                    ll_test_contents += f'    return %0 : {return_type.llvmir()}\n'
        else:
            ll_test_contents += f'    %1 = llvm.extractvalue %0[0] : !llvm.struct<({", ".join([return_type.llvmir() for return_type in intrinsic.return_types])})>\n'
            ll_test_contents += f'    func.return %1 : {intrinsic.return_types[0].llvmir()}\n'
    else:
        ll_test_contents += f'    return\n'
    ll_test_contents += '}\n\n'

with open('test/Target/LLVMIR/llhvx.mlir', 'w') as f:
    f.write(ll_test_contents)

'''
#if __HVX_ARCH__ >= 60
/* ==========================================================================
   Assembly Syntax:       Rd32=vextract(Vu32,Rs32)
   C Intrinsic Prototype: Word32 Q6_R_vextract_VR(HVX_Vector Vu, Word32 Rs)
   Instruction Type:      LD
   Execution Slots:       SLOT0
   ========================================================================== */

#define Q6_R_vextract_VR(Vu,Rs) __BUILTIN_VECTOR_WRAP(__builtin_HEXAGON_V6_extractw)(Vu,Rs)
#endif /* __HEXAGON_ARCH___ >= 60 */
'''

def get_arg_type(arg):
    arg_type = ''
    full_name, root_type, unsigned, specific_type = arg
    if root_type == 'A':
        arg_type += f'I8MemRef'
    elif root_type == 'V':
        if not specific_type:
            arg_type += f'VectorOfLengthAndType<[1024], [I1]>'
        elif specific_type == 'qf16':
            # arg_type += f'VectorOfQf16'
            arg_type += f'VectorOfLengthAndType<[64], [F16]>'
        elif specific_type == 'qf32':
            # arg_type += f'VectorOfQf32'
            arg_type += f'VectorOfLengthAndType<[32], [F32]>'
        else:
            arg_type += f'VectorOfLengthAndType<'
            if specific_type == 'b':
                arg_type += f'[128], [{"U" if unsigned else "S"}I8]'
            elif specific_type == 'h':
                arg_type += f'[64], [{"U" if unsigned else "S"}I16]'
            elif specific_type == 'w':
                arg_type += f'[32], [{"U" if unsigned else "S"}I32]'
            elif specific_type == 'bf':
                arg_type += f'[64], [BF16]'
            elif specific_type == 'sf':
                arg_type += f'[32], [F32]'
            elif specific_type == 'hf':
                arg_type += f'[64], [F16]'
            arg_type += f'>'
    elif root_type == 'W':
        if not specific_type:
            arg_type += f'VectorOfLengthAndType<[2048], [I1]>'
        elif specific_type == 'qf16':
            arg_type += f'VectorPairOfQf16'
        elif specific_type == 'qf32':
            arg_type += f'VectorPairOfQf32'
        else:
            arg_type += f'VectorOfLengthAndType<'
            if specific_type == 'b':
                arg_type += f'[256], [{"U" if unsigned else "S"}I8]'
            elif specific_type == 'h':
                arg_type += f'[128], [{"U" if unsigned else "S"}I16]'
            elif specific_type == 'w':
                arg_type += f'[64], [{"U" if unsigned else "S"}I32]'
            elif specific_type == 'bf':
                arg_type += f'[128], [BF16]'
            elif specific_type == 'sf':
                arg_type += f'[64], [F32]'
            elif specific_type == 'hf':
                arg_type += f'[128], [F16]'
            arg_type += f'>'
    elif root_type == 'R' or root_type == 'P':
        if not specific_type:
            arg_type += 'AnyI32' if root_type == 'R' else 'AnyI64'
        else:
            el_type = ''
            if unsigned:
                el_type += 'U'
            else:
                el_type += 'S'
            if specific_type == 'b':
                el_type += 'I8'
            elif specific_type == 'h':
                el_type += 'I16'
            elif specific_type == 'w':
                el_type += 'I32'
            el_count = 0
            if specific_type == 'b':
                el_count = 4 if root_type == 'R' else 8
            elif specific_type == 'h':
                el_count = 2 if root_type == 'R' else 4
            elif specific_type == 'w':
                el_count = 1 if root_type == 'R' else 2
            arg_type += f'VectorOfLengthAndType<[{el_count}], [{el_type}]>'
    elif root_type == 'I':
        arg_type += 'AnyI32'
    elif root_type == 'M':
        arg_type += 'AnyI32'
    elif root_type == 'Q':
        arg_type += 'VectorOfLengthAndType<[128], [I1]>'
    elif root_type == 'A':
        arg_type += 'AnyUnrankedMemRef'
    return arg_type
hl_tablegen_contents = '// GENERATED BY scripts/gen_hvxops.py - DO NOT EDIT MANUALLY\n\n'
pdll_contents = '// GENERATED BY scripts/gen_hvxops.py - DO NOT EDIT MANUALLY\n\n'
for hl_block in re.findall(r'#if __HVX_ARCH__ >= (?:6.|73)\n.*?#endif', headerfile_contents, re.DOTALL):
    capture = r'''\#if __HVX_ARCH__ >= (\d+).*?Assembly Syntax:(.*?)C Intrinsic Prototype:(.*?)Instruction Type:(.*?)Execution.*?Slots:(.*?)\=.*\#define (.*?)\#endif'''
    min_arch, asm_syntax, c_intrinsic, instruction_type, execution_slots, define = re.findall(capture, hl_block, re.DOTALL)[0]
    def cleanup_string(s):
        s = s.replace('\\', '')
        return re.sub(r'\s+', ' ', s).strip()
    min_arch = cleanup_string(min_arch)
    asm_syntax = cleanup_string(asm_syntax)
    c_intrinsic = cleanup_string(c_intrinsic)
    instruction_type = cleanup_string(instruction_type)
    execution_slots = cleanup_string(execution_slots)
    define = cleanup_string(define)
    min_arch = int(min_arch)
    # Replace all __BUILTIN_VECTOR_WRAP(abc) with abc in define
    define = re.sub(r'__BUILTIN_VECTOR_WRAP\((.*?)\)', r'\1', define)
    print(f'Assembly Syntax: {asm_syntax}')
    print(f'C Intrinsic Prototype: {c_intrinsic}')
    print(f'Instruction Type: {instruction_type}')
    print(f'Execution Slots: {execution_slots}')
    print(f'Define: {define}')
    print(f'Min Arch: {min_arch}')
    func = c_intrinsic.split('Q6_')[1].split('(')[0]
    has_return = True
    return_name, name, args_name = '', '', ''
    for i, block in enumerate(func.split('_')):
        if block == block.lower():
            name = block
            if len(func.split('_')) > i + 1:
                if func.split('_')[i + 1] == func.split('_')[i + 1].lower():
                    name += '_' + func.split('_')[i + 1]
                    if len(func.split('_')) > i + 2:
                        args_name = '_'.join(func.split('_')[i+2:])
                else:
                    args_name = '_'.join(func.split('_')[i+1:])
            if i == 0:
                has_return = False
            else:
                return_name = func.split('_')[i - 1]
            break
    else:
        print(f'Error(Name): {func}')
    print(f'Names: {name}, {return_name}, {args_name}')
    func = name
    if return_name:
        func = func + '_' + return_name
    if args_name:
        func = func + '_' + args_name
    print(f'Function: {func}')
    datatypes = re.findall(r'(([AVWRIPMQ])(u?)(n|w|bf|sf|hf|b|h|qf16|qf32)?)', func)
    if has_return:
        return_type = datatypes[0]
        args = datatypes[1:]
    else:
        return_type = None
        args = datatypes
    print(f'Return Type: {return_type}')

    if func in ['vmpa_Vh_VhVhVhPh_sat', 'vmpa_Vh_VhVhVuhPuh_sat', 'vmps_Vh_VhVhVuhPuh_sat']:
        args = args[1:]

    c_args = c_intrinsic.split('(')[1].split(')')[0].split(', ')
    c_args = [x.split(' ')[1] for x in c_args if x]
    print(f'C Args: {c_args}')
    if len(c_args) != len(args):
        args = [a for a in args if a[0] != 'I']
    print(f'Args: {args}')

    pdll_src = f'Pattern Rewrite_{func} {{\n'
    pdll_src += f'    let root = op<hlhvx.{func.replace("_", ".")}>({", ".join(f"arg{i}: Value" for i in range(len(args)))});\n\n'
    pdll_src += f'    rewrite root with {{\n'
    define = re.sub(r'\((Q.?)\)', r'\1', define)
    define = ' '.join(define.split()[1:])
    define = define.replace(' ', '')
    argno = len(c_args)
    ssa_vals = {}
    while define:
        calls = re.findall(r'([\w_]*?\([\w\d\+\-,_ ]*?\))', define)
        if not calls:
            break
        if not calls == ['(call0)']:
            print(f'Calls: {calls}')
            for i, call in enumerate(calls):
                callfunc = call.split('(')[0]
                callfunc = callfunc[len('__builtin_HEXAGON_V6_'):]
                callargs = [x for x in call.split('(')[1].split(')')[0].split(',') if x]
                if f'{callfunc}_128B' not in intrinsic_map:
                    print(f'Error(Call): {callfunc}')
                for arg in callargs:
                    try:
                        _ = int(arg)
                        if arg in ssa_vals:
                            continue
                        pdll_src += f'        let arg{argno} = op<arith.constant>() {{ value = attr<"{arg} : i32"> }};\n'
                        ssa_vals[arg] = f'arg{argno}'
                        argno += 1
                    except ValueError:
                        if arg in c_args:
                            if arg in ssa_vals:
                                continue
                            if 'V' in arg:
                                length = 64 if len(arg) == 3 else 32
                                pdll_src += f'        let arg{argno} = op<hlhvx.CastVectorToGeneric>(arg{c_args.index(arg)}) {{ length = attr<"{length} : ui32"> }};\n'
                                ssa_vals[arg] = f'arg{argno}'
                                argno += 1
                            elif 'Q' in arg:
                                ssa_vals[arg] = f'arg{c_args.index(arg)}'
                            elif 'M' in arg:
                                ssa_vals[arg] = f'arg{c_args.index(arg)}'
                            elif 'R' in arg:
                                if args[c_args.index(arg)][3] == 'b':
                                    pdll_src += f'        let arg{argno} = op<hlhvx.CastVectorToRegister>(arg{c_args.index(arg)}) {{ length = attr<"128 : ui32"> }};\n'
                                    ssa_vals[arg] = f'arg{argno}'
                                    argno += 1
                                else:
                                    ssa_vals[arg] = f'arg{c_args.index(arg)}'
                            elif 'I' in arg:
                                ssa_vals[arg] = f'arg{c_args.index(arg)}'
                            else:
                                ssa_vals[arg] = arg
                pdll_src += f'        let arg{argno} = op<llhvx.intr.{callfunc}_128B>('
                for j, arg in enumerate(callargs):
                    pdll_src += ssa_vals[arg]
                    if j != len(callargs) - 1:
                        pdll_src += ', '
                if intrinsic_map[f'{callfunc}_128B'].return_types:
                    pdll_src += f') -> (type<"{intrinsic_map[f"{callfunc}_128B"].return_types[0].llvmir()}">);'
                else:
                    pdll_src += f');'
                pdll_src += f'\n'
                ssa_vals[f'call{i}'] = f'arg{argno}'
                print(ssa_vals)
                argno += 1
                print(f'Callfunc: {callfunc},', intrinsic_map[f'{callfunc}_128B'])
                print(f'Callargs: {callargs}')

        for i, call in enumerate(calls):
            define = define.replace(call, f'call{i}')
    if return_type:
        print(return_type)
        if return_type[0] == 'R':
            pdll_src += f'        let arg{argno} = op<hlhvx.CastIntegerFromGeneric>(arg{argno - 1});\n'
            argno += 1
        else:
            rt = get_arg_type(return_type)
            le = 32
            print(rt)
            if 'VectorOfLengthAndType' in rt:
                newrt = rt.split(', ')[1][1:-2].lower()
                le = int(rt.split(', ')[0][len('VectorOfLengthAndType<['):-1])
                rt = newrt
            elif 'Qf16' in rt:
                rt = 'f16'
                le = 32 if 'Pair' not in rt else 64
            elif 'Qf32' in rt:
                rt = 'f32'
                le = 32 if 'Pair' not in rt else 64
            else:
                print(f'Error(Return): {return_type}')
            pdll_src += f'        let arg{argno} = op<hlhvx.CastVectorFromGeneric>(arg{argno - 1}) {{ length = attr<"{le} : ui32">, type = attr<"\\"{rt}\\""> }}'
            pdll_src += f' -> (type<"vector<{le} x {rt}>">);'
            pdll_src += f'\n'
            argno += 1
    pdll_src += f'        replace root with arg{argno - 1};\n'
    pdll_src += f'    }};\n'
    pdll_src += f'}}\n\n'
    print('PDLL Source:')
    print(pdll_src)
    pdll_contents += pdll_src
        

    # op_name = name
    # op_def = name.capitalize()
    # if return_type:
    #     op_name += '.' + return_type[0]
    #     op_def += '_' + return_type[0]
    # if args:
    #     op_name += '.'
    #     op_def += '_'
    #     for arg in args:
    #         op_name += arg[0]
    #         op_def += arg[0]
    op_name = name.replace('_', '.')
    if return_type:
        op_name += '.' + return_name
    if args:
        op_name += '.' + args_name.replace('_', '.')
    hl_tablegen_contents += f'def HVX_{func} :\n'
    hl_tablegen_contents += f'    HLHVX_Op<\n'
    hl_tablegen_contents += f'        "{op_name}"\n'
    hl_tablegen_contents += f'    >'
    if args or return_type:
        hl_tablegen_contents += f',\n'
    else:
        hl_tablegen_contents += f';\n'
    if args:
        hl_tablegen_contents += f'    Arguments<(ins\n'
        for i, arg in enumerate(args):
            hl_tablegen_contents += f'        '
            hl_tablegen_contents += get_arg_type(arg)
            hl_tablegen_contents += f':$arg{i}'
            if i == len(args) - 1:
                hl_tablegen_contents += f'\n'
            else:
                hl_tablegen_contents += f',\n'
        hl_tablegen_contents += f'    )>'
        if return_type:
            hl_tablegen_contents += f',\n'
        else:
            hl_tablegen_contents += f';\n'
    if return_type:
        hl_tablegen_contents += f'    Results<(outs\n'
        hl_tablegen_contents += f'        '
        hl_tablegen_contents += get_arg_type(return_type)
        hl_tablegen_contents += f':$result'
        hl_tablegen_contents += f'\n'
        hl_tablegen_contents += f'    )>'
        hl_tablegen_contents += f';\n'
    hl_tablegen_contents += '\n'
    print('---')

with open('include/dsplang/Dialect/HLHVX/IR/HLHVXOpsGen.td', 'w') as f:
    f.write(hl_tablegen_contents)
with open('include/dsplang/Conversion/HLHVXToLLHVX/HLHVXToLLHVXGen.pdll', 'w') as f:
    f.write(pdll_contents)
# asm_src = open('/tmp/a.S').read()
# '''
# <vmpy_qf32_qf16_128B>:
#  { 	v1:0.qf32 = vmpy(v0.qf16,v1.qf16)
#    	jumpr r31 } 
#  { 	nop
#    	nop } 

# <vmpy_qf32_sf_128B>:
#  { 	v0.qf32 = vmpy(v0.sf,v1.sf)
#    	jumpr r31 } 
#  { 	nop
#    	nop } 
# '''
# for intrinsic in hvx_intrinsics_src:
#     intrinsic = parse_line(intrinsic)
#     asm = re.findall(rf'<{intrinsic.name}>:(.*?)\n(\n|\Z)', asm_src, re.DOTALL)[0][0]
#     packets = re.findall(r'\{.*?\}', asm, re.DOTALL)
#     num_dt = len(intrinsic.args) + (intrinsic.return_type is not None) - (intrinsic.imm_index is not None)
#     if isinstance(intrinsic.return_type, DataTypeI):
#         num_dt -= 1
#     for arg in intrinsic.args:
#         if isinstance(arg, DataTypeI):
#             num_dt -= 1
#     potential_dts = []
#     for packet in packets:
#         for instruction in packet.split('\n'):
#             if '=' not in instruction:
#                 continue
#             dts = re.findall(r'\.(qf16|qf32|hf|sf|bf|b|ub|h|uh|w|uw)', instruction)
#             regs = re.findall(r'[a-z]\d+\:?\d*\.?[a-z]*', instruction)
#             if len(regs) >= 2 and regs[0] == regs[1]:
#                 dts = dts[1:]
#             if len(dts) == num_dt:
#                 potential_dts.extend(dts)
#                 break
#             if '+=' in instruction:
#                 dts = [dts[0]] + [dts[1]] + [dts[1]] + dts[2:]
#             if 'q_' in intrinsic.name and len(dts) == num_dt - 1:
#                 dts = [dts[0]] + ['any1024'] + dts[1:]
#             if len(dts) == num_dt:
#                 potential_dts.extend(dts)
#                 break
#             print(regs, dts)
#     if intrinsic.name.startswith('pred_'):
#         potential_dts = ['any1024' for _ in range(num_dt)]
#     if potential_dts or num_dt == 0:
#         print(f'Potential data types found for {intrinsic.name}: {potential_dts}')
#     else:
#         print(f'No data types found for {intrinsic.name} ({num_dt})')
#     print(intrinsic)
#     print('\n---\n'.join(packets))
#     print('--' * 20)