# Run CPUID with EAX = 7, ECX = 0 and return the result in a tuple
# (eax, ebx, ecx, edx)
# https://en.wikipedia.org/wiki/CPUID#EAX=7,_ECX=0:_Extended_Features
CPUInfo = zeros(Int32, 4)
ccall(:jl_cpuidex, Cvoid, (Ptr{Cint}, Cint, Cint), CPUInfo, 7, 0)
# if bit 8 of ebx is set, BMI2 is supported
const UM_HAS_BMI2 = CPUInfo[2] & 0x100 != 0

if UM_HAS_BMI2
    @inline function pdep(x::UInt32, y::UInt32)
        return ccall("llvm.x86.bmi.pdep.32", llvmcall, UInt32, (UInt32, UInt32), x, y)
    end
    @inline function pdep(x::UInt64, y::UInt64)
        return ccall("llvm.x86.bmi.pdep.64", llvmcall, UInt64, (UInt64, UInt64), x, y)
    end
    @inline function pext(x::UInt32, y::UInt32)
        return ccall("llvm.x86.bmi.pext.32", llvmcall, UInt32, (UInt32, UInt32), x, y)
    end
    @inline function pext(x::UInt64, y::UInt64)
        return ccall("llvm.x86.bmi.pext.64", llvmcall, UInt64, (UInt64, UInt64), x, y)
    end
end
