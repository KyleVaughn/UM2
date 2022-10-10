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
