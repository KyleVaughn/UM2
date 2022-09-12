if UM2_HAS_BMI2
    @inline pdep(x::UInt32, y::UInt32) = ccall("llvm.x86.bmi.pdep.32", llvmcall, UInt32, (UInt32, UInt32), x, y)
    @inline pdep(x::UInt64, y::UInt64) = ccall("llvm.x86.bmi.pdep.64", llvmcall, UInt64, (UInt64, UInt64), x, y)
    @inline pext(x::UInt32, y::UInt32) = ccall("llvm.x86.bmi.pext.32", llvmcall, UInt32, (UInt32, UInt32), x, y)
    @inline pext(x::UInt64, y::UInt64) = ccall("llvm.x86.bmi.pext.64", llvmcall, UInt64, (UInt64, UInt64), x, y)
end
