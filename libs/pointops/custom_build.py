from torch.utils.cpp_extension import BuildExtension

class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # Filter out problematic flags
        for extension in self.extensions:
            if hasattr(extension, 'extra_compile_args'):
                if isinstance(extension.extra_compile_args, dict):
                    for key in extension.extra_compile_args:
                        if isinstance(extension.extra_compile_args[key], list):
                            extension.extra_compile_args[key] = [
                                flag for flag in extension.extra_compile_args[key] 
                                if flag != '-mrelax-cmpxchg-loop'
                            ]
                elif isinstance(extension.extra_compile_args, list):
                    extension.extra_compile_args = [
                        flag for flag in extension.extra_compile_args 
                        if flag != '-mrelax-cmpxchg-loop'
                    ]
            
            # Also filter C++ flags passed to the compiler
            self.compiler.compiler_so = [
                flag for flag in self.compiler.compiler_so 
                if flag != '-mrelax-cmpxchg-loop'
            ]
        
        super().build_extensions()