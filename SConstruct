import installer

# cuda librarty
class CudaLibrary:
    def __init__(self, name, env):
        self.binary = env.SharedLibrary(
            '$BUILDDIR/lib{}'.format(name),
            [
                Glob('{}/*.cu'.format(env["BUILDDIR"])),
                Glob('{}/*.cpp'.format(env["BUILDDIR"])),
            ],
        )

        # save name
        self.name = name


# create environment
def make_env(builddir,**kwargs):
    env = Environment(**kwargs)

    # add cuda tools
    env.Tool('cuda')

    # set builddir
    env.Append(BUILDDIR=builddir)
    env.VariantDir(env["BUILDDIR"], 'src', duplicate=0)

    return env

# create build environment
env = make_env('build',)

# set c compiler
env.Replace(CXX='clang++')

# set compiler flags
env.Append(
    CXXFLAGS=['-std=c++11'],
    NVCCFLAGS=['-Xcompiler', '-fpic', '-m64', '-arch=sm_30', '--compiler-options', '-fno-strict-aliasing', '-use_fast_math', '--ptxas-options=-v', '-lineinfo'],
    LIBS=['cudart', 'cublas'],
)

# add build target to env
libfasteit = CudaLibrary('fasteit', env)

# create installer
install = installer.Installer(env)

# add library to installer
install.AddLibrary(libfasteit.binary)

# add header to installer
install.AddHeaders('include', '*.h', basedir=libfasteit.name)
