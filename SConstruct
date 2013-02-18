import installer

# google test tester
class Tester:
    def __init__(self, env, name):
        # copy environment
        self.env = env.Clone()

        # set googletest flags
        self.env.Append(CPPPATH=[
            'test/gtest/include',
            'test/gmock/include',
            'test/gtest',
            'test/gmock',
            'include'
        ])

        # change VariantDir
        self.env.VariantDir('{}_test'.format(self.env["BUILDDIR"]), 'test', duplicate=0)

        # create binary
        self.binary = self.env.Program(
            '{}_test/test'.format(self.env['BUILDDIR']),
            [
                Glob('{}/*.cu'.format(env["BUILDDIR"])),
                Glob('{}/*.cpp'.format(self.env["BUILDDIR"])),
                Glob('{}_test/*.cpp'.format(self.env["BUILDDIR"])),
                Glob('{}_test/gtest/src/gtest-all.cc'.format(self.env["BUILDDIR"])),
                Glob('{}_test/gmock/src/gmock-all.cc'.format(self.env["BUILDDIR"])),
            ],
        )

        # set alias
        test_alias = self.env.Alias('test', self.binary, self.binary[0].abspath)
        AlwaysBuild(test_alias)

        # set binary to Default
        self.env.Default(None)

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
opts = Options('options.conf', ARGUMENTS)

# set c compiler
env.Replace(CXX='clang++')

# set compiler flags
env.Append(
    CXXFLAGS=['-std=c++11', ],
    NVCCFLAGS=['-Xcompiler', '-fpic', '-m64', '-arch=sm_30', '--compiler-options', '-fno-strict-aliasing', '-use_fast_math', '--ptxas-options=-v', '-lineinfo'],
    LIBS=['cudart', 'cublas', 'pthread', 'dl'],
)

# MacOS specific stuff
if env['PLATFORM'] == 'darwin':
    env['CXXFLAGS'] += ['-stdlib=libc++']
    env['LIBS'] += ['c++']

    # set rpath
    env.Append(LINKFLAGS=['-Xlinker'])
    for rpath in env['RPATH']:
        env.Append(LINKFLAGS=['-rpath', rpath])

# add build target to env
libfasteit = CudaLibrary('fasteit', env)

# add the installer options
installer.AddOptions(opts)
opts.Update(env)

# create tester
tester = Tester(env, libfasteit.name)

# set Default target
env.Default(libfasteit.binary)

# create installer
install = installer.Installer(env)

# add library to installer
install.AddLibrary(libfasteit.binary)

# add header to installer
install.AddHeaders('include', '*.h', basedir=libfasteit.name)
