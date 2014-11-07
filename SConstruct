import sconshelper
from subprocess import Popen, PIPE

# create environment
env = Environment()

# use cuda tools
env.Tool('cuda')

# use clang++
env.Replace(CXX='clang++')

# get current git version
gitproc = Popen(['git', 'describe', '--tags', '--long'], stdout=PIPE)
version = '\\\"{}\\\"'.format(gitproc.communicate()[0].rstrip())

# create library
fasteit = sconshelper.Library(name='mpflow', env=env, arguments=ARGUMENTS,
    source_suffix=['cu'],
    CXXFLAGS=[
        '-std=c++11',
        '-O3',
        ],
    LINKFLAGS=[
        '-O3',
        ],
    CPPPATH=[
        '/usr/local/include',
        ],
    NVCCFLAGS=[
        '-Xcompiler',
        '-fpic',
        '-O3',
        '-use_fast_math',
        '-gencode=arch=compute_30,code=sm_30',
        '--ptxas-options=-v',
        '--compiler-options',
        '-O3',
        ],
    NVCCINC=[
        '-Iinclude',
        ],
    LIBS=[
        'cudart',
        'cublas',
        ],
    CPPDEFINES={
        'GIT_VERSION': version,
        },
    )
