import sconshelper

# create environment
env = Environment()

# use cuda tools
env.Tool('cuda')

# use clang++
env.Replace(CXX='clang++')

# create library
fasteit = sconshelper.Library(name='mpflow', env=env, arguments=ARGUMENTS,
    source_suffix=['cu'],
    CXXFLAGS=[
        '-std=c++11',
        '-O3',
        ],
    NVCCFLAGS=[
        '-Xcompiler',
        '-fpic',
        '-m64',
        '-arch=sm_30',
        '--compiler-options',
        '-O3',
        '--compiler-options',
        '-fno-strict-aliasing',
        '-use_fast_math',
        '--ptxas-options=-v',
        '-lineinfo',
        ],
    NVCCINC=[
        '-Iinclude',
        ],
    LIBS=[
        'cudart',
        'cublas',
        'pthread',
        'dl'
        ],
    )
