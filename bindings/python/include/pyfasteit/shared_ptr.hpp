#ifndef PYFASTEIT_SHARED_PTR_HPP
#define PYFASTEIT_SHARED_PTR_HPP

#ifndef __APPLE__

namespace boost {
    template <
        class T
    >
    const T* get_pointer(const std::shared_ptr<T>& p) {
        return p.get();
    }

    template <
        class T
    >
    T* get_pointer(std::shared_ptr<T>& p) {
        return p.get();
    }
}
#endif

#endif
